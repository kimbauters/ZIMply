# ZIMply is a ZIM reader written entirely in Python 3.
# ZIMply takes its inspiration from the Internet in a Box project,
#  which can be seen in some of the main structures used in this project,
#  yet it has been developed independently and is not considered a fork
#  of the project. For more information on the Internet in a Box project,
#  do have a look at https://github.com/braddockcg/internet-in-a-box .


# Copyright (c) 2016-2021, Kim Bauters, Jim Lemmers, Endless OS Foundation LLC
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS
# BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# The views and conclusions contained in the software and documentation are
# those of the authors and should not be interpreted as representing official
# policies, either expressed or implied, of the FreeBSD Project.

from __future__ import division
from __future__ import print_function

import io
import logging
import os
import random
import sqlite3
import time
from collections import namedtuple
from functools import partial
from hashlib import sha256
from itertools import chain
from struct import Struct, pack, unpack, error as struct_error
from time import sleep

from threading import Thread
from queue import Queue

import zstandard
from math import floor, pow, log

# add Xapian support - if available

try:
    import xapian

    FOUND_XAPIAN = True
except ImportError:
    FOUND_XAPIAN = False

# Python 2.7 workarounds
import sys

IS_PY3 = sys.version_info > (3, 0)

try:
    import lzma
except ImportError:
    # not default, requires backports.lzma
    # https://pypi.org/project/backports.lzma/
    from backports import lzma

try:
    from functools import lru_cache
except ImportError:
    # no Python 2.7 support; make a stub instead
    def lru_cache(**kwargs):
        def wrap(func):
            def wrapped(*args, **wrap_kwargs):
                return func(*args, **wrap_kwargs)

            return wrapped

        return wrap


# custom function to convert to bytes that is compatible both with Python 3.4+ and Python 2.7
def to_bytes(data, encoding):
    if isinstance(data, bytes):
        return data
    elif IS_PY3:
        return bytes(data, encoding)
    else:
        return data.encode(encoding)


#####
# Common error classes
#####
class ZIMException(Exception):
    pass


class ZIMFileUnpackError(Exception):
    pass


class ZIMClientNoFile(ZIMException):
    pass


class ZIMClientInvalidFile(ZIMException):
    pass


#####
# Definition of a number of basic structures/functions to simplify the code
#####

ZERO = pack("B", 0)  # defined for zero terminated fields
Field = namedtuple("Field", ["format", "field_name"])  # a tuple
Article = namedtuple("Article", ["url", "full_url", "title", "data", "namespace", "mimetype", "redirect_to_url"])
Namespace = namedtuple("Namespace", ["count", "start", "end", "namespace"])  # a quadruple

iso639_3to1 = {"ara": "ar", "dan": "da", "nld": "nl", "eng": "en",
               "fin": "fi", "fra": "fr", "deu": "de", "hun": "hu",
               "ita": "it", "nor": "no", "por": "pt", "ron": "ro",
               "rus": "ru", "spa": "es", "swe": "sv", "tur": "tr"}


def read_zero_terminated(file_resource, encoding):
    """
    Retrieve a ZERO terminated string by reading byte by byte until the ending
    ZERO terminated field is encountered.
    :param file_resource: the file to read from
    :param encoding: the encoding used for the file
    :return: the decoded string, up to but not including the ZERO termination
    """
    # read until we find the ZERO termination
    data_buffer = iter(partial(file_resource.read, 1), ZERO)
    # join all the bytes together
    field = b"".join(data_buffer)
    # transform the bytes into a string and return the string
    return field.decode(encoding=encoding, errors="ignore")


def convert_size(size):
    """
    Convert a given size in bytes to a human-readable string of the file size.
    :param size: the size in bytes
    :return: a human-readable string of the size
    """
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    power = int(floor(log(size, 1024)))
    base = pow(1024, power)
    size = round(size / base, 2)
    return "%s %s" % (size, size_name[power])


#####
# Description of the structure of a ZIM file, as of late 2017
# For the full definition: http://www.openzim.org/wiki/ZIM_file_format .
#
# The field format used are the same format definitions as for a Struct:
# https://docs.python.org/3/library/struct.html#format-characters
# Notably, as used by ZIMply, we have:
#   I   unsigned integer (4 bytes)
#   Q   unsigned long long (8 bytes)
#   H   unsigned short (2 bytes)
#   B   unsigned char (1 byte)
#   c   char (1 byte)
#####

HEADER = [  # define the HEADER structure of a ZIM file
    Field("I", "magicNumber"),
    Field("H", "major_version"),
    Field("H", "minor_version"),
    Field("Q", "uuid_low"),
    Field("Q", "uuid_high"),
    Field("I", "articleCount"),
    Field("I", "clusterCount"),
    Field("Q", "urlPtrPos"),
    Field("Q", "titlePtrPos"),
    Field("Q", "clusterPtrPos"),
    Field("Q", "mimeListPos"),
    Field("I", "mainPage"),
    Field("I", "layoutPage"),
    Field("Q", "checksumPos")
]

ARTICLE_ENTRY = [  # define the ARTICLE ENTRY structure of a ZIM file
    Field("H", "mimetype"),
    Field("B", "parameterLen"),
    Field("c", "namespace"),
    Field("I", "revision"),
    Field("I", "clusterNumber"),
    Field("I", "blobNumber")
    # zero terminated url of variable length; not a Field
    # zero terminated title of variable length; not a Field
    # variable length parameter data as per parameterLen; not a Field
]

REDIRECT_ENTRY = [  # define the REDIRECT ENTRY structure of a ZIM file
    Field("H", "mimetype"),
    Field("B", "parameterLen"),
    Field("c", "namespace"),
    Field("I", "revision"),
    Field("I", "redirectIndex")
    # zero terminated url of variable length; not a Field
    # zero terminated title of variable length; not a Field
    # variable length parameter data as per parameterLen; not a Field
]

CLUSTER = [  # define the CLUSTER structure of a ZIM file
    Field("B", "compressionType")
]


#####
# The internal classes used to easily access
# the different structures in a ZIM file.
#####

class Block(object):
    def __init__(self, structure, encoding):
        self._structure = structure
        self._encoding = encoding
        # Create a new Struct object to correctly read the binary data in this
        # block in particular, pass it along that it is a little endian (<),
        # along with all expected fields.
        self._compiled = Struct("<" + "".join(
            [field.format for field in self._structure]))
        self.size = self._compiled.size

    def unpack(self, data_buffer, offset=0):
        # Use the Struct to read the binary data in the buffer
        # where this block appears at the given offset.
        values = self._compiled.unpack_from(data_buffer, offset)
        # Match up each value with the corresponding field in the block
        # and put it in a dictionary for easy reference.
        return {field.field_name: value for value, field in
                zip(values, self._structure)}

    def _unpack_from_file(self, file_resource, offset=None):
        if offset is not None:
            # move the pointer in the file to the specified offset;
            # this is not index 0
            file_resource.seek(offset)
        # read in the amount of data corresponding to the block size
        data_buffer = file_resource.read(self.size)
        # return the values of the fields after unpacking them
        return self.unpack(data_buffer)

    def unpack_from_file(self, file_resource, seek=None):
        # When more advanced behaviour is needed,
        # this method can be overridden by subclassing.
        return self._unpack_from_file(file_resource, seek)


class HeaderBlock(Block):
    def __init__(self, encoding):
        super(HeaderBlock, self).__init__(HEADER, encoding)


class MimeTypeListBlock(Block):
    def __init__(self, encoding):
        super(MimeTypeListBlock, self).__init__("", encoding)

    def unpack_from_file(self, file_resource, offset=None):
        # move the pointer in the file to the specified offset as
        # this is not index 0 when an offset is specified
        if offset is not None:
            file_resource.seek(offset)
        mimetypes = []  # prepare an empty list to store the mimetypes
        while True:
            # get the next zero terminated field
            s = read_zero_terminated(file_resource, self._encoding)
            mimetypes.append(s)  # add the newly found mimetype to the list
            if s == "":  # the last entry must be an empty string
                mimetypes.pop()  # pop the last entry
                return mimetypes  # return the list of mimetypes we found


class ClusterBlock(Block):
    def __init__(self, encoding):
        super(ClusterBlock, self).__init__(CLUSTER, encoding)


@lru_cache(maxsize=32)  # provide an LRU cache for this object
class ClusterData(object):
    def __init__(self, file_resource, offset, encoding):
        self.file = file_resource  # store the file
        self.offset = offset  # store the offset
        cluster_info = ClusterBlock(encoding).unpack_from_file(
            self.file, self.offset)  # Get the cluster fields.
        # Verify whether the cluster has compression
        self.compression = {4: "lzma", 5: "zstd"}.get(cluster_info["compressionType"], False)
        # at the moment, we don't have any uncompressed data
        self.uncompressed = None
        self._decompress()  # decompress the contents as needed
        # Prepare storage to keep track of the offsets
        # of the blobs in the cluster.
        self._offsets = []
        # proceed to actually read the offsets of the blobs in this cluster
        self._read_offsets()

    def _decompress(self, chunk_size=32768):
        if self.compression == "lzma":
            # create a bytes stream to store the uncompressed cluster data
            self.buffer = io.BytesIO()
            decompressor = lzma.LZMADecompressor()  # prepare the decompressor
            # move the file pointer to the start of the blobs as long as we
            # don't reach the end of the stream.
            self.file.seek(self.offset + 1)

            while not decompressor.eof:
                chunk = self.file.read(chunk_size)  # read in a chunk
                data = decompressor.decompress(chunk)  # decompress the chunk
                self.buffer.write(data)  # and store it in the buffer area

        elif self.compression == "zstd":
            # create a bytes stream to store the uncompressed cluster data
            self.buffer = io.BytesIO()
            decompressor = zstandard.ZstdDecompressor().decompressobj()  # prepare the decompressor
            # move the file pointer to the start of the blobs as long as we
            # don't reach the end of the stream.
            self.file.seek(self.offset + 1)
            while True:
                chunk = self.file.read(chunk_size)  # read in a chunk
                try:
                    data = decompressor.decompress(chunk)  # decompress the chunk
                    self.buffer.write(data)  # and store it in the buffer area
                except zstandard.ZstdError:
                    break

    def _source_buffer(self):
        # get the file buffer or the decompressed buffer
        data_buffer = self.buffer if self.compression else self.file
        # move the buffer to the starting position
        data_buffer.seek(0 if self.compression else self.offset + 1)
        return data_buffer

    def _read_offsets(self):
        # get the buffer for this cluster
        data_buffer = self._source_buffer()
        # read the offset for the first blob
        offset0 = unpack("<I", data_buffer.read(4))[0]
        # store this one in the list of offsets
        self._offsets.append(offset0)
        # calculate the number of blobs by dividing the first blob by 4
        number_of_blobs = int(offset0 / 4)
        for idx in range(number_of_blobs - 1):
            # store the offsets to all other blobs
            self._offsets.append(unpack("<I", data_buffer.read(4))[0])

    # return either the blob itself or its offset (when return_offset is set to True)
    def read_blob(self, blob_index, return_offset=False):
        # check if the blob falls within the range
        if blob_index >= len(self._offsets) - 1:
            raise IOError("Blob index exceeds number of blobs available: %s" %
                          blob_index)
        data_buffer = self._source_buffer()  # get the buffer for this cluster
        # calculate the size of the blob
        blob_size = self._offsets[blob_index + 1] - self._offsets[blob_index]
        # move to the position of the blob relative to current position
        data_buffer.seek(self._offsets[blob_index], 1)
        return data_buffer.read(blob_size) if not return_offset else data_buffer.tell()


class DirectoryBlock(Block):
    def __init__(self, structure, encoding):
        super(DirectoryBlock, self).__init__(structure, encoding)

    def unpack_from_file(self, file_resource, seek=None):
        # read the first fields as defined in the ARTICLE_ENTRY structure
        field_values = super(DirectoryBlock, self)._unpack_from_file(file_resource, seek)
        # then read in the url, which is a zero terminated field
        field_values["url"] = read_zero_terminated(file_resource, self._encoding)
        # followed by the title, which is again a zero terminated field
        field_values["title"] = read_zero_terminated(file_resource, self._encoding)
        field_values["namespace"] = field_values["namespace"].decode(encoding=self._encoding, errors="ignore")
        return field_values


class ArticleEntryBlock(DirectoryBlock):
    def __init__(self, encoding):
        super(ArticleEntryBlock, self).__init__(ARTICLE_ENTRY, encoding)


class RedirectEntryBlock(DirectoryBlock):
    def __init__(self, encoding):
        super(RedirectEntryBlock, self).__init__(REDIRECT_ENTRY, encoding)


#####
# Support functions to simplify (1) the uniform creation of a URL
# given a namespace, and (2) searching in the index.
#####

def full_url(namespace, url):
    return namespace + u"/" + url


def split_path(path, assumed_namespace="A", heuristic_split=True):
    """
    split a path into the namespace and a URL
    when a namespace is missing this function returns a configurable default namespace
    as desired this function can apply a heuristic split to distinguish between what is likely a namespace and/or url
    :param path: the path to split into a namespace and a url
    :param assumed_namespace: the default namespace to return if no namespace is found
    :param heuristic_split: use heuristics to identify what is a namespace and what is part of a url
    :return: a pair consisting of the namespace and the url
    """
    splits = path.split("/")
    if len(splits) == 0:
        return assumed_namespace, ""
    elif len(splits) == 1:
        return assumed_namespace, splits[0]
    else:
        if heuristic_split:
            if len(splits[0]) == 1:
                return splits[0], "/".join(splits[1:])
            else:
                return assumed_namespace, "/".join(splits[0:])
        else:
            return splits[0], "/".join(splits[1:])


def binary_search(func, item, front, end):
    logging.debug("performing binary search with boundaries " + str(front) +
                  " - " + str(end))
    found = False
    middle = 0

    # continue as long as the boundaries don't cross and we haven't found it
    while front < end and not found:
        middle = floor((front + end) / 2)  # determine the middle index
        # use the provided function to find the item at the middle index
        found_item = func(middle)
        if found_item == item:
            found = True  # flag it if the item is found
        else:
            if found_item < item:  # if the middle is too early ...
                # move the front index to the middle
                # (+ 1 to make sure boundaries can be crossed)
                front = middle + 1
            else:  # if the middle falls too late ...
                # move the end index to the middle
                # (- 1 to make sure boundaries can be crossed)
                end = middle - 1

    return middle if found else None


class ZIMFileIterator(object):
    def __init__(self, zim_file, start_from=0):
        self._zim_file = zim_file
        self._namespace = self._zim_file.get_namespace_range("A" if zim_file.version <= (6, 0) else "C")
        start = self._namespace.start if self._namespace.start else 0
        self._idx = max(start, start_from)

    def __iter__(self):
        return self

    def __next__(self):
        end = self._namespace.end if self._namespace.end else 0
        if self._idx <= end:
            idx = self._idx
            entry = self._zim_file.read_directory_entry_by_index(idx)
            entry["fullUrl"] = full_url(entry["namespace"], entry["url"])
            self._idx += 1
            return entry["fullUrl"], entry["title"], idx
        else:
            raise StopIteration

    def next(self):
        return self.__next__()


class ZIMFile:
    """
    The main class to access a ZIM file.
    Two important public methods are:
        get_article_by_url(...)
      is used to retrieve an article given its namespace and url.

        get_main_page()
      is used to retrieve the main page article for the given ZIM file.
    """

    def __init__(self, filename, encoding):
        self._filename = filename
        self._enc = encoding
        # open the file as a binary file
        self.file = open(filename, "rb")
        # retrieve the header fields
        try:
            self.header_fields = HeaderBlock(self._enc).unpack_from_file(self.file)
            self.major = int(self.header_fields["major_version"])
            self.minor = int(self.header_fields["minor_version"])
            self.version = (self.major, self.minor)
            self.mimetype_list = MimeTypeListBlock(self._enc).unpack_from_file(self.file,
                                                                               self.header_fields["mimeListPos"])
            # create the object once for easy access
            self.redirectEntryBlock = RedirectEntryBlock(self._enc)

            self.articleEntryBlock = ArticleEntryBlock(self._enc)
            self.clusterFormat = ClusterBlock(self._enc)
        except struct_error:
            raise ZIMFileUnpackError
    
    def copy(self):
        return ZIMFile(self._filename, self._enc)

    def checksum(self, extra_fields=None):
        # create a checksum to uniquely identify this zim file
        # the UUID should be enough, but let's play it safe and also include the other header info
        if not extra_fields:
            extra_fields = {}
        checksum_entries = []
        fields = self.header_fields.copy()
        fields.update(extra_fields)
        # collect all the HEADER values and make sure they are ordered
        for key in sorted(fields.keys()):
            checksum_entries.append("'" + key + "': " + str(fields[key]))
        checksum_message = (", ".join(checksum_entries)).encode("ascii")
        return sha256(checksum_message).hexdigest()

    def _read_offset(self, index, field_name, field_format, length):
        # move to the desired position in the file
        if index != 0xffffffff:
            self.file.seek(self.header_fields[field_name] + int(length * index))

            # and read and return the particular format
            read = self.file.read(length)
            # return unpack("<" + field_format, self.file.read(length))[0]
            return unpack("<" + field_format, read)[0]
        return None

    def _read_url_offset(self, index):
        return self._read_offset(index, "urlPtrPos", "Q", 8)

    def _read_title_offset(self, index):
        return self._read_offset(index, "titlePtrPos", "L", 4)

    def _read_cluster_offset(self, index):
        return self._read_offset(index, "clusterPtrPos", "Q", 8)

    def _read_directory_entry(self, offset):
        """
        Read a directory entry using an offset.
        :return: a DirectoryBlock - either as Article Entry or Redirect Entry
        """
        logging.debug("reading entry with offset " + str(offset))

        self.file.seek(offset)  # move to the desired offset

        # retrieve the mimetype to determine the type of block
        fields = unpack("<H", self.file.read(2))

        # get block class
        if fields[0] == 0xffff:
            directory_block = self.redirectEntryBlock
        else:
            directory_block = self.articleEntryBlock
        # unpack and return the desired Directory Block
        return directory_block.unpack_from_file(self.file, offset)

    def read_directory_entry_by_index(self, index):
        """
        Read a directory entry using an index.
        :return: a DirectoryBlock - either as Article Entry or Redirect Entry
        """
        # find the offset for the given index
        offset = self._read_url_offset(index)
        if offset is not None:
            # read the entry at that offset
            directory_values = self._read_directory_entry(offset)
            # set the index in the list of values
            directory_values["index"] = index
            return directory_values  # and return all these directory values

    # return either the blob itself or its offset (when return_offset is set to True)
    def _read_blob(self, cluster_index, blob_index, return_offset=False):
        # get the cluster offset
        offset = self._read_cluster_offset(cluster_index)
        # get the actual cluster data
        cluster_data = ClusterData(self.file, offset, self._enc)
        # return the data read from the cluster at the given blob index
        return cluster_data.read_blob(blob_index, return_offset=return_offset)

    # return either the article itself or its offset (when return_offset is set to True)
    def _get_article_by_index(self, index, follow_redirect=True, return_offset=False):
        # get the info from the DirectoryBlock at the given index
        entry = self.read_directory_entry_by_index(index)
        if entry is not None:
            # check if we have a Redirect Entry
            if "redirectIndex" in entry.keys():
                # if we follow up on redirects, return the article it is pointing to following all redirects
                if follow_redirect:
                    logging.debug("redirect to " + str(entry["redirectIndex"]))
                    return self._get_article_by_index(entry["redirectIndex"], follow_redirect, return_offset)
                # otherwise, (1) return no data, (2) provide the redirect index as the metadata,
                #            and (3) provide the full URL this entry is redirecting to
                else:
                    next_link = self._get_article_by_index(entry["redirectIndex"], False, return_offset)
                    return None if return_offset else Article(entry["url"],
                                                              full_url(entry["namespace"], entry["url"]),
                                                              # by ZIM definition the URL is the alternative for a title
                                                              entry["title"] if entry["title"] else entry["url"],
                                                              None, entry["namespace"], entry["redirectIndex"],
                                                              next_link.full_url)
            else:  # otherwise, we have an Article Entry
                # get the data and return the Article
                result = self._read_blob(entry["clusterNumber"], entry["blobNumber"], return_offset)
                if return_offset:
                    return result
                else:  # we received the blob back; use it to create an Article object
                    return Article(entry["url"],
                                   full_url(entry["namespace"], entry["url"]),
                                   # by ZIM definition the URL is the alternative for a title
                                   entry["title"] if entry["title"] else entry["url"],
                                   result, entry["namespace"], self.mimetype_list[entry["mimetype"]],
                                   None)
        else:
            return None

    def _get_entry_by_url(self, namespace, url, linear=False):
        if linear:  # if we are performing a linear search ...
            # ... simply iterate over all articles
            for idx in range(self.header_fields["articleCount"]):
                # get the info from the DirectoryBlock at that index
                entry = self.read_directory_entry_by_index(idx)
                # if we found the article ...
                if entry["url"] == url and entry["namespace"] == namespace:
                    # return the DirectoryBlock entry and index of the entry
                    return entry, idx
            # return None, None if we could not find the entry
            return None, None
        else:
            front = middle = 0
            end = self.header_fields["articleCount"]
            title = full_url(namespace, url)
            logging.debug("performing binary search with boundaries " + str(front) + " - " + str(end))
            found = False
            # continue as long as the boundaries don't cross and
            # we haven't found it
            while front <= end and not found:
                middle = (front + end) // 2  # determine the middle index
                entry = self.read_directory_entry_by_index(middle)
                logging.debug("checking " + entry["url"])
                found_title = full_url(entry["namespace"], entry["url"])
                if found_title == title:
                    found = True  # flag it if the item is found
                else:
                    if found_title < title:  # if the middle is too early ...
                        # move the front index to middle
                        # (+ 1 to ensure boundaries can be crossed)
                        front = middle + 1
                    else:  # if the middle falls too late ...
                        # move the end index to middle
                        # (- 1 to ensure boundaries can be crossed)
                        end = middle - 1
            if found:
                # return the tuple with directory entry and index
                # (note the comma before the second argument)
                return self.read_directory_entry_by_index(middle), middle
            return None, None

    def get_article_by_url(self, namespace, url, follow_redirect=True):
        entry, idx = self._get_entry_by_url(namespace, url)  # get the entry
        if idx:  # we found an index and return the article at that index
            return self._get_article_by_index(idx, follow_redirect=follow_redirect)

    def get_article_by_id(self, idx, follow_redirect=True):
        return self._get_article_by_index(idx, follow_redirect=follow_redirect)

    def get_xapian_offset(self, force_title_only=False):
        # identify whether a full-text Xapian index is available
        _, xapian_idx = self._get_entry_by_url("X", "fulltext/xapian" if not force_title_only else "title/xapian")
        full = True and not force_title_only

        # if we did not get a response try a title index instead as fallback option
        if not xapian_idx and not force_title_only:
            _, xapian_idx = self._get_entry_by_url("X", "title/xapian")
            full = False
        logging.info("no Xapian index found" if not xapian_idx else "found Xapian index (full-text: " + str(full) + ")")
        # return the offset if we found either the full-text or Title index, or return None otherwise

        if not xapian_idx:
            return None, False

        return self._get_article_by_index(xapian_idx, follow_redirect=True, return_offset=True), full

    def get_main_page(self):
        """
        Get the main page of the ZIM file.
        """
        main_page = self._get_article_by_index(self.header_fields["mainPage"])
        if main_page is not None:
            return main_page

    def metadata(self):
        """
        Retrieve the metadata attached to the ZIM file.
        :return: a dict with the entry url as key and the metadata as value
        """
        metadata = {}

        metadata_namespace = self.get_namespace_range("M")
        for i in range(metadata_namespace.start, metadata_namespace.end + 1):
            entry = self.read_directory_entry_by_index(i)  # get the entry
            # turn the key to lowercase as per Kiwix standards
            m_name = entry["url"].lower()
            # get the data, which is encoded as an article
            entry = self._get_article_by_index(i)
            if entry:
                metadata[m_name] = self._get_article_by_index(i).data
        return metadata

    def __len__(self):  # retrieve the number of articles in the ZIM file
        result = self.get_namespace_range("A" if self.version <= (6, 0) else "C")
        return result.count

    def get_iterator(self, start_from=0):
        return ZIMFileIterator(self, start_from=start_from)

    def __iter__(self):
        return ZIMFileIterator(self)

    @lru_cache(maxsize=32)  # provide an LRU cache for this object
    def get_namespace_range(self, namespace):
        """
        Retrieve information on a namespace including the number of entries and the start/end index
        :param namespace: the namespace to look for such as "A" or "C"
        :return: a Namespace object with the count, and start/end index of entries (inclusive)
        """
        start_low = 0
        start_high = self.header_fields["articleCount"] - 1
        start = None

        while start_high >= start_low and start is None:
            start_mid = (start_high + start_low) // 2
            entry = self.read_directory_entry_by_index(start_mid)
            before = None
            try:
                before = self.read_directory_entry_by_index(start_mid - 1)
            except struct_error:
                pass

            if entry["namespace"] == namespace and (before is None or before["namespace"] != namespace):
                start = start_mid
            elif entry["namespace"] >= namespace:
                start_high = start_mid - 1
            else:
                start_low = start_mid + 1

        if start is None:
            return Namespace(0, None, None, namespace)

        end_low = start
        end_high = self.header_fields["articleCount"] - 1
        end = None

        while end_high >= end_low and end is None:
            end_mid = (end_high + end_low) // 2
            entry = self.read_directory_entry_by_index(end_mid)
            after = None
            try:
                after = self.read_directory_entry_by_index(end_mid + 1)
            except struct_error:
                pass
            if entry["namespace"] == namespace and (after is None or after["namespace"] != namespace):
                end = end_mid
            elif entry["namespace"] <= namespace:
                end_low = end_mid + 1
            else:
                end_high = end_mid - 1

        if end is None:
            return Namespace(0, None, None, namespace)

        return Namespace(end - start + 1, start, end, namespace)

    def get_articles_range(self):
        return self.get_namespace_range("A" if self.version <= (6, 0) else "C")

    def close(self):
        self.file.close()

    def __exit__(self, *_):
        """
        Ensure the ZIM file is properly closed when the object is destroyed.
        """
        self.close()


#####
# BM25 ranker for ranking search results.
#####


class BM25:
    """
    Implementation of a BM25 ranker; used to determine the score of results
    returned in search queries. More information on Best Match 25 (BM25) can
    be found here: https://en.wikipedia.org/wiki/Okapi_BM25
    """

    def __init__(self, k1=1.2, b=0.75):
        self.k1 = k1  # set the k1 ...
        self.b = b  # ... and b free parameter

    def calculate_scores(self, query, corpus):
        """
        Calculate the BM25 scores for all the documents in the corpus,
        given the query.
        :param query: a tuple containing the words that we're looking for.
        :param corpus: a list of strings, each string corresponding to
                       one result returned based on the query.
        :return: a list of scores (lower is better),
                 in the same order as the documents in the corpus.
        """

        corpus_size = len(corpus)  # total number of documents in the corpus
        query = [term.lower() for term in query]  # force to a lowercase query
        # also turn each document into lowercase
        corpus = [document.lower().split() for document in corpus]

        result = []  # prepare a list to keep the resulting scores
        if corpus_size == 0:
            return result  # nothing to do

        # Determine the average number of words in each document
        # (simply count the number of spaces) store them in a dict with the
        # hash of the document as the key and the number of words as value.
        doc_lens = [len(doc) for doc in corpus]
        avg_doc_len = sum(doc_lens) / corpus_size
        query_terms = []

        for term in query:
            frequency = sum(document.count(term) for document in corpus)
            query_terms.append((term, frequency))

        # calculate the score of each document in the corpus
        for i, document in enumerate(corpus):
            total_score = 0
            for term, frequency in query_terms:  # for every term ...
                # determine the IDF score (numerator and denominator swapped
                # to achieve a positive score)
                idf = log((frequency + 0.5) / (corpus_size - frequency + 0.5))

                # count how often the term occurs in the document itself
                doc_freq = document.count(term)
                doc_k1 = doc_freq * (self.k1 + 1)
                if avg_doc_len == 0:
                    total_score += 0
                else:
                    doc_b = (1 - self.b + self.b * (doc_lens[i] / avg_doc_len))
                    total_score += idf * (doc_k1 / (doc_freq + (self.k1 * doc_b)))

            # once the score for all terms is summed up,
            # add this score to the result list
            result.append(total_score)

        return [-1 * item for item in result]  # "flip" all results so that lower scores are better (matches sqlite)


SearchResult = namedtuple("SearchResult", ["score", "index", "namespace", "url", "title"])  # a quintuple


class SearchIndex(object):
    @property
    def has_search(self):
        """
        :return: Whether or not the search index provides search abilities.
        """
        return False

    def search(self, query, start=0, end=-1, separator=" "):
        """
        Search the index for the given query. Optional arguments allow for pagination and non-standard query formats.
        :param query: the query to search for.
        :param start: the first index of the results to be returned; defaults to 0 to indicate the first result.
        :param end: the last index of the results to be returned; defaults to -1 to indicate all results.
        :param separator: the character(s) separating different elements in the search query, defaults to one space.
        :return: a list of SearchResult objects, sorted by score (highest first).
        """
        return []

    def get_search_results_count(self, query, separator=" "):
        """
        Get the number of search results. Optional argument allows for non-standard query formats.
        :param query: the query to search for.
        :param separator: the character(s) separating different elements in the search query, defaults to one space.
        :return: the number of expected search results.
        """
        return 0

    def suggest(self, query, start=0, end=9, separator=" "):
        """
        Search a (potentially smaller) index for the given query.
        Optional arguments allow for pagination and non-standard query formats.
        :param query: the query to search for.
        :param start: the first index of the results to be returned; defaults to 0 to indicate the first result.
        :param end: the last index of the results to be returned; defaults to -1 to indicate all results.
        :param separator: the character(s) separating different elements in the search query, defaults to one space.
        :return: a list of SearchResult objects, sorted by score (highest first).
        """
        return []

    def get_suggestions_results_count(self, query, separator=" "):
        """
        Get the number of suggestions. Optional argument allows for non-standard query formats.
        :param query: the query to search for.
        :param separator: the character(s) separating different elements in the search query, defaults to one space.
        :return: the number of expected search results.
        """
        return 0


class FTSIndex(SearchIndex):
    def __init__(self, db, level, zim_file):
        super(FTSIndex, self).__init__()
        self.db = db
        self.level = level
        self.zim = zim_file
        self.bm25 = BM25()

    @property
    def has_search(self):
        return True

    def search(self, query, start=0, end=-1, separator=" "):
        logging.info("Searching for the terms '" + query + "' using FTS.")
        keywords = query.split(separator)
        term = "* ".join(keywords) + "*"
        cursor = self.db.cursor()

        # USING FTS5 we can perform pagination as part of the SQL query
        if self.level < 5:
            cursor.execute("SELECT rowid FROM docs WHERE title MATCH ?", (term,))
        else:
            offset = " OFFSET " + str(start) if end != 0 and end >= start else ""
            limit = "" if end == -1 or not offset else " LIMIT " + str(max(0, end - start))
            cursor.execute("SELECT rowid, rank FROM docs WHERE title MATCH ? ORDER BY rank" + limit + offset, (term,))

        results = cursor.fetchall()
        response = []

        if results:
            entries = []
            redirects = []
            scores = []
            for row in results:  # ... iterate over all the results
                # read the directory entry by index (rather than URL)
                entry = self.zim.read_directory_entry_by_index(row[0])
                if self.level >= 5:
                    scores.append(float(row[1]))
                # add the full url to the entry
                if entry.get("redirectIndex"):
                    redirects.append(entry)
                else:
                    entries.append(entry)
            indexes = set(entry["index"] for entry in entries)
            redirects = [entry for entry in redirects if entry["redirectIndex"] not in indexes]

            entries = list(chain(entries, redirects))
            titles = [entry["title"] for entry in entries]
            # calculate the scores or provide identical scores for all recors
            if not self.level >= 5:
                scores = self.bm25.calculate_scores(keywords, titles)
            weighted_result = sorted(zip(scores, entries), reverse=False, key=lambda x: x[0])
            response = [SearchResult(item[0], item[1]["index"], item[1]["namespace"],
                                     item[1]["url"], item[1]["title"]) for item in weighted_result]

        logging.info("Found " + str(len(response)) + " search results.")

        if self.level >= 5:
            return response
        else:
            # NOTE: pagination is a convenience feature for FTS3/4 - all results are fetched to calculate BM25 scores
            return response[start:] if end == -1 else response[start:end + 1]

    def get_search_results_count(self, query, separator=" "):
        keywords = query.split(separator)
        search_for = "* ".join(keywords) + "*"
        cursor = self.db.cursor()
        cursor.execute("SELECT COUNT(rowid) FROM docs WHERE title MATCH ?", (search_for,))
        results = cursor.fetchone()
        return results[0] if results and len(results) > 0 else 0

    def suggest(self, query, start=0, end=9, separator=" "):
        return self.search(query, start, end, separator)

    def get_suggestions_results_count(self, query, separator=" "):
        return self.get_search_results_count(query, separator)


class XapianIndex(SearchIndex):
    def __init__(self, db_offset, language, encoding, zim_filename, zim_version, alt_db_offset=None):
        super(XapianIndex, self).__init__()
        self._xapian_file = open(zim_filename)
        self._xapian_file.seek(db_offset)
        db = xapian.Database(self._xapian_file.fileno())

        alt_db = None
        if alt_db_offset is not None:
            self._alt_xapian_file = open(zim_filename)
            self._alt_xapian_file.seek(alt_db_offset)
            alt_db = xapian.Database(self._alt_xapian_file.fileno())

        self.xapian_index = db
        self.xapian_title_index = alt_db
        self.zim_version = zim_version
        self.language = language
        self.encoding = encoding

    @property
    def has_search(self):
        return True

    def search(self, query, start=0, end=-1, separator=" ", full_index=True, xapian_flags=None):
        if xapian_flags is None:
            xapian_flags = xapian.QueryParser.FLAG_WILDCARD | xapian.QueryParser.FLAG_SPELLING_CORRECTION
        # this supports Xapian flags - for an overview, see:
        # https://xapian.org/docs/apidoc/html/classXapian_1_1QueryParser.html#ae96a58a8de9d219ca3214a5a66e0407e
        search_index = self.xapian_index if full_index or self.xapian_title_index is None else self.xapian_title_index

        parser = xapian.QueryParser()
        parser.set_stemmer(xapian.Stem(self.language))
        # NOTE: the STEM_SOME strategy is not working as expected
        parser.set_stemming_strategy(xapian.QueryParser.STEM_ALL)
        parser.set_default_op(xapian.Query.OP_AND)

        # add a wildcard * at the end of each search term if the wildcard flag is set
        if (xapian_flags & xapian.QueryParser.FLAG_WILDCARD) == xapian.QueryParser.FLAG_WILDCARD:
            splits = query.split(separator)
            query = "* ".join(splits) + "*"
        parser.set_database(search_index)  # needed for some flags such as FLAG_SPELLING_CORRECTION
        query = parser.parse_query(query, xapian_flags)

        # create the enquirer that will do the search
        enquire = xapian.Enquire(search_index)
        enquire.set_query(query)
        end = search_index.get_doccount() if end == -1 else min(end, search_index.get_doccount())
        matches = enquire.get_mset(start, end - start)

        entries = []
        for match in matches:  # ... iterate over all the results
            location = match.document.get_data().decode(encoding=self.encoding)  # the document is the URL
            namespace, url = split_path(location,
                                        assumed_namespace="A" if self.zim_version <= (6, 0) else "C",
                                        heuristic_split=False)

            # beware there be magic numbers - taken from the C++ code of libzim
            title = match.document.get_value(0).decode(encoding=self.encoding)
            idx = match.document.get_value(1).decode(encoding=self.encoding)

            # do some duck typing to make each entry behave like a DirectoryBlock
            entries.append(SearchResult(match.weight, idx, namespace, url, title))
        return sorted(entries, reverse=True, key=lambda x: x.score)

    def get_search_results_count(self, query, separator=" ", full_index=True, xapian_flags=None):
        if xapian_flags is None:
            xapian_flags = xapian.QueryParser.FLAG_WILDCARD | xapian.QueryParser.FLAG_SPELLING_CORRECTION
        search_index = self.xapian_index if full_index or self.xapian_title_index is None else self.xapian_title_index

        parser = xapian.QueryParser()
        parser.set_stemmer(xapian.Stem(self.language))
        # NOTE: the STEM_SOME strategy is not working as expected
        parser.set_stemming_strategy(xapian.QueryParser.STEM_ALL)
        parser.set_default_op(xapian.Query.OP_AND)

        # add a wildcard * at the end of each search term if the wildcard flag is set
        if (xapian_flags & xapian.QueryParser.FLAG_WILDCARD) == xapian.QueryParser.FLAG_WILDCARD:
            splits = query.split(separator)
            query = "* ".join(splits) + "*"
        parser.set_database(search_index)  # needed for some flags such as FLAG_SPELLING_CORRECTION
        query = parser.parse_query(query, xapian_flags)

        # create the enquirer that will do the search
        enquire = xapian.Enquire(search_index)
        enquire.set_query(query)
        matches = enquire.get_mset(0, search_index.get_doccount())
        return matches.size()

    def suggest(self, query, start=0, end=9, separator=" "):
        # always assume the last word is a partial match/wildcard
        xapian_flags = xapian.QueryParser.FLAG_PARTIAL | xapian.QueryParser.FLAG_SPELLING_CORRECTION
        if not self.xapian_title_index:
            return self.search(query, start, end, separator, xapian_flags=xapian_flags)
        else:
            return self.search(query, start, end, separator, full_index=False, xapian_flags=xapian_flags)

    def get_suggestions_results_count(self, query, separator=" "):
        xapian_flags = xapian.QueryParser.FLAG_PARTIAL | xapian.QueryParser.FLAG_SPELLING_CORRECTION
        if not self.xapian_title_index:
            return self.get_search_results_count(query, separator, xapian_flags=xapian_flags)
        else:
            return self.get_search_results_count(query, separator, full_index=False, xapian_flags=xapian_flags)


class ZIMClient:
    def __init__(self, zim_filename, encoding="utf-8", index_file=None, auto_delete=False, enable_search=True):
        """ Create a new ZIM client to easily access the provided ZIM file.
        :param zim_filename: the path to the file to open as a ZIM file.
        :param encoding: the encoding used in the ZIM file which is usually UTF-8 - this is not verified for you!
        :param index_file: the location of where to create an index file if relying on SQLite FTS search.
                           At this location a file *.idx and a file *.chk will be created.
        :param auto_delete: by default the ZIMClient (silently) fails when a SQLite FTS index is opened
                            for which the checksum (of the ongoing indexation) fails. By enabling this
                            option the incorrect index will be deleted and recreated instead.
        :param enable_search: set to False to disable ZIMClient's search functionality.
        :raises:
            ZIMClientNoFile: if zim_filename is recognised as a path to a file.
            ZIMClientInvalidFile: if the file at zim_filename could not be successfully opened as a ZIM file.
        """

        if not os.path.isfile(zim_filename):
            raise ZIMClientNoFile

        try:
            # create the object to access the ZIM file
            self._zim_file = ZIMFile(zim_filename, encoding)
        except ZIMFileUnpackError:
            raise ZIMClientInvalidFile

        # determine the language if set in the ZIM file
        default_iso = to_bytes("eng", encoding=encoding)
        iso639 = self._zim_file.metadata().get("language", default_iso).decode(encoding=encoding, errors="ignore")
        self.language = iso639_3to1.get(iso639, "en")
        version = str(self._zim_file.major) + "," + str(self._zim_file.minor)
        logging.info("ZIM file + (" + str(version) + ") - language: " +
                     self.language + " (ISO639-1), articles: " + str(len(self._zim_file)))

        if not index_file:
            base = os.path.basename(zim_filename)
            name = os.path.splitext(base)[0]
            # name the index file the same as the zim file with a different extension
            index_file = os.path.join(os.path.dirname(zim_filename), name + ".idx")
        logging.info("The index file is determined to be located at " + str(index_file) + ".")

        # set this object to a class variable of ZIMRequestHandler
        self.search_index = None

        if enable_search and not self.search_index:
            self.search_index = self.__get_xapian_search_index(self._zim_file)

        if enable_search and not self.search_index:
            self.search_index = self.__create_search_indexer_thread(
                self._zim_file, index_file, auto_delete=auto_delete
            )

        if not self.search_index:
            print("NULL SEARCH INDEX")
            self.search_index = SearchIndex()
        else:
            print("REAL SEARCH INDEX")

    def __get_xapian_search_index(self, zim_file):
        if not FOUND_XAPIAN:
            return None
            
        xapian_offset, full_index = zim_file.get_xapian_offset()

        if xapian_offset is None:
            return None

        db_offset = xapian_offset

        # try and retrieve the secondary search index for quick suggestions
        alt_offset = None
        if full_index:
            xapian_offset, _ = zim_file.get_xapian_offset(force_title_only=True)
            if xapian_offset is not None:
                alt_offset = xapian_offset

        return XapianIndex(db_offset, self.language, zim_file._enc, zim_file._filename, zim_file.version, alt_offset)

    def __create_search_indexer_thread(self, zim_file, index_file, **kwargs):
        result = Queue()
        thread = CreateFTSThread(result, index_file, zim_file.copy(), **kwargs)
        thread.start()
        index_file = result.get()

        if index_file:
            logging.info("Search index available; continuing.")
            return FTSIndex(sqlite3.connect(index_file), thread.level, zim_file)

        return None

    def _split_path(self, path, heuristic_split):
        return split_path(path,
                          assumed_namespace="A" if self._zim_file.version <= (6, 0) else "C",
                          heuristic_split=heuristic_split)

    def get_article(self, path, follow_redirect=True, robust_namespace=True):
        """
        Retrieve an article based on its local path, e.g. A/Article.html
        :param path: the local path without a leading slash
        :param follow_redirect: whether or not to follow the full redirect chain to the intended article
        :param robust_namespace: whether or not to use a robust namespace method (with a small performance penalty)
        :return: an Article object if it exists
        :throws: KeyError when the path does not exist
        """
        article = None
        if robust_namespace:
            # get the desired article assuming a namespace/url format
            namespace, url = self._split_path(path, heuristic_split=False)
            article = self._zim_file.get_article_by_url(namespace, url, follow_redirect=follow_redirect)

        # rely on a heuristic fallback when no article is found, or immediately use it when no robust result is required
        if not article:
            # get the desired article assuming a namespace/url format where namespace is a single character
            namespace, url = self._split_path(path, heuristic_split=True)
            article = self._zim_file.get_article_by_url(namespace, url, follow_redirect=follow_redirect)

        if not article:
            raise KeyError("There is no resource available at '" + str(path) + "' .")

        return article

    def get_namespace_count(self, namespace):
        return self._zim_file.get_namespace_range(namespace).count

    @property
    def random_article(self):
        namespace = self._zim_file.get_articles_range()
        idx = random.randint(namespace.start, namespace.end)
        return self._zim_file.get_article_by_id(idx)

    @property
    def random_article_url(self):
        article = self.random_article
        return article.namespace + "/" + article.url if article.namespace else article.url

    @property
    def has_search(self):
        return self.search_index.has_search

    def search(self, query, start, end, separator=" "):
        return self.search_index.search(query, start=start, end=end, separator=separator)

    def get_search_results_count(self, query, separator=" "):
        return self.search_index.get_search_results_count(query, separator=separator)

    def suggest(self, query, start=0, end=9, separator=" "):
        return self.search_index.suggest(query, start, end, separator)

    def get_suggestions_results_count(self, query, separator=" "):
        return self.search_index.get_suggestions_results_count(query, separator)

    @property
    def main_page(self):
        return self._zim_file.get_main_page()

    def __exit__(self, *_):
        self._zim_file.close()


class CreateFTSThread(Thread):
    def __init__(self, connect_queue, index_file, zim_file, auto_delete=False):
        super(CreateFTSThread, self).__init__(daemon=True)
        self.connect_queue = connect_queue
        self.index_file = index_file
        self.zim_file = zim_file
        self.auto_delete = auto_delete
        self.level = self._highest_fts_level()

    def run(self):
        try:
            self.safe_run()
        except Exception as exception:
            raise exception
        return

    @staticmethod
    def _get_continuation_checksum(idx, checksum):
        message = (str(idx) + "+" + checksum).encode("ascii")
        return sha256(message).hexdigest()

    def safe_run(self):
        update_every = 10000

        # the index_file is a full path; use it to construct a full path for the checksum .chk file
        base = os.path.basename(self.index_file)
        name = os.path.splitext(base)[0]
        # name the checksum file the same as the index file with a different extension
        checksum_file = os.path.join(os.path.dirname(self.index_file), name + ".chk")

        # there are a number of steps to walk through:
        #  - if the index exists, then calculate the checksum and verify it with the checksum file
        #     - if the checksum matches -> open the index
        #     - if the checksum is wrong -> do not open the file
        #  - if the index does not exist create it as well as the checksum file

        # retrieve the maximum FTS level supported by SQLite
        level = self.level
        if level is None:
            logging.info("No FTS supported - cannot create search index.")
            self.connect_queue.put(None)
        logging.info("Support found for FTS" + str(level) + ".")

        checksum = self.zim_file.checksum({"fts": level})

        checksum_valid = None  # None if not found, True if verified, False if incorrect
        start_idx = -1

        if os.path.isfile(checksum_file):
            checksum_valid = False  # as long as the file exists we must assume that the checksum is tampered with
            with open(checksum_file, "r") as file:
                lines = file.readlines()
                if len(lines) >= 1:
                    checksum_valid = lines[0] == checksum
                    if checksum_valid is False:
                        logging.info("The checksum of the search index does not match the opened ZIM file.")
                if len(lines) >= 2:
                    continuation = lines[1]
                    splits = continuation.split(" ")
                    if len(splits) == 2:
                        try:
                            idx = int(splits[0])
                            cont_checksum = splits[1]
                            if cont_checksum == self._get_continuation_checksum(idx, checksum):
                                start_idx = idx
                        except ValueError:
                            pass

        if checksum_valid is None or (checksum_valid is True and start_idx > -1):
            if checksum_valid is None:
                logging.info("No index was found at " + str(self.index_file) + ", so now creating the index.")
            else:
                logging.info("Continuing the creation of the index starting from ID #" + str(start_idx) + ".")
            logging.info("The index is being created, this can take quite some time! - " + time.strftime("%X %x"))

            created_checksum = False
            created_search_index = False
            try:
                start_idx = max(0, start_idx)

                with open(checksum_file, "w") as file:
                    file.write(checksum + "\n")
                    file.write(str(start_idx) + " " + self._get_continuation_checksum(start_idx, checksum))
                    created_checksum = True

                db = sqlite3.connect(self.index_file)
                created_search_index = True
                cursor = db.cursor()
                # limit memory usage to 64MB
                cursor.execute("PRAGMA CACHE_SIZE = -65536")

                # create a content-less virtual table using full-text search (FTS) and the porter tokenizer
                fts = "fts" + str(level)
                cursor.execute("CREATE VIRTUAL TABLE IF NOT EXISTS docs USING " +
                               str(fts) + "(content='', title, tokenize=porter);")
                db.commit()
                # get an iterator to access all the articles

                logging.info("sending data in queue after table creation")
                self.connect_queue.put(self.index_file)
                sleep(0.05)  # give the queue a bit of time to pick up on the data before continuing

                iterator = self.zim_file.get_iterator(start_from=start_idx)

                unsaved = 0
                for url, title, idx in iterator:  # retrieve articles one by one
                    cursor.execute("INSERT OR REPLACE INTO docs(rowid, title) VALUES (?, ?)", (idx, title))
                    unsaved += 1
                    if unsaved >= update_every:
                        db.commit()
                        with open(checksum_file, "w") as file:
                            file.write(checksum + "\n")
                            file.write(str(idx) + " " + self._get_continuation_checksum(idx, checksum))
                        logging.info("  ... completed intermediate save of index under creation.")
                        unsaved = 0
                # once all articles are added, commit the changes to the database
                db.commit()
                with open(checksum_file, "w") as file:
                    file.write(checksum)

                logging.info("Index creation complete - " + time.strftime("%X %x"))
                db.close()
            except (sqlite3.Error, IOError) as error:
                if isinstance(error, sqlite3.Error):
                    logging.info("Unable to create the search index - unexpected SQLite error.")
                else:
                    logging.info("Unable to write the checksum or the search index.")
                if created_checksum:
                    os.remove(checksum_file)
                if created_search_index:
                    os.remove(self.index_file)
                self.connect_queue.put(None)
        elif checksum_valid is False and self.auto_delete:
            logging.info("... trying to delete the search index so it can be updated.")
            try:
                # first delete the checksum file
                # this prevents the need to recreate the index if the checksum file cannot be deleted
                # and the correct checksum file can be recovered from its corrupted state
                os.remove(checksum_file)
                os.remove(self.index_file)
                self.safe_run()
            except IOError:
                logging.info("... unable to delete the files.")
                self.connect_queue.put(None)
        elif checksum_valid is True:
            logging.info("all data checks out, sending back index file")
            self.connect_queue.put(self.index_file)
        else:
            self.connect_queue.put(None)

    @staticmethod
    def _highest_fts_level():
        # test FTS support in SQLite3; return True, False, or None when only available when loading extension
        def verify_fts_level(level):
            # try to create an FTS table using an in-memory DB, or try to explicitly load the extension
            tmp_db = sqlite3.connect(":memory:")
            try:
                tmp_db.execute("CREATE VIRTUAL TABLE capability USING fts" + str(level) + "(title);")
            except sqlite3.Error:
                try:
                    tmp_db.enable_load_extension(True)
                    tmp_db.load_extension("fts" + str(level))
                except sqlite3.Error:
                    return False
                return None
            finally:
                tmp_db.close()
            return True

        if verify_fts_level(5) is True:
            return 5
        if verify_fts_level(4) is True:
            return 4
        if verify_fts_level(3) is True:
            return 3
        return None
