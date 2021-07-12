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


from gevent import monkey, pywsgi
# make sure to do the monkey-patching before loading the falcon package!
monkey.patch_all()

import io
import logging
import lzma
import zstandard
import os
import pkg_resources
import re
import sqlite3
import time
import urllib
from collections import namedtuple
from functools import partial, lru_cache
from math import floor, pow, log
from struct import Struct, pack, unpack
from itertools import chain

# non-standard required packages are gevent and falcon (for its web server),
# as well as and make (for templating)
from mako.template import Template

import falcon

verbose = False

logging.basicConfig(filename="zimply.log", filemode="w",
                    format="%(levelname)s: %(message)s",
                    level=logging.DEBUG if verbose else logging.INFO)

#####
# Definition of a number of basic structures/functions to simplify the code
#####

ZERO = pack("B", 0)  # defined for zero terminated fields
Field = namedtuple("Field", ["format", "field_name"])  # a tuple
Article = namedtuple("Article", ["data", "namespace", "mimetype"])  # a triple

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
    Field("I", "version"),
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

class Block:
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
        super().__init__(HEADER, encoding)


class MimeTypeListBlock(Block):
    def __init__(self, encoding):
        super().__init__("", encoding)

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

    def read_blob(self, blob_index):
        # check if the blob falls within the range
        if blob_index >= len(self._offsets) - 1:
            raise IOError("Blob index exceeds number of blobs available: %s" %
                          blob_index)
        data_buffer = self._source_buffer()  # get the buffer for this cluster
        # calculate the size of the blob
        blob_size = self._offsets[blob_index + 1] - self._offsets[blob_index]
        # move to the position of the blob relative to current position
        data_buffer.seek(self._offsets[blob_index], 1)
        return data_buffer.read(blob_size)


class DirectoryBlock(Block):
    def __init__(self, structure, encoding):
        super().__init__(structure, encoding)

    def unpack_from_file(self, file_resource, seek=None):
        # read the first fields as defined in the ARTICLE_ENTRY structure
        field_values = super()._unpack_from_file(file_resource, seek)
        # then read in the url, which is a zero terminated field
        field_values["url"] = read_zero_terminated(file_resource, self._encoding)
        # followed by the title, which is again a zero terminated field
        field_values["title"] = read_zero_terminated(file_resource, self._encoding)
        field_values["namespace"] = field_values["namespace"].decode(
            encoding=self._encoding, errors="ignore")
        return field_values


class ArticleEntryBlock(DirectoryBlock):
    def __init__(self, encoding):
        super().__init__(ARTICLE_ENTRY, encoding)


class RedirectEntryBlock(DirectoryBlock):
    def __init__(self, encoding):
        super().__init__(REDIRECT_ENTRY, encoding)


#####
# Support functions to simplify (1) the uniform creation of a URL
# given a namespace, and (2) searching in the index.
#####

def full_url(namespace, url):
    return str(namespace) + '/' + str(url)


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
        self._enc = encoding
        # open the file as a binary file
        self.file = open(filename, "rb")
        # retrieve the header fields
        self.header_fields = HeaderBlock(self._enc).unpack_from_file(self.file)
        self.mimetype_list = MimeTypeListBlock(self._enc).unpack_from_file(
            self.file, self.header_fields["mimeListPos"])
        # create the object once for easy access
        self.redirectEntryBlock = RedirectEntryBlock(self._enc)

        self.articleEntryBlock = ArticleEntryBlock(self._enc)
        self.clusterFormat = ClusterBlock(self._enc)

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

    def _read_blob(self, cluster_index, blob_index):
        # get the cluster offset
        offset = self._read_cluster_offset(cluster_index)
        # get the actual cluster data
        cluster_data = ClusterData(self.file, offset, self._enc)
        # return the data read from the cluster at the given blob index
        return cluster_data.read_blob(blob_index)

    def _get_article_by_index(self, index, follow_redirect=True):
        # get the info from the DirectoryBlock at the given index
        entry = self.read_directory_entry_by_index(index)
        if entry is not None:
            # check if we have a Redirect Entry
            if "redirectIndex" in entry.keys():
                # if we follow up on redirects, return the article it is
                # pointing to
                if follow_redirect:
                    logging.debug("redirect to " + str(entry["redirectIndex"]))
                    return self._get_article_by_index(entry["redirectIndex"],
                                                      follow_redirect)
                # otherwise, simply return no data
                # and provide the redirect index as the metadata.
                else:
                    return Article(None, entry["namespace"],
                                   entry["redirectIndex"])
            else:  # otherwise, we have an Article Entry
                # get the data and return the Article
                data = self._read_blob(entry["clusterNumber"],
                                       entry["blobNumber"])
                return Article(data, entry["namespace"],
                               self.mimetype_list[entry["mimetype"]])
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
            end = len(self)
            title = full_url(namespace, url)
            logging.debug("performing binary search with boundaries " +
                          str(front) + " - " + str(end))
            found = False
            # continue as long as the boundaries don't cross and
            # we haven't found it
            while front <= end and not found:
                middle = floor((front + end) / 2)  # determine the middle index
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
            return self._get_article_by_index(
                idx, follow_redirect=follow_redirect)

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
        # iterate backwards over the entries
        for i in range(self.header_fields["articleCount"] - 1, -1, -1):
            entry = self.read_directory_entry_by_index(i)  # get the entry
            if entry["namespace"] == "M":  # check that it is still metadata
                # turn the key to lowercase as per Kiwix standards
                m_name = entry["url"].lower()
                # get the data, which is encoded as an article
                metadata[m_name] = self._get_article_by_index(i)[0]
            else:  # stop as soon as we are no longer looking at metadata
                break
        return metadata

    def __len__(self):  # retrieve the number of articles in the ZIM file
        return self.header_fields["articleCount"]

    def __iter__(self):
        """
        Create an iterator generator to retrieve all articles in the ZIM file.
        :return: a yielded entry of an article, containing its full URL,
                  its title, and the index of the article
        """
        for idx in range(self.header_fields["articleCount"]):
            # get the Directory Entry
            entry = self.read_directory_entry_by_index(idx)
            if entry["namespace"] == "A":
                # add the full url to the entry
                entry["fullUrl"] = full_url(entry["namespace"], entry["url"])
                yield entry["fullUrl"], entry["title"], idx

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
        :return: a list of scores (higher is better),
                 in the same order as the documents in the corpus.
        """

        corpus_size = len(corpus)  # total number of documents in the corpus
        query = [term.lower() for term in query]  # force to a lowercase query
        # also turn each document into lowercase
        corpus = [document.lower().split() for document in corpus]

        # Determine the average number of words in each document
        # (simply count the number of spaces) store them in a dict with the
        # hash of the document as the key and the number of words as value.
        doc_lens = [len(doc) for doc in corpus]
        avg_doc_len = sum(doc_lens) / len(corpus)
        query_terms = []

        for term in query:
            frequency = sum(document.count(term) for document in corpus)
            query_terms.append((term, frequency))

        result = []  # prepare a list to keep the resulting scores

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
                doc_b = (1 - self.b + self.b * (doc_lens[i] / avg_doc_len))
                total_score += idf * (doc_k1 / (doc_freq + (self.k1 * doc_b)))

            # once the score for all terms is summed up,
            # add this score to the result list
            result.append(total_score)

        return result


#####
# The supporting classes to provide the HTTP server. This includes the template
# and the actual request handler that uses the ZIM file to retrieve the desired
# page, images, CSS, etc.
#####

class ZIMRequestHandler:
    # provide for a class variable to store the ZIM file object
    zim = None
    # provide a class variable to store the index file
    reverse_index = None
    # provide another class variable to store the schema for the index file
    schema = None
    # store the location of the template file in a class variable
    template = None
    # the encoding, stored in a class variable, for the ZIM file contents
    encoding = ""

    def __init__(self):
        self.bm25 = BM25()

    def on_get(self, request, response):
        """
        Process a HTTP GET request. An object is this class is created whenever
        an HTTP request is generated. This method is triggered when the request
        is of any type, typically a GET. This method will redirect the user,
        based on the request, to the index/search/correct page, or an error
        page if the resource is unavailable.
        """

        location = request.relative_uri
        # replace the escaped characters by their corresponding string values
        location = urllib.parse.unquote(location)
        components = location.split("?")
        navigation_location = None
        is_article = True  # assume an article is requested, for now
        # if trying for the main page ...
        if location in ["/", "/index.htm", "/index.html",
                        "/main.htm", "/main.html"]:
            # ... return the main page as the article
            article = ZIMRequestHandler.zim.get_main_page()
            if article is not None:
                navigation_location = "main"
        else:
            # The location is given as domain.com/namespace/url/parts/ ,
            # as used in the ZIM link or, alternatively, as domain.com/page.htm
            _, namespace, *url_parts = location.split("/")

            # are we dealing with an address bar request, eg. /article_name.htm
            if len(namespace) > 1:
                url = namespace  # the namespace is then the URL
                namespace = "A"  # and the namespace is an article
            else:
                # combine all the url parts together again
                url = "/".join(url_parts)
            # get the desired article
            article = ZIMRequestHandler.zim.get_article_by_url(namespace, url)
            # we have an article when the namespace is A
            # (i.e. not a photo, etc.)
            is_article = (namespace == "A")

        # from this point forward, "article" refers to an element in the ZIM
        # database whereas is_article refers to a Boolean to indicate whether
        # the "article" is a true article, i.e. a webpage
        success = True  # assume the request succeeded
        search = False  # assume we do not have a search
        keywords = ""  # the keywords to search for

        if not article and len(components) <= 1:
            # there is no article to be retrieved,
            # and there is no ? in the URI to indicate a search
            success = False
        elif len(components) > 1:  # check if URI of the form main?arguments
            # retrieve the arguments part by popping the top of the sequence
            arguments = components.pop()
            # check if arguments starts with ?q= to indicate a proper search
            if arguments.find("q=") == 0:
                search = True  # if so, we have a search
                navigation_location = "search"  # update navigation location
                arguments = re.sub(r"^q=", r"", arguments)  # remove the q=
                keywords = arguments.split("+")  # split all keywords using +
            else:  # if the required ?q= is not found at the start ...
                success = False  # not a search, although we thought it was one

        template = Template(filename=ZIMRequestHandler.template)
        result = body = head = title = ""  # preset all template variables
        if success:  # if successful, i.e. we found the requested resource
            response.status = falcon.HTTP_200  # respond with a success code
            # set the content type based on the article mimetype
            response.content_type = "text/HTML" if search else article.mimetype

            if not navigation_location:  # check if the article location is set
                # if not, default to "browse" (non-search, non-main page)
                navigation_location = "browse"

            if not search:  # if we did not have a search, but a plain article
                if is_article:
                    text = article.data  # we have an actual article
                    # decode its contents into a string using its encoding
                    text = text.decode(encoding=ZIMRequestHandler.encoding)
                    # retrieve the body from the ZIM article
                    m = re.search(r"<body.*?>(.*?)</body>", text, re.S)
                    body = m.group(1) if m else ""
                    # retrieve the head from the ZIM article
                    m = re.search(r"<head.*?>(.*?)</head>", text, re.S)
                    head = m.group(1) if m else ""
                    # retrieve the title from the ZIM article
                    m = re.search(r"<title.*?>(.*?)</title>", text, re.S)
                    title = m.group(1) if m else ""
                    logging.info("accessing the article: " + title)
                else:
                    # just a binary blob, so use it as such
                    result = article.data
            else:  # if we did have a search form
                # show the search query in the title
                title = "search results for >> " + " ".join(keywords)
                logging.info("searching for keywords >> " + " ".join(keywords))
                # load the parser for the given schema
                # qp = QueryParser("title", schema=ZIMRequestHandler.schema)

                # use the keywords to search the index
                # q = qp.parse(" ".join(keywords))

                cursor = ZIMRequestHandler.reverse_index.cursor()
                search_for = "* ".join(keywords) + "*"
                cursor.execute("SELECT docid FROM papers WHERE title MATCH ?",
                               [search_for])

                results = cursor.fetchall()
                if not results:
                    body = "no results found for: " + " <i>" + " ".join(
                        keywords) + "</i>"  # ... let the user know
                else:
                    entries = []
                    redirects = []
                    for row in results:  # ... iterate over all the results
                        # read the directory entry by index (rather than URL)
                        entry = self.zim.read_directory_entry_by_index(row[0])
                        # add the full url to the entry
                        if entry.get("redirectIndex"):
                            redirects.append(entry)
                        else:
                            entries.append(entry)
                    indexes = set(entry["index"] for entry in entries)
                    print(indexes)
                    redirects = [entry for entry in redirects if
                                 entry["redirectIndex"] not in indexes]

                    entries = list(chain(entries, redirects))
                    titles = [entry["title"] for entry in entries]
                    scores = self.bm25.calculate_scores(keywords, titles)
                    weighted_result = sorted(zip(scores, entries),
                                             reverse=True, key=lambda x: x[0])

                    for weight, entry in weighted_result:
                        print(weight, entry)
                        body += "<a href=\"{}\">{}</a><br />".format(
                            entry["url"], entry["title"])

        else:  # if we did not achieve success
            response.status = falcon.HTTP_404
            response.content_type = "text/HTML"
            title = "Page 404"
            body = "requested resource not found"

        if not result:  # if the result hasn't been prefilled ...
            result = template.render(location=navigation_location, body=body,
                                     head=head, title=title)  # render template
            response.data = bytes(result, encoding=ZIMRequestHandler.encoding)
        else:
            # if result is already filled, push it through as-is
            # (i.e. binary resource)
            response.data = result


class ZIMServer:
    def __init__(self, filename, index_file="",
                 template=pkg_resources.resource_filename(
                     __name__, "template.html"),
                 ip_address="", port=9454, encoding="utf-8"):
        # create the object to access the ZIM file
        self._zim_file = ZIMFile(filename, encoding)
        # get the language of the ZIM file and convert it to ISO639_1 or
        # default to "en" if unsupported
        default_iso = bytes("eng", encoding=encoding)
        iso639 = self._zim_file.metadata().get("language", default_iso) \
            .decode(encoding=encoding, errors="ignore")
        lang = iso639_3to1.get(iso639, "en")
        logging.info("A ZIM file in the language " + str(lang) +
                     " (ISO639-1) was found, " +
                     "containing " + str(len(self._zim_file)) + " articles.")
        if not index_file:
            index_file = os.path.join(os.path.dirname(filename), "index.idx")
        logging.info("The index file is determined to be located at " +
                     str(index_file) + ".")

        # set this object to a class variable of ZIMRequestHandler
        ZIMRequestHandler.zim = self._zim_file
        # set the index schema to a class variable of ZIMRequestHandler
        # ZIMRequestHandler.schema = self._schema
        # set (and create) the index to a class variable
        ZIMRequestHandler.reverse_index = self._bootstrap(index_file)
        # set the template to a class variable of ZIMRequestHandler
        ZIMRequestHandler.template = template
        # set the encoding to a class variable of ZIMRequestHandler
        ZIMRequestHandler.encoding = encoding

        app = falcon.API()
        main = ZIMRequestHandler()
        # create a simple sync that forwards all requests; TODO: only allow GET
        app.add_sink(main.on_get, prefix="/")
        _address = "localhost" if ip_address == "" else ip_address
        print("up and running on http://" + _address + ":" + str(port) + "")
        # start up the HTTP server on the desired port
        pywsgi.WSGIServer((ip_address, port), app).serve_forever()

    def _bootstrap(self, index_file):
        if not os.path.exists(index_file):  # check whether the index exists
            logging.info("No index was found at " + str(index_file) +
                         ", so now creating the index.")
            print("Please wait as the index is created, "
                  "this can take quite some time! - " + time.strftime("%X %x"))

            db = sqlite3.connect(index_file)
            cursor = db.cursor()
            # limit memory usage to 64MB
            cursor.execute("PRAGMA CACHE_SIZE = -65536")
            # create a contentless virtual table using full-text search (FTS4)
            # and the porter tokeniser
            cursor.execute("CREATE VIRTUAL TABLE papers "
                           "USING fts4(content='', title, tokenize=porter);")
            # get an iterator to access all the articles
            articles = iter(self._zim_file)

            for url, title, idx in articles:  # retrieve articles one by one
                cursor.execute(
                    "INSERT INTO papers(docid, title) VALUES (?, ?)",
                    (idx, title))  # and add them
            # once all articles are added, commit the changes to the database
            db.commit()

            print("Index created, continuing - " + time.strftime("%X %x"))
            db.close()
        # return an open connection to the SQLite database
        return sqlite3.connect(index_file)

    def __exit__(self, *_):
        self._zim_file.close()

# to start a ZIM server using ZIMply,
# all you need to provide is the location of the ZIM file:
# server = ZIMServer("wiki.zim")

# alternatively, you can specify your own location for the index,
# use a custom template, or change the port:
# server = ZIMServer("wiki.zim", "index.idx", "template.html", 80)

# all arguments can also be named,
# so you can also choose to simply change the port:
# server = ZIMServer("../wiki.zim", port=8080)
