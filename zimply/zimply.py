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

from gevent import monkey, pywsgi

# make sure to do the monkey-patching before loading the falcon package!
monkey.patch_all()

import logging
import pkg_resources
import re

import falcon
from mako.template import Template
from zim_core import ZIMClient, to_bytes

# Python 2.7 workarounds
try:
    from urllib.parse import unquote
except ImportError:
    from urllib import unquote as stdlib_unquote

    # make unquote in Python 2.7 behave as in Python 3
    def unquote(string, encoding="utf-8", errors="replace"):
        if not isinstance(string, bytes):
            raise TypeError("A bytes-like object is required: got '{}'".format(type(string)))
        return stdlib_unquote(string.encode(encoding)).decode(encoding, errors=errors)

#####
# The supporting classes to provide the HTTP server. This includes the template
# and the actual request handler that uses the ZIM file to retrieve the desired
# page, images, CSS, etc.
#####

class ZIMRequestHandler:
    # store the location of the template file in a class variable
    template = None

    def __init__(self, client):
        self.client = client

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
        location = unquote(location, self.client.encoding)
        components = location.split("?")
        navigation_location = None
        is_article = True  # assume an article is requested, for now
        # if trying for the main page ...
        if location in ["/", "/index.htm", "/index.html", "/main.htm", "/main.html"]:
            # ... return the main page as the article
            article = self.client.main_page
            if article is not None:
                navigation_location = "main"
        else:
            # The location is given as domain.com/namespace/url/parts/ ,
            # as used in the ZIM link or, alternatively, as domain.com/page.htm
            splits = location.split("/")
            path = "/".join(splits[1:])

            # get the desired article
            try:
                article = self.client.get_article(path)
                # we have an article when the namespace is A (i.e. not a photo, etc.)
                is_article = (article.namespace == "A")
            except KeyError:
                article = None
                is_article = False

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
                # CAREFUL: the str() in the next line convert unicode to ASCII string
                arguments = re.sub(r"^q=", r"", str(arguments))  # remove the q=
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
                    text = text.decode(encoding=self.client.encoding)
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

                # use the keywords to search the index
                # q = qp.parse(" ".join(keywords))

                weighted_result = []

                if self.client.search_index:
                    weighted_result = self.client.search_index.search(" ".join(keywords))

                # present the results irrespective of the index we used
                if not weighted_result:
                    # ... let the user know there are no results
                    body = "no results found for: " + " <i>" + " ".join(keywords) + "</i>"
                else:
                    for entry in weighted_result:
                        logging.info(str(entry.score) + ": " + str(entry))
                        body += "<a href=\"{}\">{}</a><br />".format(entry.url, entry.title)

        else:  # if we did not achieve success
            response.status = falcon.HTTP_404
            response.content_type = "text/HTML"
            title = "Page 404"
            body = "requested resource not found"

        if not result:  # if the result hasn't been prefilled ...
            result = template.render(location=navigation_location, body=body,
                                     head=head, title=title)  # render template
            response.data = to_bytes(result, encoding=self.client.encoding)
        else:
            # if result is already filled, push it through as-is
            # (i.e. binary resource)
            response.data = result


class ZIMServer:
    def __init__(self, filename, index_file="", template=pkg_resources.resource_filename(__name__, "template.html"),
                 ip_address="", port=9454, encoding="utf-8", *, auto_delete=False):
        # create the object to access the ZIM file
        self.client = ZIMClient(filename, encoding, index_file=index_file, auto_delete=auto_delete)

        # set the template to a class variable of ZIMRequestHandler
        ZIMRequestHandler.template = template

        app = falcon.API()
        main = ZIMRequestHandler(self.client)
        # create a simple sync that forwards all requests; TODO: only allow GET
        app.add_sink(main.on_get, prefix="/")
        _address = "localhost" if ip_address == "" else ip_address
        print("up and running on http://" + _address + ":" + str(port) + "")
        # start up the HTTP server on the desired port
        pywsgi.WSGIServer((ip_address, port), app).serve_forever()

# to start a ZIM server using ZIMply,
# all you need to provide is the location of the ZIM file:
# server = ZIMServer("wiki.zim")

# alternatively, you can specify your own location for the index,
# use a custom template, or change the port:
# server = ZIMServer("wiki.zim", "index.idx", "template.html", 80)

# all arguments can also be named,
# so you can also choose to simply change the port:
# server = ZIMServer("../wiki.zim", port=8080)
