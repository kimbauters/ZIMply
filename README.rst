**ZIMply** is an easy to use, offline reader for `Wikipedia <https://www.wikipedia.org>`__  – or any other ZIM file. When up and running the information can be accessed through any normal browser. **ZIMply** is written entirely in `Python <https://www.python.org>`__ with minimal dependencies and, as the name implies, relies on `ZIM files <http://www.openzim.org/wiki/OpenZIM>`__. Each ZIM file is a bundle containing thousands of articles, images, etc. as found on websites such as `Wikipedia <https://www.wikipedia.org>`__. The format is made popular by `Kiwix <http://www.kiwix.org>`__, which is a program to read such files offline on your device. As indicated, **ZIMply** differs from `Kiwix <http://www.kiwix.org>`__ in that it provides access through the browser. It accomplishes this by running its own HTTP server. This furthermore makes it easy to install **ZIMply** on one device (a *server*, such as a `Raspberry Pi <https://www.raspberrypi.org/products/>`__) and access it on others (the *clients*). To install Python3 on *e.g.* a `Raspbian lite distribution <https://www.raspberrypi.org/downloads/raspbian/>`__ it suffices to install the following packages:

.. code:: bash

    sudo apt-get -qq python3 python3-setuptools python3-dev python3-pip

Once you have Python 3 up and running, the easiest way to install **ZIMply** is through pip:

.. code:: bash

    pip install zstandard zimply

When you have both Python 2.* and Python 3.* installed on your system, you may need to replace `pip` with `pip3` depending on your setup. All you need to do then is download a ZIM file from `this site <https://www.mirrorservice.org/sites/download.kiwix.org/zim/wikipedia/>`__ and use a command such as:

.. code:: bash

    curl -o wiki.zim https://download.kiwix.org/zim/wikipedia_en_climate_change_maxi.zim

All that's left is for you to create your own Python file to start the server:

.. code:: python

    from zimply import ZIMServer
    ZIMServer("wiki.zim")

That is all there is to it. Using the default settings, you can now access your offline Wiki from http://localhost:9454 - spelling out as *:WIKI* on a keypad. To access **ZIMply** from another system, you need to know the IP address of the system that is running **ZIMply**. You can access it by visiting http://ip_address:9454 where you replace `ip_address` with the actual IP address.

To modify the default settings you can call ZIMServer with your desired options:

.. code:: python

    from zimply import ZIMServer
    ZIMServer("wiki.zim", index_file="index.idx", template="template.html", ip_address="192.168.1.200", port=9454, encoding="utf-8")
    # please leave '192.168.1.200' to blank("") to serve the ZIM on localhost, or replace it with your real ip_address

Want to tinker with the code yourself? **ZIMply** depends on `gevent <http://www.gevent.org>`__ (for networking), `falcon <https://falconframework.org>`__ (for the web service), and `mako <http://www.makotemplates.org>`__ (for templating). The easiest way to install these dependencies is by using:

.. code:: bash

    sudo pip install gevent falcon mako

As before, when you have both Python 2.* and Python 3.* installed on your system, you may need to replace `pip` with `pip3` depending on your setup.

Version 2 of **ZIMply** brings a large number of improvements. Many of these improvements have been made possible thanks to the support from the Endless OS Foundation:

* there is support for Python 2.7 to be able to run on even older systems;
* optional support for Xapian indexes is added. In many cases this means the search function works instantly when the ZIM file comes with a bundled Xapian search index. If you don't have Xapian installed, or do not want to install Xapian, **ZIMply** falls back to the original SQLite FTS indexing;
* there is support for the newer FTS5 indexing in SQLite which is much more memory-efficient;
* fallback indexing now runs in the background (so there is no need to wait on indexing to complete), and indexing can be interrupted and continued (so you can close **ZIMply** and continue the next time);
* **ZIMply** is now split in a core component and a server component. The core component only relies on the `lzma` package (with optional support for Xapian). This makes it easier to add support for reading ZIM files in your own projects;
* **ZIMply** now supports a lot of additional functionality, such as opening a random article, or paginating results;
* better support for non-trivial ZIM files, such as wikibooks, which introduce potential namespace collisions;
* support for the new ZIM file format 6,1 namespaces such as in https://mirror.download.kiwix.org/zim/.hidden/dev/beer_stackexchange_com_2021-07.zim .
