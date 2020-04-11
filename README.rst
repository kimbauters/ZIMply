**ZIMply** is an easy to use, offline reader for `Wikipedia <https://www.wikipedia.org>`__  - or any other ZIM file - which provides access to them offline through any ordinary browser. **ZIMply** is written entirely in `Python <https://www.python.org>`__ and, as the name implies, relies on `ZIM files <http://www.openzim.org/wiki/OpenZIM>`__. Each ZIM file is a bundle containing thousands of articles, images, etc. as found on websites such as `Wikipedia <https://www.wikipedia.org>`__. The format is made popular by `Kiwix <http://www.kiwix.org>`__, which is a program to read such files offline on your device. As indicated, **ZIMply** differs from `Kiwix <http://www.kiwix.org>`__ in that it provides access through the browser. It accomplishes this by running its own HTTP server. This furthermore makes it easy to install **ZIMply** on one device (a *server*, such as a `Raspberry Pi <https://www.raspberrypi.org/products/>`__) and access it on others (the *clients*). To install Python3 on a `Raspbian lite distribution <https://www.raspberrypi.org/downloads/raspbian/>`__ it suffices to install the following packages:

.. code:: bash

    sudo apt-get install -qq python3 python3-setuptools python3-dev python3-pip

Once you have Python 3 up and running, the easiest way to install **ZIMply** is through pip:

.. code:: bash

    pip install zimply

When you have both Python 2.* and Python 3.* installed on your system, you may need to replace `pip` with `pip3` depending on your setup. All you need to do then is download a ZIM file from `this site <https://www.mirrorservice.org/sites/download.kiwix.org/zim/wikipedia/>`__ and use a command such as: **(Be careful! Executing the next command downloads the full English Wikipedia, which is a massive file. Instead, replace the url with your desired ZIM file!)**

.. code:: bash

    curl -o wiki.zim https://www.mirrorservice.org/sites/download.kiwix.org/zim/wikipedia/wikipedia_en_all_novid_2017-08.zim

All that's left is for you to create your own Python file to start the server:

.. code:: python

    from zimply import ZIMServer
    ZIMServer("wiki.zim")

That is all there is to it. Using the default settings, you can now access your offline Wiki from http://localhost:9454 - spelling out as :WIKI on a keypad. To access **ZIMply** from another system, you need to know the IP address of the system that is running **ZIMply**. You can access it by visiting http://ip_address:9454 where you replace ip_address with the actual IP address.

*Note:* the first time you run **ZIMply**, it will take care of creating the index to enable searching. **This can take some time**. Unless you see error messages, you can assume that it all works as planned and **ZIMply** will notify you as soon as the index is fully created. The largest ZIM file (the full English Wikipedia) takes about half an hour to index on a core i7 processor, and can take over half a day on a Raspberry Pi Zero. Creating the index is a step that only needs to be done once though, and subsequent restarts of **ZIMply** will only take a matter of seconds. **ZIMply** is heavily optimised, and *will* run comfortably on a Raspberry Pi Zero.

To modify the default settings, simply call ZIMServer with your desired options:

.. code:: python

    from zimply import ZIMServer
    ZIMServer("wiki.zim", index_file="index.idx", template="template.html", ip_address="192.168.1.200", port=9454, encoding="utf-8")
    # please leave '192.168.1.200' to blank("") to serve the ZIM on localhost, or replace it with your real ip_address

Want to tinker with the code yourself? **ZIMply** depends on `gevent <http://www.gevent.org>`__ (for networking), `falcon <https://falconframework.org>`__ (for the web service), and `mako <http://www.makotemplates.org>`__ (for templating). The easiest way to install these dependencies is by using:

.. code:: bash

    sudo pip install gevent falcon mako

As before, when you have both Python 2.* and Python 3.* installed on your system, you may need to replace `pip` with `pip3` depending on your setup.
