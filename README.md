# ZIMply Core

ZIMply Core is an unstable fork of [ZIMply](https://pypi.org/project/zimply/),
intended for developers who would like to use its pure Python zim file parser
without the additional dependencies.

## Usage

Install zimply-core from pypi:

    pip install zimply-core

Now, in Python code, you can import from zimply_core:

    from zimply_core import ZIMClient
    zim_file = ZIMClient("wikipedia.zim", encoding="utf-8", auto_delete=True)
    print("{name} contains {count} articles".format(
        name=zim_file.get_article("M/Name").data.decode(),
        count=zim_file.get_namespace_count("A")
    ))
