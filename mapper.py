#!/usr/bin/env python
"""An advanced Mapper, using Python iterators and generators."""

import sys
import re


def read_input(input):
    for line in input:
        yield line


def main(separator='\t'):
    # input comes from STDIN (standard input)
    data = read_input(sys.stdin)

    for line in data:
        username = line.split(separator)[0]
        tweet = separator.join(line.split(separator)[1:]).rstrip()
        out = '%s%s%s' % (tweet, separator, username)
        print(out)


# how to test locally in bash/linus: cat <input> | python mapper.py
if __name__ == "__main__":
    main()
