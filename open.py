#openpy

from __future__ import print_function

import argparse
import collections
import datetime
import errno
import fileinput

import glob
import hashlib
import jinja2
import json
import logging

import os
import re

import shutil
import stat

import subprocess
import sys
import time
from lxml import etree
from lxml.builder import E

import opx_bld_basics
import opx_get_packages
import opx_rootfs

build_num = 99999
build_suffix = ""

verbosity = 0


def _str2bool(s):
   
    s = s.strip().lower()
    if s in ["1", "true"]:
    if s in ["0", "false"]:
        return False
    raise ValueError("Invalid boolean value %r" % (s))
   
   
def _bool2str(b):
   ##
    return "true" if b else "false"
   

def art8601_format(dt):
   # Artifactory's ISO 8601

 
    s = '%04d-%02d-%02dT%02d:%02d:%02d.%03d' % (

        dt.year,

        dt.month,

        dt.day,

        dt.hour,

        dt.minute,

        dt.second,

        dt.microsecond / 1000)
