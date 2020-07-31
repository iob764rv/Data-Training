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

    utc_offset = dt.utcoffset()

    if utc_offset is not None:
        if utc_offset.days < 0:
            sign = '-'
            utc_offset = - utc_offset
        else:
            sign = '+'
        hh, mm = divmod(utc_offset.seconds, 3600)
      
s += "%s%02d%02d" % (sign, hh, mm)
    else:
        s += "Z"
    return s

class OpxRelPackageRestriction(object):
   
    def __init__(self, lower_bound,

                 lower_bound_inclusive,
                 upper_bound,
                 upper_bound_inclusive):
        self.lower_bound = lower_bound
        self.lower_bound_inclusive = lower_bound_inclusive
        self.upper_bound = upper_bound
        self.upper_bound_inclusive = upper_bound_inclusive

    
   def toDebian(self):

        if (self.lower_bound_inclusive and

                self.lower_bound == self.upper_bound and
                self.upper_bound_inclusive):

            return ['=' + self.lower_bound]

        if (not self.lower_bound_inclusive and

                self.lower_bound == self.upper_bound and
                not self.upper_bound_inclusive):

            return ['!=' + self.lower_bound]


        restrictions = list()

        if self.lower_bound is not None:

            if self.lower_bound_inclusive:
                restrictions.append('>=' + self.lower_bound)

            else:
                restrictions.append('>>' + self.lower_bound)

        if self.upper_bound is not None:

            if self.upper_bound_inclusive:
                restrictions.append('<=' + self.upper_bound)

            else:

                restrictions.append('<<' + self.upper_bound)

        return restrictions

   
    def __str__(self):

        # special case equality
        if (self.lower_bound_inclusive and

                self.lower_bound == self.upper_bound and

                self.upper_bound_inclusive):

            return '[' + self.lower_bound + ']'

        # special case inequality

        if (not self.lower_bound_inclusive and

                self.lower_bound == self.upper_bound and
                not self.upper_bound_inclusive):
            return '(' + self.lower_bound + ')'
        s = '[' if self.lower_bound_inclusive else '('

        if self.lower_bound is not None:
            s += self.lower_bound
        s += ','
        if self.upper_bound is not None:
            s += self.upper_bound
        s += ']' if self.upper_bound_inclusive else ')'

        return s

class OpxRelPackage(object):

    def __init__(self, name, restriction):

        self.name = name
        self.restriction = restriction

    @classmethod

    def fromElement(cls, elem):

        if elem.text:
            match = re.match(r'\A([a-zA-Z0-9][a-zA-Z0-9+-.]+)\s*(?:\(\s*(<<|<=|!=|=|>=|>>)\s*([0-9][a-z0-9+-.:~]+)\s*\))?\s*\Z', elem.text)

            if not match:

                raise ValueError("Can't parse version: ->%s<-" % elem.text)

            name = match.group(1)

            relation = match.group(2)
            version = match.group(3)

            restriction = None
            
            if relation:

                if relation == '<<':

                    lower_bound = None
                    lower_bound_inclusive = False
                    upper_bound = version

                    upper_bound_inclusive = False

                elif relation == '<=':

                    lower_bound = None

                    lower_bound_inclusive = False

                    upper_bound = version

                    upper_bound_inclusive = True

                elif relation == '!=':

                    lower_bound = version
                    lower_bound_inclusive = False

                    upper_bound = version

                    lower_bound_inclusive = False

                elif relation == '=':

                    lower_bound = version
                    lower_bound_inclusive = True

                    upper_bound = version
                    lower_bound_inclusive = True

                elif relation == '>=':
                    lower_bound = version
                    lower_bound_inclusive = True
                    upper_bound = None
                    upper_bound_inclusive = False

                elif relation == '>>':
                    lower_bound = version
                    lower_bound_inclusive = True
                    upper_bound = None
                    upper_bound_inclusive = False
                    restriction = OpxRelPackageRestriction(
                    lower_bound,
                    lower_bound_inclusive,
                    upper_bound,
                    upper_bound_inclusive)

            return OpxRelPackage(name, restriction)
            return OpxRelPackage(name, restriction)

        name = elem.get('name')
        version = elem.get('version')
        if not version:
            return OpxRelPackage(name, None)
        match = re.match(r'\A([[(])([0-9][a-z0-9+-.:~]+)?,([0-9][a-z0-9+-.:~]+)?([])])\Z', version)

        if match:
            restriction = OpxRelPackageRestriction(

                match.group(2),
                match.group(1) == '[',
                match.group(3),
                match.group(4) == ']')

            return OpxRelPackage(name, restriction)
 match = re.match(r'\A\[([0-9][a-z0-9+-.:~]+)\]\Z', version)

        if match:

            restriction = OpxRelPackageRestriction(

                match.group(1),

                True,
                match.group(1),
                True)

            return OpxRelPackage(name, restriction)


        # special case inequality

        match = re.match(r'\A\(([0-9][a-z0-9+-.:~]+)\)\Z', version)

        if match:

            restriction = OpxRelPackageRestriction(

                match.group(1),
                False,

                match.group(1),
                False)
            return OpxRelPackage(name, restriction)

        raise ValueError("Can't parse version: ->%s<-" % version)
