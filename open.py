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

        match = re.match(r'\A\(([0-9][a-z0-9+-.:~]+)\)\Z', version)

        if match:

            restriction = OpxRelPackageRestriction(

                match.group(1),
                False,
                match.group(1),
                False)
            return OpxRelPackage(name, restriction)

        raise ValueError("Can't parse version: ->%s<-" % version)
      
def toElement(self):
        """
        Return :class:`etree.Element` representing :class:`OpxRelPackage`
        :returns: :class:`etree.Element`

        """
      
        attributes = collections.OrderedDict()
        attributes['name'] = self.name

        if self.restriction:
            attributes['version'] = str(self.restriction)
        return E.package(attributes)

    def toDebian(self):
        """
        Return list of package name+version restrictions in Debian format
        :returns: list of version specifications for this package

        """
        if self.restriction is not None:
            return ["{}({})".format(self.name, x)
                    for x in self.restriction.toDebian()]
        else:
            return [self.name]
         
   def __str__(self):
        """
        Override str method for a pretty format of the data members.
        """
        s = self.name
        if self.restriction is not None:
            s += " "
            s += str(self.restriction)
        return s

class OpxRelPackageList(object):
    """
    Defines a list of packages, each one being an :class:`OpxRelPackage`
    """
   def __init__(self, package_list, no_package_filter=False):
        self.packages = package_list
        self.no_package_filter = no_package_filter
 def fromElement(cls, element):
        """
        Construct :class:`OpxRelPackageList` object from :class:`etree.Element`
        """
        # no_package_filter is local as this is a classmethod
        if element.find('no_package_filter') is not None:
            no_package_filter = True
        else:
            no_package_filter = False
        package_list = []
        for package_elem in element.findall('package'):
            package_list.append(OpxRelPackage.fromElement(package_elem))

        return OpxRelPackageList(package_list, no_package_filter)
   
 def toElement(self):
        """
        Return :class:`etree.Element` representing :class:`OpxRelPackageList`
        :returns: :class:`etree.Element`
        """
        elem = E.package_list()

        if self.no_package_filter:
            elem.append(E.no_package_filter())

        for package in self.packages:
            elem.append(package.toElement())

        return elem
"""class OpxRelPackageSet(object):
    """
    Defines a package set, including a list of packages,
     and where to find/get them.
    """
    def __init__(self, name, kind, default_solver, platform, flavor,
                    package_sources, package_lists):
        self.name = name
        self.kind = kind
        self.default_solver = default_solver
        self.platform = platform
        self.flavor = flavor
        self.package_sources = package_sources
        self.package_lists = package_lists

    @classmethod
    def fromElement(cls, elem):
"""
        Construct :class:`OpxRelPackageSet` object from :class:`etree.Element`
"""       
"""


        name = elem.find('name').text
        kind = elem.find('type').text

        if elem.find('default_solver') is not None:
            default_solver = True
        else:
            default_solver = False

        _tmp = elem.find('platform')
        if _tmp is not None:
            platform = _tmp.text
        else:
            platform = None

        _tmp = elem.find('flavor')
        if _tmp is not None:
            flavor = _tmp.text
        else:
            flavor = None

        package_sources = []
        for package_desc_elem in elem.findall('package_desc'):
            package_sources.append(
                opx_get_packages.OpxPackageSource(
                    package_desc_elem.find('url').text,
                    package_desc_elem.find('distribution').text,
                    package_desc_elem.find('component').text,
                )
            )

        package_lists = []
        for package_list_elem in elem.findall('package_list'):
            package_lists.append(OpxRelPackageList.fromElement(
                                                        package_list_elem))

        return OpxRelPackageSet(name, kind, default_solver, platform,
                                flavor, package_sources, package_lists)
                           

    def toElement(self):
        """
        Return :class:`etree.Element` representing :class:`OpxRelPackageSet`
        :returns: :class:`etree.Element`
       
        """
        

        elem = E.package_set(
            E.name(self.name),
            E.type(self.kind)
        )

        if self.default_solver:
        
            elem.append(E.default_solver())

        if self.platform is not None:
            elem.append(E.platform(self.platform))
            
        if self.flavor is not None:
            elem.append(E.flavor(self.flavor))

        for package_source in self.package_sources:
         

            elem.append(
              E.package_desc(
                 """   E.url(package_source.url),
"""1
                    E.distribution(package_source.distribution),
                    E.component(package_source.component),
                    """2
                )
            )

        elem.extend([package_list.toElement()
                        for package_list in self.package_lists])

"""
        return elem

