from __future__ import with_statement

from distutils.core import setup

import numpy

def get_version():
    
    d={}
    version_line=''
    with open('classy/__init__.py') as fid:
        for line in fid:
            if line.startswith('__version__'):
                version_line=line
    print version_line
    
    exec(version_line,d)
    return d['__version__']
    

setup(
  name = 'classy',
  version=get_version(),
  description="Python Classification",
  author="Brian Blais",
  packages=['classy',],
)


