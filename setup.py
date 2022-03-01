"""
litGL Littera in Graphics Library
====================================================

- litGL
    Python package for writing text in an OpenGL frame

Authors:
    Nicola Creati, ncreati@inogs.it
    Roberto Vidmar, rvidmar@inogs.it
    2020-2021

Istituto Nazionale di Oceanografia e di Geofisica Sperimentale - OGS
https://www.inogs.it

"""
import os
import sys
import pip
from setuptools import setup, find_packages
import distutils.spawn

PKGNAME = "litGL"
SUBVERSION = "0"
REQFILE = "requirements.txt"
EXECUTABLES = ()
DATAFILES = []
SCRIPTS = ['bin/%s' % sc for sc in EXECUTABLES]
EXTRAS = {
}
# Retrieve version information
exec(open(os.path.join(PKGNAME, "__version__.py")).read())
version = "%s.%s" % (__version__, SUBVERSION)
# Retrieve requirements
with open(REQFILE) as fp:
    requirements = fp.read()

# Check pip version
if int(pip.__version__.split('.')[0]) < 20:
    # pip upgrade needed!
    print("\n\nWARNING!\n    pip version is too old (%s < 20.0.0)\n"
            % pip.__version__)
    raise SystemExit("\n\n Please run 'pip3 install -U pip' and then"
            " run again this command.\n")

#------------------------------------------------------------------------------
# Check for VIRTUAL_ENV
if 'VIRTUAL_ENV' in os.environ:
    bin_path = os.path.join(os.environ.get('VIRTUAL_ENV'),
                       'lib',
                       'python%d.%d' % sys.version_info[:2],
                       'site-packages', PKGNAME, 'bin')
else:
    bin_path = None

# Avoid name clash for scripts
ok = True
conflicting = []
for ex in EXECUTABLES:
    executable = distutils.spawn.find_executable(ex, bin_path)
    if executable:
        ok = False
        for line in open(executable):
            if ("This script belongs to Python package %s" % PKGNAME in line
                    or (PKGNAME in line
                        and 'EASY-INSTALL-DEV-SCRIPT'in line)):
                # This executable belong to this package
                ok = True
                break
        if not ok:
            conflicting.append(executable)
if not ok:
    raise SystemExit("\nWARNING!\n"
            "Installation will overwrite the following files:\n"
            " --> %s\nPlease resolve conflict before retrying.\n"
            "***Installation aborted***" % conflicting)

#==============================================================================
description = __doc__.split('\n')[1:-1][0]
classifiers = """
Development Status ::4 - Beta
Intended Audience :: Developers
Intended Audience :: Science/Research
License :: OSI Approved :: MIT License
Operating System :: POSIX
Programming Language :: Python
Topic :: Scientific/Engineering
Topic :: Software Development :: Libraries
"""

setup(name=PKGNAME,
        version=version,
        install_requires=requirements,
        description=description,
        extras_require=EXTRAS,
        long_description=open("README.md", "r").read(),
        long_description_content_type='text/markdown',
        classifiers=classifiers.split('\n')[1:-1],
        keywords=[PKGNAME, 'OpenGL', ],
        platforms=['POSIX'],
        license='MIT',
        scripts=SCRIPTS,
        include_package_data=True,
        data_files=DATAFILES,
        url='https://bitbucket.org/ncreati/litGL',
        download_url='https://bitbucket.org/ncreati/litGL/',
        author='Nicola Creati',
        author_email='ncreati@inogs.it, rvidmar@inogs.it',
        packages=find_packages(),
        )
