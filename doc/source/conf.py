# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))
import matplotlib
matplotlib.use('Agg')
from sphinx.highlighting import lexers
#import large
#from large.tools.largeinfo import CfgLexer, CfgStyle
#lexers['cobj'] = CfgLexer()

# -- Project information -----------------------------------------------------

project = 'litGL'
copyright = '2020-2021, Nicola Creati, Roberto Vidmar'
author = 'Nicola Creati, Roberto Vidmar'

# Retrieve version information
exec(open(os.path.join("..", "..", "litGL", "__version__.py")).read())
version =  __version__
for line in open(os.path.join("..", "..", "setup.py")).readlines():
    if "SUBVERSION" in line:
        exec(line)
        release = "%s.%s" % (version, SUBVERSION)
        del SUBVERSION
        break
#pygments_style = 'xcode'
#pygments_style = 'vs'
#pygments_style = 'trac'
#pygments_style = 'tango'
#pygments_style = 'stata'
#pygments_style = 'sas'
#pygments_style = 'rrt' #
#pygments_style = 'rainbow_dash'
#pygments_style = 'perldoc'
#pygments_style = 'native' #
#pygments_style = 'murphy'
#pygments_style = 'lovelace'
#pygments_style = 'igor'
#pygments_style = 'fruity'
#pygments_style = 'friendly'
#pygments_style = 'emacs'
#pygments_style = 'default'
#pygments_style = 'colorful'
#pygments_style = 'bw'
#pygments_style = 'autumn'
#pygments_style = 'arduino'
#pygments_style = 'algol_nu'
#pygments_style = 'algol'
#pygments_style = 'abap'
#pygments_style = 'pastie' #
#pygments_style = 'manni' # Carino, ma no highlighting
#pygments_style = 'rrt' #
#pygments_style = 'borland'
def pygments_monkeypatch_style(mod_name, cls):
    import sys
    import pygments.styles
    cls_name = cls.__name__
    mod = type(__import__("os"))(mod_name)
    setattr(mod, cls_name, cls)
    setattr(pygments.styles, mod_name, mod)
    sys.modules["pygments.styles." + mod_name] = mod
    from pygments.styles import STYLE_MAP
    STYLE_MAP[mod_name] = mod_name + "::" + cls_name


#pygments_monkeypatch_style("cfg_style", CfgStyle)
#pygments_style = "cfg_style"
#pygments_style = 'monokai'
pygments_style = 'sphinx'

# -- General configuration ---------------------------------------------------
extensions = [
        'sphinx.ext.napoleon', #1st place!
        'sphinx.ext.autodoc',
        #'sphinx.ext.autosummary', !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        'sphinx.ext.viewcode',

        'sphinx.ext.inheritance_diagram',
        'sphinx.ext.intersphinx',
        'sphinx.ext.todo',

        #'sphinxarg.ext',
        'sphinxcontrib.autoprogram',
        'sphinxcontrib.programoutput',
        'sphinxarg.ext',
        #'sphinxcontrib.fulltoc',
        #'sphinx.ext.imgmath',
        'sphinx.ext.mathjax', 'sphinx.ext.ifconfig', #'sphinx.ext.numfig',
]
# todo settings
###############
todo_include_todos = True
# inheritance_diagram settings
##############################
#inheritance_graph_attrs = dict(rankdir="TB", size='"10.0, 20.0"',
  #ratio='compress', fontsize=12)
inheritance_node_attrs = dict(fontname='Arial', colorscheme='accent8',
  style='filled')
# Napoleon settings
###################
napoleon_google_docstring = True
napoleon_numpy_docstring = False
#napoleon_include_init_with_doc = False
napoleon_include_init_with_doc = True
#napoleon_include_private_with_doc = False
napoleon_include_private_with_doc = True
#napoleon_include_special_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_use_keyword = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


#------------------------------------------------------------------------------
# -- Options for HTML output -------------------------------------------------
html_theme = 'classic' # classic == default
#html_theme = 'agogo'
#html_theme = 'basic'
#html_theme = 'bizstyle'
#html_theme = 'classic'
#html_theme = 'default' # == classic
#html_theme = 'epub' # No link!
#html_theme = 'haiku'
#html_theme = 'nature' # Bello
#html_theme = 'nonav' # No nav!!
#html_theme = 'pyramid'
#html_theme = 'scrolls' # :-(
#html_theme = 'sphinxdoc'
#html_theme = 'traditional' # NO!
#html_theme = 'sphinx_rtd_theme'
#html_theme = 'default'
#html_theme = 'alabaster' # THIS is the default
ogs_link_color = "blue"
html_theme_options = {
        "sidebarbgcolor": "lightgray",
        "sidebartextcolor": "black",
        "sidebarlinkcolor": ogs_link_color,
        "stickysidebar": True,

        "footerbgcolor": "lightgray",
        "footertextcolor": "black",

        "relbarbgcolor": "lightgray",
        "relbartextcolor": "black",
        "relbarlinkcolor": ogs_link_color,
        "externalrefs": True,
        }
html_logo = 'images/ogs.png'
#html_logo = 'images/ogsold.png'
html_show_sourcelink = True
html_sidebars = {
        '**': [#'localtoc.html',
               'globaltoc.html',
               'relations.html',
               'sourcelink.html',
               'searchbox.html',
              ]
        }
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
#html_static_path = ['_static']

#def setup(app):
    #app.add_stylesheet('_static/themeAA_overrides.css')  # may also be an URL

#html_context = {
    #'cssfiles': [
        #'_static/themeAA_overrides.css',  # override wide tables in RTD theme
        #],
     #}
#html_style = 'custom.css'
#------------------------------------------------------------------------------

# Try to get objects.inv
intersphinx_mapping = {
    'numpy': ('https://numpy.org/doc/stable/', None),
}

numfig = True
math_number_all = True
math_numfig = True
numfig_secnum_depth = 2
math_eqref_format = "Eq.{number}"


# =============================================================================
# exec DIRECTIVE
import sys
from os.path import basename

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

from docutils.parsers.rst import Directive
from docutils import nodes, statemachine

class ExecDirective(Directive):
    """Execute the specified python code and insert the output into the document"""
    has_content = True

    def run(self):
        oldStdout, sys.stdout = sys.stdout, StringIO()

        tab_width = self.options.get('tab-width', self.state.document.settings.tab_width)
        source = self.state_machine.input_lines.source(self.lineno - self.state_machine.input_offset - 1)

        try:
            exec('\n'.join(self.content))
            text = sys.stdout.getvalue()
            lines = statemachine.string2lines(text, tab_width, convert_whitespace=True)
            self.state_machine.insert_input(lines, source)
            return []
        except Exception:
            return [nodes.error(None, nodes.paragraph(text = "Unable to execute python code at %s:%d:" % (basename(source), self.lineno)), nodes.paragraph(text = str(sys.exc_info()[1])))]
        finally:
            sys.stdout = oldStdout

# =============================================================================
from importlib import import_module
from docutils  import nodes
from sphinx    import addnodes
from inspect   import getsource
from docutils.parsers.rst import Directive
import ast

class PrettyPrintIterable(Directive):
    required_arguments = 2

    def run(self):
        def retrieve_class_attribute_src(code, tree, classattr):
            cls, att = classattr.split('.')
            for obj in tree.body:
                if isinstance(obj,  ast.ClassDef) and obj.name == cls:
                    for node in ast.walk(obj):
                        if (isinstance(node, ast.Assign)
                                and node.targets[0].id == att):
                            source = ast.get_source_segment(code, node)
                            return source
            return f"Class attribute {classattr} not found!"

        module_path, classattr = self.arguments
        src = getsource(import_module(module_path))
        tree = ast.parse(src)
        code = retrieve_class_attribute_src(src, tree, classattr)

        literal = nodes.literal_block(code, code)
        literal['language'] = 'python'

        return [addnodes.desc_name(),
                addnodes.desc_content('', literal)]

# =============================================================================
def setup(app):
    app.add_directive('exec', ExecDirective)
    app.add_directive('pprint', PrettyPrintIterable)

