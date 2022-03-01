.. include:: include.rst

|project|_ Description
===============================================================================
|project|_ (**lit**\ ographic  **G**\ raphics **L**\ ibrary) is a Python
package containing the modules needed to render text to the screen in an
OpenGL_ window.

It relays on the rendering of the Bezier curves that define the outlines of
each character. It is partially based on the
`SLUG <https://sluglibrary.com/>`_ alghoritm described in some public
documents. Unlike common methods based on the rendering of the raster version
of each glyph, the quality of the glyphs is not affected by the level of
zoom and preserves its accuracy at almost all scales. 

|project|_ cannot use directly TrueType/OpenType font files to render
characters on the fly, it needs to analyze them first to create
a data structure suitable for the OpenGL_ rendering. This process is done
automatically by the 
(:class:`litGL.fontDistiller`) module that interprets the TrueType/OpenType
font file and creates an output file that contains all the information
needed by |project|_ to render any glyph defined herein. The output of the
`distillation` process is a standard Python dictionary serialized with
the pickle module and gzipped.

The :class:`litGL.label.Label` class is in charge of rendering any text string
in an OpenGL_ context using the distilled font file.

|project|_ is hosted on `GitHub <project_home_>`_.


Installation
-------------------------------------------------------------------------------
To install |project|_ from `GitHub <project_home_>`_:

.. parsed-literal::

    $ pip3 install |projectg|


Usage
-------------------------------------------------------------------------------
Usage examples of |project|_ can be found in the :doc:`examples` section.

Contents:
-------------------------------------------------------------------------------
.. toctree::
    :maxdepth: 2

    api/index
    examples
    credits
    license
    changes

Indices and tables
===============================================================================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
