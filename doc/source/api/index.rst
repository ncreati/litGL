.. include:: ../include.rst

litGL package reference
===============================================================================

.. automodule:: litGL.__init__
    :members:
    :exclude-members: __dict__, __module__, __weakref__
    :inherited-members:
    :special-members: __str__, __repr__, __len__
    :undoc-members:
    :show-inheritance:

:mod:`litGL.fontDistiller` 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: litGL.fontDistiller
    :members:
    :exclude-members: __dict__, __module__, __weakref__, FontDistiller
    :inherited-members:
    :special-members:  __str__, __repr__, __len__
    :undoc-members:
    :show-inheritance:

    .. autoclass:: FontDistiller
        :members:
        :exclude-members: NBF_DIR, __dict__, __module__, __weakref__
        :inherited-members:
        :special-members: __str__, __repr__, __len__
        :undoc-members:
        :show-inheritance:

        .. autoattribute:: NBF_DIR
            :annotation: = '/home/user/.local/share/fonts/nbf'

:mod:`litGL.font` 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: litGL.font
    :members:
    :exclude-members: __dict__, __module__, __weakref__, DEFAULT_FONT_FILE
    :inherited-members:
    :special-members: __str__, __repr__, __len__
    :undoc-members:
    :show-inheritance:

    .. autodata:: DEFAULT_FONT_FILE
        :annotation: = 'LiberationSans-Regular.nbf'

:mod:`litGL.glsl_base` 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: litGL.glsl_base

    .. autodata:: VERTEX_SHADER
        :annotation: = The vertex shader code
    .. autodata:: FRAGMENT_SHADER
        :annotation: = The fragment shader code

:mod:`litGL.glsl_base_c` 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: litGL.glsl_base_c

    .. autodata:: VERTEX_SHADER
        :annotation: = The vertex shader code
    .. autodata:: FRAGMENT_SHADER
        :annotation: = The fragment shader code

:mod:`litGL.glsl_bitmap` 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: litGL.glsl_bitmap

    .. autodata:: VERTEX_SHADER
        :annotation: = The vertex shader code
    .. autodata:: FRAGMENT_SHADER
        :annotation: = The fragment shader code

:mod:`litGL.glsl_color` 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: litGL.glsl_color

    .. autodata:: VERTEX_SHADER
        :annotation: = The vertex shader code
    .. autodata:: FRAGMENT_SHADER
        :annotation: = The fragment shader code

:mod:`litGL.label` 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: litGL.label
    :members:
    :exclude-members: __dict__, __module__, __weakref__
    :inherited-members:
    :special-members: __str__, __repr__, __len__
    :undoc-members:
    :show-inheritance:

:mod:`litGL.dlabel` 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: litGL.dlabel
    :members:
    :exclude-members: __dict__, __module__, __weakref__
    :inherited-members:
    :special-members: __init__, __str__, __repr__, __len__
    :undoc-members:
    :show-inheritance:

:mod:`litGL.mesh` 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: litGL.mesh
    :members:
    :exclude-members: __dict__, __module__, __weakref__
    :inherited-members:
    :special-members: __str__, __repr__, __len__
    :undoc-members:
    :show-inheritance:

:mod:`litGL.glyph` 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: litGL.glyph
    :members:
    :exclude-members: __dict__, __module__, __weakref__
    :inherited-members:
    :special-members: __str__, __repr__, __len__
    :undoc-members:
    :show-inheritance:

..  .. exec::
        import pprint
        import textwrap
        from litGL.shader import Shader
        pad = "    "
        cb = ".. code-block:: Python\n\n"
        data = pprint.pformat(dict(Shader.GL_type_to_string))
        data = textwrap.indent(text=data, prefix=pad)
        data = pad + "GL_type_to_string = {\n" + pad + " " + data.lstrip()[1:]
        cb += data
        print(cb)

    .. pprint:: litGL.shader.Shader.EXTENSIONS
    .. literalinclude:: ../../../litGL/shader.py
        :language: python
        :lines: 318-323
    .. literalinclude:: ../../../litGL/shader.py
        :language: python
        :lines: 209-316
    .. literalinclude:: ../../../litGL/shader.py
        :language: python
        :lines: 172-207

:mod:`litGL.shader` 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: litGL.shader
    :members:
    :exclude-members: __dict__, __module__, __weakref__, GL_type_to_string,
        UNIFORM_FUNCS, EXTENSIONS
    :inherited-members:
    :special-members: __str__, __repr__, __len__
    :undoc-members:
    :show-inheritance:

    .. autoattribute:: litGL.shader.Shader.EXTENSIONS
        :annotation:
    .. pprint:: litGL.shader Shader.EXTENSIONS
        
    .. autoattribute:: litGL.shader.Shader.GL_type_to_string
        :annotation:
    .. pprint:: litGL.shader Shader.GL_type_to_string 

    .. autoattribute:: litGL.shader.Shader.UNIFORM_FUNCS
        :annotation:
    .. pprint:: litGL.shader Shader.UNIFORM_FUNCS


:mod:`litGL.texture` 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: litGL.texture
    :members:
    :exclude-members: __dict__, __module__, __weakref__
    :inherited-members:
    :special-members: __str__, __repr__, __len__
    :undoc-members:
    :show-inheritance:

Subpackages                                                                     
------------------------------------------------------------------------------- 
                                                                                
.. toctree::                                                                    
    :maxdepth: 2                                                                
                                                                                
    litGL.examples   
