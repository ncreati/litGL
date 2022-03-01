""" The texture class module.

        Author:
            - 2020-2021 Nicola Creati

            - 2020-2021 Roberto Vidmar

        Copyright:
            2020-2021 Nicola Creati <ncreati@inogs.it>

            2020-2021 Roberto Vidmar <rvidmar@inogs.it>

        License:
            MIT/X11 License (see
            :download:`license.txt <../../../license.txt>`)
"""
import numpy as np
import OpenGL.GL as gl

# To use half_float (np.float16)
# from OpenGL.GL.NV import half_float
# change np.dtype(np.float16): gl.GL_FLOAT
# to np.dtype(np.float16): half_float.GL_HALF_NV
# By the way it is still possible to send an np.float16 array and set the
# texelType to GL_FLOAT and the internalFormat to GL_RGBA16F size
# the GPU makes the conversion internally storing a texture with half size

vertex_type = {
               np.dtype(np.int8)    : gl.GL_BYTE,
               np.dtype(np.uint8)   : gl.GL_UNSIGNED_BYTE,
               np.dtype(np.int16)   : gl.GL_SHORT,
               np.dtype(np.uint16)  : gl.GL_UNSIGNED_SHORT,
               np.dtype(np.int32)   : gl.GL_INT,
               np.dtype(np.uint32)  : gl.GL_UNSIGNED_INT,
               np.dtype(np.float16) : gl.GL_FLOAT,
               np.dtype(np.float32) : gl.GL_FLOAT,
               np.dtype(np.float64) : gl.GL_DOUBLE
               }

# =============================================================================
class Texture(object):
    """
::

    https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/glTexImage2D.xhtml
    border = must be 0
    data_type(type) = Specifies the data type of the pixel data. The following
                    symbolic values are accepted: GL_UNSIGNED_BYTE, GL_BYTE,
                    GL_UNSIGNED_SHORT, GL_SHORT, GL_UNSIGNED_INT, GL_INT,
                    GL_HALF_FLOAT, GL_FLOAT, GL_UNSIGNED_BYTE_3_3_2,
                    GL_UNSIGNED_BYTE_2_3_3_REV, GL_UNSIGNED_SHORT_5_6_5,
                    GL_UNSIGNED_SHORT_5_6_5_REV, GL_UNSIGNED_SHORT_4_4_4_4,
                    GL_UNSIGNED_SHORT_4_4_4_4_REV, GL_UNSIGNED_SHORT_5_5_5_1,
                    GL_UNSIGNED_SHORT_1_5_5_5_REV, GL_UNSIGNED_INT_8_8_8_8,
                    GL_UNSIGNED_INT_8_8_8_8_REV, GL_UNSIGNED_INT_10_10_10_2,
                    and GL_UNSIGNED_INT_2_10_10_10_REV.
    data = array
    """
    def __init__(self, array, width, height, target=gl.GL_TEXTURE_2D, level=0,
                 internalFormat=gl.GL_RGBA, pixFormat=gl.GL_RGBA,
                 interpolator=gl.GL_NEAREST, clamp=gl.GL_REPEAT):
        """
            Args:
                array (:class:`numpy.ndarray`): data to be converted to
                    texture
                width (int): texture width
                height (int): texture height
                target (:class:`OpenGL.constant.IntConstant`): Specifies
                    the target texture. Must be any of:
                    GL_TEXTURE_2D, GL_PROXY_TEXTURE_2D, GL_TEXTURE_1D_ARRAY,
                    GL_PROXY_TEXTURE_1D_ARRAY,
                    GL_TEXTURE_RECTANGLE, GL_PROXY_TEXTURE_RECTANGLE,
                    GL_TEXTURE_CUBE_MAP_POSITIVE_X,
                    GL_TEXTURE_CUBE_MAP_NEGATIVE_X,
                    GL_TEXTURE_CUBE_MAP_POSITIVE_Y,
                    GL_TEXTURE_CUBE_MAP_NEGATIVE_Y,
                    GL_TEXTURE_CUBE_MAP_POSITIVE_Z,
                    GL_TEXTURE_CUBE_MAP_NEGATIVE_Z,
                    GL_PROXY_TEXTURE_CUBE_MAP
                level (int): Specifies the level-of-detail number.

                    - Level 0 is the base image level.
                    - Level n is the nth mipmap reduction image.

                    If target is GL_TEXTURE_RECTANGLE or
                    GL_PROXY_TEXTURE_RECTANGLE, level must be 0.
                internalFormat (:class:`OpenGL.constant.IntConstant`):
                    Specifies the number of color components in the texture.
                pixFormat (:class:`OpenGL.constant.IntConstant`): Specifies
                    the format of the pixel data. The following
                    symbolic values are accepted: GL_RED, GL_RG, GL_RGB,
                    GL_BGR, GL_RGBA, GL_BGRA, GL_RED_INTEGER, GL_RG_INTEGER,
                    GL_RGB_INTEGER, GL_BGR_INTEGER, GL_RGBA_INTEGER,
                    GL_BGRA_INTEGER, GL_STENCIL_INDEX, GL_DEPTH_COMPONENT,
                    GL_DEPTH_STENCIL.
                interpolator (:class:`OpenGL.constant.IntConstant`):
                    interpolator
                clamp (:class:`OpenGL.constant.IntConstant`): ??
        """
        self.array = array
        self.target = target
        self._width = width
        self._height = height
        border = 0
        texelType = vertex_type[array.dtype]
        self._texelType = texelType
        self.texID = gl.glGenTextures(1)
        self._pix_format = pixFormat
        gl.glBindTexture(target, self.texID)

        gl.glTexImage2D(target, border,  internalFormat, width, height, 0,
                pixFormat, texelType, array)

        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER,
                interpolator)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER,
                interpolator)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, clamp)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, clamp)

        gl.glBindTexture(target, 0)

    def bind(self, samplerSlot=0):
        """ Select active texture unit and bind the texture to a
            texturing target.

            Args:
                samplerSlot (int): slot offset from gl.GL_TEXTURE0.
        """
        assert (samplerSlot >= 0 and samplerSlot <= 31)
        gl.glActiveTexture(gl.GL_TEXTURE0 + samplerSlot)
        gl.glBindTexture(self.target, self.texID)

    def unbind(self):
        """ Unbind texture.
        """
        gl.glBindTexture(self.target, 0)
