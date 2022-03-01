import numpy as np
import OpenGL.GL as gl
import ctypes

vertex_type = {
               np.dtype(np.int8)   : gl.GL_BYTE,
               np.dtype(np.uint8)  : gl.GL_UNSIGNED_BYTE,
               np.dtype(np.int16)  : gl.GL_SHORT,
               np.dtype(np.uint16) : gl.GL_UNSIGNED_SHORT,
               np.dtype(np.float16) : None,
               np.dtype(np.int32)  : gl.GL_INT,
               np.dtype(np.uint32) : gl.GL_UNSIGNED_INT,
               np.dtype(np.float32): gl.GL_FLOAT,
               np.dtype(np.float64): gl.GL_DOUBLE
               }

# =============================================================================
class Mesh(object):
    """ The mesh class.

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
    def __init__(self, vertices, indices=None, mode=gl.GL_STATIC_DRAW):
        """ Create new instance.

            Args:
                vertices (:class:`numpy.ndarray`): mesh vertices
                indices (:class:`numpy.ndarray`): mesh indices
                mode (int): specifies the expected usage pattern of the data
                    store (https://www.khronos.org/registry/\
OpenGL-Refpages/gl4/html/glBufferData.xhtml)
        """
        self.vertices = vertices
        self.indices = indices

        # Build the Vertex Array Object
        self._vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self._vao)

        if indices is not None:
            self._ibo = gl.glGenBuffers(1)
            gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self._ibo)
            gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, self.indices, mode)

        self._vbo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices, mode)

        strides = [field[1][1] for field in self.vertices.dtype.fields.items()]
        fields = [field for field in self.vertices.dtype.fields.items()]
        sortedIndices = np.argsort(strides)
        sortedFields = [fields[i] for i in sortedIndices]

        for index, field in enumerate(sortedFields):
            gl.glEnableVertexAttribArray(index)
            size = field[1][0].shape[0]
            stride = ctypes.c_void_p(field[1][-1])
            glType = vertex_type[field[1][0].base]
            normalized = gl.GL_FALSE
            gl.glVertexAttribPointer(index, size, glType, normalized,
                    self.vertices.itemsize, stride)
        gl.glBindVertexArray(0)

    def rebuild(self, vertices, indices):
        """ Rebuild the mesh with new vertices and indices.

            Args:
                vertices (:class:`numpy.ndarray`): vertices
                indices (:class:`numpy.ndarray`): indices
        """
        if vertices is None and indices is None:
            return
        self.vertices = vertices
        self.indices = indices

        gl.glBindVertexArray(self._vao)

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices, gl.GL_STATIC_DRAW)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self._ibo)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, self.indices,
                gl.GL_STATIC_DRAW)

        strides = [field[1][1] for field in self.vertices.dtype.fields.items()]
        fields = [field for field in self.vertices.dtype.fields.items()]
        sortedIndices = np.argsort(strides)
        sortedFields = [fields[i] for i in sortedIndices]

        for index, field in enumerate(sortedFields):
            gl.glEnableVertexAttribArray(index)
            size = field[1][0].shape[0]
            stride = ctypes.c_void_p(field[1][-1])
            glType = vertex_type[field[1][0].base]
            normalized = gl.GL_FALSE
            gl.glVertexAttribPointer(index, size, glType, normalized,
                    self.vertices.itemsize, stride)
        gl.glBindVertexArray(0)

    def update(self):
        """ Update mesh buffer.
        """
        gl.glBindVertexArray(self._vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._vbo)
        gl.glBufferSubData(gl.GL_ARRAY_BUFFER, 0, self.vertices)
        gl.glBindVertexArray(0)

    def draw(self, primitive=gl.GL_TRIANGLES):
        """ Draw the mesh.
        """
        gl.glBindVertexArray(self._vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._vbo)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self._ibo)
        if self.indices is not None:
            gl.glDrawElements(primitive, self.indices.size, gl.GL_UNSIGNED_INT, None)
        else:
            gl.glDrawArray(primitive, 0, self.vertices.size)
        gl.glBindVertexArray(0)
