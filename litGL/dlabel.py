import numpy as np
import glm
import difflib
np.set_printoptions(suppress=True)

# Local imports
from . import glsl_base_c
from .mesh import Mesh
from .label import Label, countInSet
from .fontDistiller import GlyphTypes

# =============================================================================
class Dlabel(Label):
    """ Class for rendering with OpenGL a string of one or more characters.

        Author:
                - 2020-2021 Nicola Creati

                - 2020-2021 Roberto Vidmar

            Copyright:
                2020-2021 Nicola Creati <ncreati@inogs.it>

                2020-2021 Roberto Vidmar <rvidmar@inogs.it>

            License:
                MIT/X11 License (see |license|)

    """
    __doc__ = ("\n".join(Label.__doc__.splitlines()[:-2])
            + "\nText **CAN** be modified using the *setText* method.")
    def __init__(self, *args, **kargs):
        self.text = ""
        glyphs = kargs.get('glyphs', None)
        self.colors = []
        if glyphs in (GlyphTypes.LAYER_COLOR, GlyphTypes.CBDT_COLOR,
                GlyphTypes.EBDT_COLOR):
            raise SystemExit("FATAL: glyphs value of %d unsupported for"
                    " class %s. Use Label instead."
                    % (glyphs, self.__class__.__name__))
        super().__init__(*args, **kargs)
        self.color = None

    def _setMesh(self):
        """ Set Mesh arrays.
        """
        dtype = np.dtype([('vtx', '<f4', (2,)), ('tex', '<f4', (2,)),
            ('gp', '<u2', (4,)), ('rgba', '<f4', (4,))])
        self.mesh = Mesh(np.empty((0, 0, 0, 0), dtype=dtype),
                np.array((), dtype=np.uint32))

        self.mesh.rebuild(self.allVertices, self.allIndices)

    def _getShaderCode(self):
        """ Return shaders code.

            Returns:
                tuple: vertex shader, fragment shader codes
        """
        if self.glyphs == GlyphTypes.BASE:
            shaders = (glsl_base_c.VERTEX_SHADER, glsl_base_c.FRAGMENT_SHADER)
        else:
            raise RuntimeError("Invalid glyphs code (%d)." % glyphs)
        return shaders

    def _setText(self, text):
        """ Set initial text vertices, indices and color.

            Args:
                text (str): text to set
        """
        self.text = ""
        for ch in text:
            char, vertices, glyph = self._extractGlyph(ch)
            if not vertices is None and self.glyphs in (
                    GlyphTypes.BASE, GlyphTypes.LAYER_COLOR):
                vertices['rgba'] = glm.vec4(self.color)
            if not self.text:
                off, kern = self._updateMetric(0, char)
                if char in self.NO_GLYPH_CHARS:
                    self.colors.append([char, None])
                else:
                    vertices['vtx'] +=  off + glyph['offset']
                    self.allVertices = np.hstack(vertices)
                    self.allIndices = self._baseInd
                    self.colors.append([char, self.color])
                self.text += char
            else:
                pos = len(self.text)
                nonGlyph = countInSet(self.text, self.NO_GLYPH_CHARS)
                # Set the metric
                off, kern = self._updateMetric(pos, char)
                if char in self.NO_GLYPH_CHARS:
                    self.colors.append([char, None])
                else:
                    vertices['vtx'] += off + kern + glyph['offset']
                    if self.allVertices is None:
                        self.allVertices = np.hstack(vertices)
                    else:
                        self.allVertices = np.append(self.allVertices,
                                vertices)
                    if self.allIndices is None:
                        self.allIndices = self._baseInd
                    else:
                        self.allIndices = np.vstack((self.allIndices,
                                self._baseInd + (pos - nonGlyph) * 4))
                    self.colors.append([char, self.color])
                self.text += char
        self.setUniforms()

    def setText(self, text, color=(1, 1, 1, 1)):
        """ Update text and color.

            Args:
                text (str): text to set
                color (:class:`numpy.ndarray`): colors for the text
        """
        if self.filterControl:
            text = self._filterControl(text)
        pos = 0
        for diffs in difflib.ndiff(self.text, text):
            c = diffs[2]
            if diffs[0] == ' ':
                self.logger.debug('NO change at pos %d' % pos)
                pos += 1
            else:
                if diffs[0] == '-':
                    self._delChar(pos)
                    self.logger.info('Removed char at pos %d.' % pos)
                elif diffs[0] == '+':
                    self._insChar(c, pos, color)
                    self.logger.info("Inserted char %r at pos %d." % (c, pos))
                    pos += 1
                self.mesh.rebuild(self.allVertices, self.allIndices)

    def setColors(self, colors, indexes=None):
        """ Set color(s) for the characters in the string.

            Args:
                colors (iterable): color(s), rgba in [0, 1]
                indexes (None or iterable): indexes,
                     can be any of:

                        - None  (default): set the same color for all
                            characters

                        - iterable: iterable of indexes.
                            In this case must have the same length of
                            the colors array
        """
        colors =  np.array(colors, np.float32)
        if indexes is None:
            # Change colors to the whole string
            self.allVertices['rgba'][:] = glm.vec4(colors)
            for item in self.colors:
                item[-1] = colors
        else:
            indexes = np.array(indexes, np.int32)
            assert len(colors) == len(indexes)
            # Adjust indexes
            off = 0
            j = 0
            for i, c in enumerate(self.text):
                if c in self.NO_GLYPH_CHARS:
                    off += 1
                    if i == indexes[j]:
                        if j < len(indexes) - 1:
                            j += 1
                        break
                    continue
                elif i < indexes[j]:
                    continue
                else:
                    self.allVertices['rgba'][
                            4 * (i - off):4 * (i - off + 1)] = colors[j]
                    self.colors[i][-1] = colors[j]
                    if j < len(indexes) - 1:
                        j += 1
                    else:
                        break
        self.mesh.update()

    def _insChar(self, char, pos, color):
        """ Insert `char` at position `pos` with color `color`.

            Args:
                char (str): char to insert
                pos (int): position
                color (tuple): color
        """
        char, vertices, glyph = self._extractGlyph(char, glm.vec4(color))
        if not self.text:
            off, kern = self._updateMetric(pos, char)
            if char in self.NO_GLYPH_CHARS:
                self.colors.insert(pos, [char, None])
            else:
                vertices['vtx'] +=  off + glyph['offset']
                self.allVertices = np.hstack(vertices)
                self.allIndices = self._baseInd
                self.colors.insert(pos, [char, color])
            self.text += char
        else:
            self.logger.debug("Inserting %r at %d" % (char, pos))
            nonGlyph = countInSet(self.text[:pos], self.NO_GLYPH_CHARS)
            # Arrange vertices
            if pos < len(self.text):
                self.allVertices = self.allVertices[:(pos - nonGlyph) * 4]
                self.allIndices = self.allIndices[:pos - nonGlyph]

            # Set the metric
            off, kern = self._updateMetric(pos, char)
            if char in self.NO_GLYPH_CHARS:
                color = None
            else:
                vertices['vtx'] += off + kern + glyph['offset']
                if self.allVertices is None:
                    self.allVertices = np.hstack(vertices)
                else:
                    self.allVertices = np.append(self.allVertices, vertices)
                if self.allIndices is None:
                    self.allIndices = self._baseInd
                else:
                    self.allIndices = np.vstack((self.allIndices,
                            self._baseInd + (pos - nonGlyph) * 4))

            self.colors.insert(pos, [char, color])
            if pos < len(self.text):
                self.text = self.text[:pos] + char + self.text[pos:]
                self._updateGlyphs(pos, char)
            else:
                self.text += char

    def _delChar(self, pos):
        """ Delete char at position `pos`.

            Args:
                char (str): char to insert
                pos (int): position
        """
        nonGlyph = countInSet(self.text[:pos], self.NO_GLYPH_CHARS)

        self.allVertices = self.allVertices[:(pos - nonGlyph) * 4]
        self.allIndices = self.allIndices[:pos - nonGlyph]
        self.colors.pop(pos)
        self._string_metric = self._string_metric[:pos]
        self.text = self.text[:pos] + self.text[pos + 1:]
        self._updateGlyphs(pos)

    def _updateGlyphs(self, pos, char=None):
        """ Update glyphs positions and vertices for chars starting at pos

            Args:
                pos (int): index of the char to insert
                char (str): char to insert or None when deleting
        """
        allVertices = []
        allIndices = []
        for k in range(len(self.text) - pos):
            idx = pos + k
            # Metric
            off, kern = self._updateMetric(idx, self.text[idx])
            # Handle special char
            if self.text[idx] in self.NO_GLYPH_CHARS:
                continue
            # Arrange vertices
            vertices = self.extracted[ord(self.text[idx])]['vertices'].copy()
            vertices['rgba'] = glm.vec4(self.colors[idx][1])
            vertices['vtx'] +=  (off + kern
                    + self.extracted[ord(self.text[idx])]['offset'])
            allVertices.append(vertices)
            if char is None:
                head = self.text[:pos] + self.text[pos:idx]
            else:
                head = self.text[:pos + 1] + self.text[pos:idx]
            s = len(head.replace(' ', '').replace('\n', ''))
            allIndices.append(self._baseInd + 4 * s)
        if len(allVertices) > 0 and len(allIndices) > 0:
            # Arrange vertices indices
            self.allVertices = np.append(self.allVertices,
                    np.hstack(allVertices))
            self.allIndices = np.append(self.allIndices,
                    np.vstack(allIndices), axis=0)
