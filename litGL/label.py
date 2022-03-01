""" A class that implements static string (text) rendering with OpenGL.

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
import OpenGL.GL as gl
import numpy as np
import glm
from pathlib import Path
import unicodedata
from collections import Counter
np.set_printoptions(suppress=True)

# Local imports
from . import namedLogger, glsl_base, glsl_color, glsl_bitmap
from .shader import Shader
from .font import Font
from .mesh import Mesh
from .fontDistiller import GlyphTypes

# -----------------------------------------------------------------------------
def countInSet(a, aSet):
    """ Return sum of all occurrencies of any character in `aSet` in
        string s.

        Args:
            s (str): input string
            aSet (iterable): characters to sesrch for

        Returns:
            int: Return sum of all occurrencies of any character in `aSet` in
                string s.
    """
    return sum(v for k, v in Counter(a).items() if k in aSet)

# =============================================================================
class Transform:
    """ Transformation matrix with shear and stretch. Tansformation order is
        translation * scaling * rotation.
    """
    def __init__(self,
                pos=glm.vec3(0.0),
                rot=glm.quat(1, 0, 0, 0),
                scale=glm.vec3(1.0),
                shear=glm.vec3(0.0),
                stretch=glm.vec3(1.0)):
        """ Create new instance.

            Args:
                pos (:class:`glm.vec3`): OpenGL postition x, y, z
                rot (:class:`glm.quat`): OpenGL rotation quaternion w, x, y, z
                scale (:class:`glm.vec3`): scaling in x, y, z
                shear (:class:`glm.vec3`): shear in x, y, z
                stretch (:class:`glm.vec3`): stretch in x, y, z
        """
        self.pos = pos
        self.rot = rot
        self.scale = scale
        self.shear = shear
        self.stretch = stretch

    def setScale(self, sx, sy=None, sz=None):
        """ Set scaling factors.

            Args:
                sx (float): x scaling
                sy (float): y scaling
                sz (float): z scaling
        """
        if sy is None:
            sy = sx
        if sz is None:
            sz = sx
        self.scale = glm.vec3(sx, sy, sz)

    def setPos(self, pos):
        """ Set transform position.

            Args:
                pos (:class:`glm.vec3`): transform position
        """
        self.pos = pos

    def setRot(self, rot):
        """ Set rotation quaternion.

            Args:
                pos (:class:`glm.quat`): rotation quaternion
        """
        self.rot = rot

    def setShear(self, x=0, y=0, z=0):
        """ Set shear vector.

            Args:
                x (float): x shear
                y (float): y shear
                z (float): z shear
        """
        self.shear = glm.vec3(x, y,z)

    def setStretch(self, x=1, y=1, z=1):
        """ Set stretch vector.

            Args:
                x (float): x stretch
                y (float): y stretch
                z (float): z stretch
        """
        self.stretch = glm.vec3(x, y,z)

    def getTransformation(self):
        """ Return transformation matrix i.e. (translationMatrix
            * scaleMatrix * rotationMatrix * stretchMatrix * shearMatrix)

            Returns:
                :class:`glm.mat4x4`: transformation matrix
        """
        translationMatrix = glm.translate(glm.mat4(1.0), self.pos)
        rot = glm.normalize(self.rot)
        rotationMatrix = glm.rotate(glm.mat4(1.0), glm.angle(self.rot),
                glm.axis(self.rot))
        scaleMatrix = glm.scale(glm.mat4(1.0), self.scale)

        # Python glm has no shear matrix transformation
        shearMatrix = glm.mat4(1.0)
        shearMatrix[1][0] = self.shear.x
        shearMatrix[0][1] = self.shear.y

        # Stretching matrix
        stretchMatrix = glm.mat4(1.0)
        stretchMatrix[1][1] = self.stretch.y
        stretchMatrix[0][0] = self.stretch.x

        return (translationMatrix * scaleMatrix * rotationMatrix
                * stretchMatrix * shearMatrix)

# =============================================================================
class Label:
    """ A class that implements string (text) rendering with OpenGL.

        The text **can not** be modified.
    """
    DEFAULT_FONT = "LiberationSans-Regular.nbf"
    PIXEL_UNITS = 0
    POINT_UNITS = 1
    NO_GLYPH_CHARS = "\n, ".split(",")
    "These characters have no corresponding glyphs."
    DPI = None
    "Dot per inch. Must be set."
    Y_POINTS_UP = False
    "Y axis points up if True."

    def __init__(self, text, pos=(0, 0), anchor='ll',
            size=16,
            sizeUnits=POINT_UNITS,
            color=(1, 0, 1, 1),
            dpi=None,
            yPointsUp=None,
            font_file=None,
            angle=0.,
            glyphs=GlyphTypes.BASE,
            filterControl=True):
        r""" Create new instance

            Args:
                text (str): text pos (tuple): x, y position in
                    *window coordinates*.
                anchor (str): optional anchor position in
                    ::

                           ul         uc          ur
                            +----------+----------+
                            |                     |
                         cl +         cc          + cr
                            |                     |
                            +----------+----------+
                           ll         lc          lr

                        Default is `ll` (lower left).

                size (float): text size (point size or pixel size)
                sizeUnits (int): any of (
                    :class:`litGL.label.Label.PIXEL_UNITS`,
                    :class:`litGL.label.Label.POINT_UNITS`)
                color (tuple): r, g, b, a in range [0, 1].
                    Alpha value is used also to set transparency of colored
                    fonts (enabled setting `glyphs` argument with
                    any value except
                    :class:`litGL.fontDistiller.GlyphTypes.BASE`)
                dpi (float): set dot per inch screen resolution for this
                    instance.

                    - if dpi is None instance will use class
                      attribute DPI instead.

                    - else sets also class attribute DPI if unset.
                yPointsUp(bool): set Y axis orientation: True means
                    y axis points up
                font_file (str): font file (must be an **.nbf** distilled file)
                angle (float): rotation angle from x axis, positive
                    anti-clockwise
                glyphs (:class:`litGL.fontDistiller.GlyphTypes`): glyph type
                filterControl (bool): remove control characters (characters
                    in range [0, 31] except '\\n') from text
        """
        self.logger = namedLogger(__name__, self.__class__)
        if dpi:
            if Label.DPI is None:
                Label.DPI = dpi
            else:
                self.DPI = dpi
        if yPointsUp:
            if Label.Y_POINTS_UP is None:
                Label.Y_POINTS_UP = yPointsUp
            else:
                self.Y_POINTS_UP = yPointsUp
        self.color = color
        self.glyphs = glyphs
        self.filterControl = filterControl

        # Model matrix
        self.transform = Transform()

        if font_file is None:
            font_file = Path(__file__).parent.joinpath(Label.DEFAULT_FONT)
        self.font = Font(font_file)
        self._lineWidth = 0
        self._labelWidth = 0
        self._labelHeight = self.font.table['linespace']
        self.setSize(size, sizeUnits)

        self._baseInd = np.array([0, 1, 2, 2, 3, 0], np.uint32)
        self.allVertices = None
        self.allIndices = None
        self.extracted = {}
        # Offet, kerning, next_char_shift
        self._string_metric = []

        # Set text
        if self.filterControl:
            text = self._filterControl(text)
        self.shader = Shader.fromString(*self._getShaderCode())
        self._setText(text)
        self._setMesh()
        self.model = Transform()
        self.setPos(*pos, anchor)
        self.setRotation(angle)

    def _filterControl(self, text):
        # Remove control characters
        return "".join(ch for ch in text if ch == '\n' or (ord(ch) > 31))

    def _getShaderCode(self):
        """ Return shader codes

            Returns:
                tuple: vertex shader, fragment shader codes
        """
        if self.glyphs == GlyphTypes.BASE:
            shaders = (glsl_base.VERTEX_SHADER, glsl_base.FRAGMENT_SHADER)
        elif self.glyphs == GlyphTypes.LAYER_COLOR:
             shaders = (glsl_color.VERTEX_SHADER, glsl_color.FRAGMENT_SHADER)
        elif self.glyphs in (
                GlyphTypes.CBDT_COLOR, GlyphTypes.EBDT_COLOR):
             shaders = (glsl_bitmap.VERTEX_SHADER, glsl_bitmap.FRAGMENT_SHADER)
        else:
            raise RuntimeError("Invalid glyphs code (%d)." % glyphs)
        return shaders

    def _setMesh(self):
        """ Set Mesh arrays.
        """
        dtype = np.dtype([('vtx', '<f4', (2,)), ('tex', '<f4', (2,)),
            ('gp', '<u2', (4,)), ])
        self.mesh = Mesh(np.empty((0, 0, 0, 0), dtype=dtype),
                np.array((), dtype=np.uint32))
        self.mesh.rebuild(self.allVertices, self.allIndices)

    def nextCharLowerLeft(self):
        """ Return next character lower left position (in window
            coordinates) according to current metric.

            Returns:
                tuple: next character lower left x, y position
        """
        pmodel = self.model.pos * self.transform.scale
        x, y, _ = self.transform.pos + pmodel
        y += ((self.font.table['ascent'] + self.y_sign * self._labelHeight)
                * self.transform.scale[1])
        x += self._string_metric[-1][2][0] * self.transform.scale[0]
        return x, y

    def setShear(self, x=0, y=0):
        """ Set shear vector.

            Args:
                x (float): x shear
                y (float): y shear
        """
        self.transform.setShear(x, y, 0)

    def setStretch(self, x=1, y=1):
        """ Set stretch vector.

            Args:
                x (float): x stretch
                y (float): y stretch
        """
        self.transform.setStretch(x, y, 0)

    def setSize(self, size, units=POINT_UNITS):
        """ Size can be expressed in pixel or point units.
            This will be the size at 0 zoom level and without window resizing.

            :class:`Label.PIXEL_UNITS`: Size is calculated according to
                the maximum range along x and y (bounding box metrics)
                of the passed string.
            :class:`Label.POINT_UNITS`: Size is calculated according to
                the freetype formula to convert points units to pixels.
                It depends on the screen dpi that must be supplied at
                initialization.

            Args:
                size (float): character size in `units`
                units (int): :class:`Label.POINT_UNITS` or
                    :class:`Label.PIXEL_UNITS`
        """
        assert units in (Label.PIXEL_UNITS, Label.POINT_UNITS)
        if self.Y_POINTS_UP:
            self.y_sign = 1
        else:
            self.y_sign = -1
        if units == Label.PIXEL_UNITS:
            scale = size / self.font.table['units_per_EM']
        else:
            scale = size * self.DPI / (
                    72 * self.font.table['units_per_EM'])
        self.transform.setScale(scale, self.y_sign * scale)
        self.size = size
        self.sizeUnits = units

    def boundingBox(self):
        """ Return the label bounding box for the **un**-rotated text.

            Returns:
                tuple: xmin, xmax, ymin, ymax (window coordinates)
        """
        pmodel = (glm.vec3(1, -self.y_sign, 0)
                * self.model.pos * self.transform.scale)
        x, y, _ = self.transform.pos + pmodel
        y += -self.y_sign * self.font.table['ascent'] * self.transform.scale[1]
        return x, y, self.pixwidth(), self.pixheight()

    def pixheight(self):
        """ Return label height in pixels (screen coordinates).

            Returns:
                float: label height in pixels
        """
        return self._labelHeight * self.y_sign * self.transform.scale[1]

    def pixwidth(self):
        """ Return label width in pixels (screen coordinates).

            Returns:
                float: label width in pixels
        """
        return self._labelWidth * self.transform.scale[0]

    def setPos(self, x, y, anchor='ll'):
        """ Set text position in **screen coordinates**.

            Args:
                x (float): x position in *screen coordinates*.
                y (float): y position in *screen coordinates*.
                anchor (str): optional anchor position in

            ::

                   ul         uc          ur
                    +----------+----------+
                    |                     |
                 cl +         cc          + cr
                    |                     |
                    +----------+----------+
                   ll         lc          lr

        """
        self.transform.setPos(glm.vec3(x, y, 0))
        if anchor == 'ul':
            offx = 0
            offy = - self.font.table['ascent']
        elif anchor == 'uc':
            offx = - self._labelWidth / 2
            offy = - self.font.table['ascent']
        elif anchor == 'ur':
            offx = - self._labelWidth
            offy = - self.font.table['ascent']
        elif anchor == 'cl':
            offx = 0
            offy = self._labelHeight / 2 - self.font.table['ascent']
        elif anchor == 'cc':
            offx = - self._labelWidth / 2
            offy = self._labelHeight / 2 - self.font.table['ascent']
        elif anchor == 'cr':
            offx = - self._labelWidth
            offy = self._labelHeight / 2 - self.font.table['ascent']
        elif anchor == 'll':
            offx = 0
            offy = self._labelHeight - self.font.table['ascent']
        elif anchor == 'lc':
            offx = - self._labelWidth / 2
            offy = self._labelHeight - self.font.table['ascent']
        elif anchor == 'lr':
            offx = - self._labelWidth
            offy = self._labelHeight - self.font.table['ascent']
        else:
            raise SystemExit(f"Unimplemented anchor '{anchor}'")
        self.model.setPos(glm.vec3(offx, offy, 0))

    def setScale(self, sx, sy=None, sz=None):
        """ Set transformation scale for all axes.

            Args:
                sx (float): scale x
                sy (float): scale y
                sz (float): scale z
        """
        self.transform.setScale(sx, sy, sz)

    def setRotation(self, angle=0.0):
        """ Set rotation transformation with `angle` degrees anti clockwise
            around Z axis (positive towards viewer).

            Args:
                angle (float): rotation angle in degrees
        """
        axis = (0, 0, 1)
        oldp = self.transform.pos
        newpos = oldp + glm.vec3(0, -40, 0)
        self.transform.setPos(newpos)
        self.transform.setRot(glm.angleAxis(glm.radians(angle),
                glm.vec3(axis)))
        self.transform.setPos(oldp)

    def setUniforms(self):
        """ Set shader uniforms.
        """
        self.shader.bind()
        if self.glyphs in (GlyphTypes.BASE, GlyphTypes.LAYER_COLOR):
            self.shader.setUniform('u_curvesTex', 0)
            self.shader.setUniform('u_bandsTex', 1)
            if self.glyphs == GlyphTypes.LAYER_COLOR:
                self.shader.setUniform('u_colorsTex', 2)
        elif self.glyphs in (
                GlyphTypes.CBDT_COLOR, GlyphTypes.EBDT_COLOR):
            self.shader.setUniform('u_colorsTex', 0)
        self.shader.unbind()

    def render(self, proj):
        """ Render text using OpenGL

            Args:
                proj (:class:`glm.mat4x4`): projection matrix
        """
        if self.text == '' or not self.mesh:
            return

        model = self.model.getTransformation()
        mvp = proj * self.transform.getTransformation() * model

        gl.glEnable(gl.GL_FRAMEBUFFER_SRGB)

        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        self.shader.bind()
        if self.color:
            self.shader.setUniform('u_color', self.color)
        self.font.bindAtlas()
        self.shader.setUniform('T_MVP', mvp)
        self.mesh.draw()
        gl.glDisable(gl.GL_BLEND)
        self.shader.unbind()
        self.font.unbindAtlas()
        gl.glDisable(gl.GL_FRAMEBUFFER_SRGB)

    def _updateMetric(self, i, char):
        """ Update glyphs metric at index i for character `char`.

            Args:
                i (int): index
                char (str): unicode character
        """
        # Set the metric
        if i == 0:
            off = glm.vec2(0., 0.)
            kern = glm.vec2(0.0, 0.0)
        else:
            off = glm.vec2(self._string_metric[i-1][2])
            kern = glm.vec2(self.font.getKerning(self.text[i - 1], char))
        glyph = self.extracted.get(ord(char), None)
        if glyph:
            horz_advance = glyph.get('horz_advance', 0.0)
        else:
            horz_advance = 0.0
        self._lineWidth += horz_advance
        self._labelWidth = max(self._labelWidth, self._lineWidth)
        if char == '\n':
            off.x = 0.0
            next_char_shift = glm.vec2(off)
            self._lineWidth = 0
            textHeight = self.font.table['linespace']
            next_char_shift.y -= textHeight
            self._labelHeight += textHeight
        else:
            next_char_shift = glm.vec2(off) + glm.vec2(horz_advance, 0) + kern
        self._string_metric = self._string_metric[:i]
        self._string_metric.append((glm.vec2(off), kern, next_char_shift))
        return off, kern

    def _setText(self, text):
        """ Set text vertices, indices and color.

            Args:
                text (str): text to set
        """
        self.text = ""
        for ch in text:
            char, vertices, glyph = self._extractGlyph(ch)
            if not self.text:
                off, kern = self._updateMetric(0, char)
                if vertices is not None and not char in self.NO_GLYPH_CHARS:
                    vertices['vtx'] +=  off + glyph['offset']
                    self.allVertices = np.hstack(vertices)
                    self.allIndices = self._baseInd
                self.text += char
            else:
                pos = len(self.text)
                nonGlyph = countInSet(self.text, self.NO_GLYPH_CHARS)
                # Set the metric
                off, kern = self._updateMetric(pos, char)
                if vertices is not None and not char in self.NO_GLYPH_CHARS:
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
                self.text += char
        self.setUniforms()

    def _extractGlyph(self, char):
        """ Return vertices and glyph for character `char`.
            If no glyph exists for `char` replace `char` with blank.

            Args:
                char (str): the unicode char

            Returns:
                tuple:
                    - char (str): the input char or balnk (chr(32))
                    - vertices (:class:`numpy.ndarray`): glyph vertices
                    - currentGlyph (dict): the extracted glyph
        """
        charno = ord(char)
        vertices = None
        currentGlyph = None

        if charno in self.extracted:
            currentGlyph = self.extracted[charno]
        else:
            if char in ('\n', ):
                # No glyph for these chars
                pass
            else:
                glyph = self.font.getGlyph(charno, self.glyphs)
                if glyph is None:
                    save_char = char
                    save_charno = charno
                    # Use '.notdef' glyph if it is defined in the font
                    repcharno = None
                    if self.glyphs != GlyphTypes.CBDT_COLOR:
                        glyph = self.font.getGlyph(repcharno, self.glyphs)
                    if glyph is None:
                        # Use WHITE SQUARE gplyph: \u25A1
                        repcharno = 9633
                        glyph = self.font.getGlyph(repcharno, self.glyphs)
                        if glyph is None:
                            # Still None? Replace character with blank
                            repcharno = 32
                            glyph = self.font.getGlyph(repcharno, self.glyphs)
                            charno = 32
                            char = chr(charno)
                            if glyph is None:
                                self.logger.error("Font %s has no space"
                                        " character!" % self.font.fontFile)

                    self.logger.warning("Char %r (%d) not found in"
                            " font %s has been replaced with chr(%s)"
                            % (save_char, save_charno, self.font.fontFile,
                            repcharno))

                currentGlyph = glyph
                self.extracted[charno] = currentGlyph

        if currentGlyph is not None and 'vertices' in currentGlyph:
            vertices = currentGlyph['vertices'].copy()

        return char, vertices, currentGlyph
