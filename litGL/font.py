""" The font class module.

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
from pathlib import Path
import gzip
import pickle
import copy

# Local imports
from .fontDistiller import FontDistiller, GlyphTypes
from .texture import Texture
from . import namedLogger

_THIS_DIR = Path(__file__).parent
#: This is the default font file
DEFAULT_FONT_FILE = Path.joinpath(_THIS_DIR, 'LiberationSans-Regular.nbf')

#==============================================================================
class Singleton(type):
    """ Metaclass.
    """
    _instances = {}

    def __call__(cls, args):
        """ Ensure only one instance exists with the same font.
        """
        if (cls, args) not in cls._instances:
            cls._instances[(cls, args)] = super().__call__(args)
        instance = cls._instances[(cls, args)]
        #print("Font Singleton: ID=", id(instance))
        return instance

#==============================================================================
class Font(metaclass=Singleton):
    def __init__(self, fontFile=DEFAULT_FONT_FILE, buildAtlas=True):
        """__init__(self, fontFile=DEFAULT_FONT_FILE, buildAtlas=True)
            Font files must be in the nbf format, otherwise they will
            be compiled only once to nbf in the
            :class:`litGL.fontDistiller.FontDistiller.NBF_DIR` folder.

            Args:
                fontFile (str): pathname of the font file
        """
        self.logger = namedLogger(__name__, self.__class__)
        fontFile = Path(fontFile)
        if fontFile.suffix == FontDistiller.EXT:
            nbfFile = fontFile
        else:
            # Create the directory
            Path(FontDistiller.NBF_DIR).mkdir(parents=True, exist_ok=True)
            nbfFile = Path.joinpath(FontDistiller.NBF_DIR,
                    "%s%s" % (fontFile.stem, FontDistiller.EXT))
            if not nbfFile.is_file():
                # Compile it
                try:
                    FontDistiller(fontFile).save(nbfFile)
                except (RuntimeError, ValueError) as e:
                    self.logger.critical("Cannot distill font"
                            f" {fontFile}, reason is '{e}'.")
                    nbfFile = DEFAULT_FONT_FILE

        # Read the nbf fonr file abd retriev the data table
        data = gzip.GzipFile(nbfFile)
        self.table = pickle.loads(data.read())
        data.close()
        self.fontFile = nbfFile
        self.atlas = []
        if buildAtlas:
            self.buildAtlasTextures()

    def buildAtlasTextures(self):
        """ Create all atlas Textures.
        """
        if self.atlas:
            self.logger.debug("self.atlas exists, no need to build!")
            return
        if self.table.get('curvesArrayShape'):
            width, height, b = self.table['curvesArrayShape']
            # Curves array
            curvesArray = np.ascontiguousarray(self.table['curvesArray'])
            self.atlas.append(Texture(curvesArray, width, height,
                    target=gl.GL_TEXTURE_RECTANGLE,
                    internalFormat=gl.GL_RGBA16F, pixFormat=gl.GL_RGBA))
            # Bands array
            width, height, b = self.table['bandsArrayShape']
            bandsArray = np.ascontiguousarray(self.table['bandsArray'])
            if bandsArray.dtype == np.uint16:
                internalFormat = gl.GL_RG16UI
            elif bandsArray.dtype == np.uint32:
                internalFormat = gl.GL_RG32UI
            self.atlas.append(Texture(bandsArray, width, height,
                    target=gl.GL_TEXTURE_RECTANGLE,
                    internalFormat=internalFormat, pixFormat=gl.GL_RG_INTEGER))

        # Get the colored array for layered or bitmap glyph if any
        colored = self.table.get('colored')
        if colored != GlyphTypes.BASE:
            if colored == GlyphTypes.LAYER_COLOR:
                width, height, b = self.table['colorsArrayShape']
                colorsArray = np.ascontiguousarray(self.table['colorsArray'])
                self.atlas.append(Texture(colorsArray, width, height,
                        target=gl.GL_TEXTURE_RECTANGLE,
                        internalFormat=gl.GL_RGBA16UI,
                        pixFormat=gl.GL_RGBA_INTEGER))
            elif colored == GlyphTypes.CBDT_COLOR:
                colorsArray = np.ascontiguousarray(self.table['colorsArray'])
                height, width, bands = colorsArray.shape
                self.atlas.append(Texture(colorsArray, width, height,
                        target=gl.GL_TEXTURE_2D, internalFormat=gl.GL_RGBA,
                        pixFormat=gl.GL_RGBA))

    def bindAtlas(self):
        """ Bind all atlases.
        """
        for i, atlas in enumerate(self.atlas):
            atlas.bind(i)

    def unbindAtlas(self):
        """ Unbind all atlases.
        """
        for i, atlas in enumerate(self.atlas):
            atlas.unbind()

    def chars(self, glyphType):
        """ Return all characters for glyph type.

            Args:
                glyphType (:class:`litGL.fontDistiller.GlyphTypes`):
                    glyph type

            Returns:
                tuple: unicode characters for existing glyphs
        """
        return [chr(k) for k in self.cmap(glyphType)]

    def cmap(self, glyphType):
        """ Return character map for glyph type.

            Args:
                glyphType (:class:`litGL.fontDistiller.GlyphTypes`):
                    glyph type

            Returns:
                tuple: unicode codepoints for existing glyphs
        """
        cmap = ()
        for key, g in self.table['glyphs'].items():
            if glyphType in g['glyphTypes']:
                cmap += (key, )
        return cmap

    def getKerning(self, right, left):
        """ Return kerning values (horizontal, vertical) for (`right`,
            `left`) pair of unicode characters.

            Args:
                right (str): right unicode character
                left (str): left unicode character

            Returns:
                tuple: (horizontal, vertical) kerning values
        """
        kern = self.table['kerning_table'].get((right, left))
        if kern is None:
            return 0.0, 0.0
        return kern, 0.0

    def getGlyph(self, codepoint, glyphType=GlyphTypes.BASE):
        """ Return glyph for unicode character with code point
            `codepoint`.

            Args:
                codepoint (int): unicode code point
                glyphType (:class:`litGL.fontDistiller.GlyphTypes`):
                    glyph type

            Returns:
                :class:`Glyph`: glyph for codepoint
        """
        try:
            glyph = self.table['glyphs'][codepoint]
        except KeyError:
            self.logger.info(f"Code point '{codepoint}' not found"
                    " in glyphs table.")
        else:
            if glyphType in glyph['glyphTypes']:
                if glyphType in (
                        GlyphTypes.BASE, GlyphTypes.CBDT_COLOR,
                        GlyphTypes.EBDT_COLOR):
                    pass
                elif glyphType == GlyphTypes.LAYER_COLOR:
                    # glyph.copy is NOT sufficient, deepcopy is needed
                    # otherwise glyph['vertices'] are replaced permanently
                    glyph = copy.deepcopy(glyph)
                    if 'gpc' in glyph:
                        vertices = glyph['vertices']
                        r = 0
                        c = glyph['gpc']
                        if c > 4095:
                            r = vertices['gpc'] / 4095
                            c = vertices['gpc'] - (4095 * r)
                        vertices['gp'][:, 0] = c
                        vertices['gp'][:, 1] = r
                        glyph['vertices'] = vertices
                else:
                    raise NotImplementedError(f"glyphType {glyphType}"
                            " not implemented!")
                return glyph

    @staticmethod
    def getGlyphTypes(fontFile):
        data = gzip.GzipFile(fontFile)
        return pickle.loads(data.read())['glyphTypes']

