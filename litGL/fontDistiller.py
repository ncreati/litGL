""" Font glyphs distiller

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
import os
import pickle
import time
from PIL import Image
import io
from fontTools.unicode import Unicode
from fontTools.pens.cu2quPen import Cu2QuPen
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib import TTFont, newTable, TTLibError
import numpy.lib.recfunctions as nrec
import numpy as np
import logging
import sys
import gzip
import datetime
from itertools import chain
from pathlib import Path
from enum import IntEnum

# Local imports
from .glyph import Glyph, BitmapAtlas

# -----------------------------------------------------------------------------
def key_from_value(v, d):
    """ Return key for value `v` in dictionary `d`.

        Args:
            v (object): value
            d (dict): dictionary

        Returns:
            object: key for value `v` in dictionary `d` or None
    """
    ret = []
    for k in d:
        if d[k] == v:
            ret.append(k)
    if len(ret) > 1:
        logging.getLogger(__name__).debug(f"Duplicate key for value '{v}'"
                " in the character map")
    return ret[0] if ret else None

# -----------------------------------------------------------------------------
def normalize_outlines(data, xs, ys):
    """ Normalize Bezier curves vertices coordinates.

        Args:
            data (:class:`numpy.ndarray`): structured array with Bezier curves
                vertices
            xs (float): x scale factor
            ys (float): y scale factor
    """
    scale = np.asarray([xs, ys], np.float32)

    for contour in data:
        contour['p1'] /= scale
        contour['p2'] /= scale
        contour['p3'] /= scale

# -----------------------------------------------------------------------------
def remove_cubic(data):
    """ Remove cubic dummy vertices from the structured array of Bezier curves.

        Args:
            data (:class:`numpy.ndarray`)): structured array with Bezier
                curves vertices
    """
    newData = [nrec.drop_fields(contour, 'p3') for contour in data]
    return [nrec.rename_fields(contour, {'p4':'p3'}) for contour in newData]

# -----------------------------------------------------------------------------
def sort_bands(outlines, bands, mode='x'):
    """ Sort Bezier curves inside band in direction specified by `mode`.

        Args:
            outlines (:class:`numpy.ndarray`): structured array with Bezier
                curves vertices
            bands (list): list of Bezier curves in each band
            mode (str): direction of sorting: either 'x' or 'y'

        Returns:
            list: sorted bands in the requested direction
    """
    sortedBands = []
    index = 0 if mode == 'x' else 1
    for group in bands:
        curves = outlines[list(group)]
        cMax = np.nanmax([curves['p1'][:, index], curves['p2'][:, index],
                curves['p3'][:, index]], axis=0)
        ind = np.argsort(cMax)[::-1]
        sortedBands.append(group[ind])
    return sortedBands

# -----------------------------------------------------------------------------
def bandIntersections(outlines, bands):
    """ Return index of bezier curves inside bands.

        Args:
            outlines (:class:`numpy.ndarray`): structured array with Bezier
                curves vertices
            bands (tuple): number of bands in the horizontal and vertical
                direction

        Returns:
            tuple: sorted bands in x and y direction
    """
    nh, nv = bands
    hBands = np.linspace(0, 1, nh+1)
    vBands = np.linspace(0, 1, nv+1)

    hb = []
    # Parallel test cause artefact
    ind = ((outlines['p1'][:, 1] == outlines['p2'][:, 1])
            & (outlines['p2'][:, 1] == outlines['p3'][:, 1]))
    ind = np.logical_not(ind)
    y = np.vstack((outlines['p1'][:, 1], outlines['p2'][:, 1],
        outlines['p3'][:, 1])).T[ind]
    yMin = np.min(y, axis=1)
    yMax = np.max(y, axis=1)
    for b0, b1 in zip(hBands[:-1], hBands[1:]):
        ind2 = (yMin > b1) | (yMax < b0)
        i = np.flatnonzero(ind)
        hb.append(i[np.logical_not(ind2)])

    ## Work on vertical bands
    vb = []
    # Parallel test cause artefacts
    ind = ((outlines['p1'][:, 0] == outlines['p2'][:, 0])
            & (outlines['p2'][:, 0] == outlines['p3'][:, 0]))
    ind = np.logical_not(ind)
    x = np.vstack((outlines['p1'][:, 0], outlines['p2'][:, 0],
        outlines['p3'][:, 0])).T[ind]
    xMin = np.min(x, axis=1)
    xMax = np.max(x, axis=1)
    for b0, b1 in zip(vBands[:-1], vBands[1:]):
        ind2 = (xMin > b1) | (xMax < b0)
        i = np.flatnonzero(ind)
        vb.append(i[np.logical_not(ind2)])

    # Sort curve in ascending order
    # Horizontal bands from max x
    hb = sort_bands(outlines, hb, mode='x')
    # Verticalal bands from max y
    vb = sort_bands(outlines, vb, mode='y')

    return hb, vb

# -----------------------------------------------------------------------------
def cff_to_glyf(face):
    """ Convert the CFF/CFF2 table to a glyf table

        Args:
            face (:class:`fontTools.ttLib:TTFont`): instance

        Returns:
            dict: of :class:`fontTools.ttLib.tables._g_l_y_f.Glyph` instances
    """
    def glyphs_to_quadratic(glyphs, max_err=1.0, reverse_direction=True):
        quadGlyphs = {}
        for gname in glyphs.keys():
            glyph = glyphs[gname]
            ttPen = TTGlyphPen(glyphs)
            cu2quPen = Cu2QuPen(ttPen, max_err,
                                reverse_direction=reverse_direction)
            glyph.draw(cu2quPen)
            quadGlyphs[gname] = ttPen.glyph()
        return quadGlyphs

    # -------------------------------------------------------------------------
    def update_hmtx(face, glyf):
        hmtx = face["hmtx"]
        for glyphName, glyph in glyf.glyphs.items():
            if hasattr(glyph, 'xMin'):
                hmtx[glyphName] = (hmtx[glyphName][0], glyph.xMin)

    # -------------------------------------------------------------------------
    assert face.sfntVersion == "OTTO"
    assert "CFF " in face

    glyphOrder = face.getGlyphOrder()

    face["loca"] = newTable("loca")
    face["glyf"] = glyf = newTable("glyf")
    glyf.glyphOrder = glyphOrder
    glyf.glyphs = glyphs_to_quadratic(face.getGlyphSet())
    del face["CFF "]
    glyf.compile(face)
    update_hmtx(face, glyf)

    face["maxp"] = maxp = newTable("maxp")
    maxp.tableVersion = 0x00010000
    maxp.maxZones = 1
    maxp.maxTwilightPoints = 0
    maxp.maxStorage = 0
    maxp.maxFunctionDefs = 0
    maxp.maxInstructionDefs = 0
    maxp.maxStackElements = 0
    maxp.maxSizeOfInstructions = 0
    maxp.maxComponentElements = max(
        len(g.components if hasattr(g, 'components') else [])
        for g in glyf.glyphs.values())
    maxp.compile(face)

    post = face["post"]
    # default 'post' table format
    post.formatType = 2.0
    post.extraNames = []
    post.mapping = {}
    post.glyphOrder = glyphOrder
    try:
        post.compile(face)
    except OverflowError:
        post.formatType = 3
        logging.debug("Dropping glyph names, they do not fit in 'post' table.")

# -----------------------------------------------------------------------------
def glyferize(glyf, name, codePoint=None):
    """ Extract glyph attributes from the TTF glyf table

        Args:
            glyf (:class:`fontTools.ttLib.tables._g_l_y_f.Glyph`): instance
            name (str): glyph name
            codePoint (int): unicode code point

        Returns:
            :class:`Glyph`: new Glyph instance
    """
    g = Glyph(name)
    g.codePoint = codePoint
    g.index = glyf.getGlyphID(name)
    g.charCode = chr(g.codePoint) if g.codePoint else None
    if g.charCode is not None:
        g.description = Unicode[ord(g.charCode)]
    else:
        g.description = None
    return g

# =============================================================================
class GlyphTypes(IntEnum):
    #: Basic (monochrome) glyph
    BASE = 0
    #: Colored glyph with layered structure
    LAYER_COLOR = 1
    #: Colored Bitmap glyph
    CBDT_COLOR = 2
    #: Black and white Bitmap glyph
    EBDT_COLOR = 3
    #: Colored Bitmap glyph (Apple implementation)
    SBIX_COLOR = 4
    #: SVG glyph
    SVG_COLOR = 5

# =============================================================================
class FontDistiller:
    #: The suffix for the distilled fonts
    EXT = '.nbf'
    #: The default folder for distilled fonts
    NBF_DIR = Path.joinpath(Path.home(), ".local/share/fonts/nbf")
    def __init__(self, fontFile, bands=0):
        """ Create a FontDistiller instance.

            Args:
                fontFile (str): pathname of font file
                bands (int): number of horizontal and vertical bands,
                    0 means automatic selection
        """
        try:
            self.face = TTFont(fontFile, lazy=False, allowVID=True)
        except TTLibError as e:
            raise SystemExit(f"Cannot distill {fontFile}: reason is"
                    f" '{e}'.")
        #Note: It is strongly advised that you do not add any handlers
        #other than NullHandler to your library’s loggers.
        #This is because the configuration of handlers is the prerogative
        #of the application developer who uses your library.
        #The application developer knows their target audience and what
        #handlers are most appropriate for their application:
        #if you add handlers ‘under the hood’, you might well interfere
        #with their ability to carry out unit tests and deliver logs which
        #suit their requirements.
        self.logger = logging.getLogger(__name__)

        self.logger.info('Face defined tables:\n==>\t%s' % (
            "\n==>\t".join(sorted(self.face.keys()))))
        self.fontFile = fontFile
        self.colored = None
        self.colorsArray = None
        self.colorsArrayShape = None
        self.curvesArray = None
        self.bandsArray = None

        # dict of glyphs unicode code point(integers) as keys and glyph names
        # as values
        self.cmap = self.face.getBestCmap()

        # bands 0 means automatic selection
        self.nBands = bands
        self.glyphs = {}

        # HEAD: https://docs.microsoft.com/en-us/typography/opentype/spec/head
        # This table gives global information about the font.
        head = self.face.get('head')

        if head.fontDirectionHint == 0:
            fontDirection = 'mixed'
        elif head.fontDirectionHint == 1:
            fontDirection = 'lr'
        elif head.fontDirectionHint == -1:
            fontDirection = 'rl'
        else:
            fontDirection = 'mixed'

        self.kerningTable = {}
        if self.face.has_key('kern'):
            kern = self.face.get('kern')
            self.logger.info(f'The face has kerning')
            if len(kern.kernTables) > 1:
                self.logger.info('Font has more than 1 kerning tables.')
            self.kerningTable = kern.kernTables[0].kernTable

        linespace = (self.face.get('hhea').ascent
                - self.face.get('hhea').descent
                + self.face.get('hhea').lineGap)

        # HMTX: https://docs.microsoft.com/en-us/typography/opentype/spec/hmtx
        # Glyph metrics used for horizontal text layout
        if self.face.has_key('hmtx'):
            layout = 'h'
            self.logger.info(f'The face has horizontal layout')
        # VMTX: https://docs.microsoft.com/en-us/typography/opentype/spec/vmtx
        # Glyph metrics used for vertical text layout
        elif self.face.has_key('vmtx'):
            layout = 'v'
            self.logger.info(f'The face has vertical layout')

        # Without CPAL color has no meaning
        # COLR: https://docs.microsoft.com/en-us/typography/opentype/spec/colr
        # contains a list of colored glyph and their associated glyph layers
        # CPAL: https://docs.microsoft.com/en-us/typography/opentype/spec/cpal
        # The palette table is a set of one or more palettes, each containing
        # a predefined number of color records
        self.coloredGlyphs = {}
        self.cglyphs = {}
        if self.face.has_key('COLR') and self.face.has_key('CPAL'):
            self.colored = GlyphTypes.LAYER_COLOR
            # Dict of font with colors each value is a list of records
            cl = self.face.get('COLR').ColorLayers
            for item, values in cl.items():
                self.coloredGlyphs[item] = [vars(entry) for entry in values]
            self.logger.info(f'The face has colors')

            # Extract the color palette
            cpal = self.face.get('CPAL')
            # cpal.palettesTypes can be:
            # USABLE_WITH_LIGHT_BACKGROUND 0
            # USABLE_WITH_DARK_BACKGROUND 1

            self.palettes = []
            for paletteType in cpal.palettes:
                rgba = np.zeros((cpal.numPaletteEntries, 4), np.uint8)
                for i, color in enumerate(paletteType):
                    rgba[i] = (color.red, color.green, color.blue, color.alpha)
                self.palettes.append(rgba)

        # Colord Bitmap as glyphs
        # CBDT: https://docs.microsoft.com/en-us/typography/opentype/spec/cbdt
        # The CBDT table is used to embed color bitmap glyph data.
        # CBLC: https://docs.microsoft.com/en-us/typography/opentype/spec/cblc
        # The CBLC table provides embedded bitmap locators
        if 'CBDT' in self.face:
            # Bitmap font are stored as byte string of PNG files
            self.colored = GlyphTypes.CBDT_COLOR
        # Colord Bitmap as glyphs
        # SBIX: https://docs.microsoft.com/en-us/typography/opentype/spec/sbix
        # This table provides access to bitmap data in a standard graphics
        # format, such as PNG, JPEG or TIFF.
        if self.face.has_key('sbix'):
            self.colored = GlyphTypes.SBIX_COLOR
            self.logger.critical(f'The face has colored bitmap as glyphs:'
                    ' UNSUPPORTED.')

        # Colord Bitmap as glyphs
        # SVG: https://docs.microsoft.com/en-us/typography/opentype/spec/svg
        # This table contains SVG descriptions for some or all of the glyphs
        # in the font.
        if self.face.has_key('SVG'):
            self.colored = GlyphTypes.SVG_COLOR
            self.logger.critical(f'The face has SVG glyphs: UNSUPPORTED')

        # HHEA: https://docs.microsoft.com/en-us/typography/opentype/spec/hhea
        # This table contains information for horizontal layout.
        hhea = self.face.get('hhea')

        # OS/2: https://docs.microsoft.com/en-us/typography/opentype/spec/os2
        # The OS/2 table consists of a set of metrics and other data that are
        # required in OpenType fonts.

        # POST: https://docs.microsoft.com/en-us/typography/opentype/spec/post
        # This table contains additional information needed to use TrueType or
        # OpenType™ fonts on PostScript printers. This includes data for the
        # FontInfo dictionary entry and the PostScript names of all the glyphs

        # MAXP: https://docs.microsoft.com/en-us/typography/opentype/spec/maxp
        # This table establishes the memory requirements for this font.

        # Save several face parameters
        self.fontTable = {
                # map character name to integer ordinal value
                'cmap'                : self.cmap,
                'ascent'              : self.face.get('hhea').ascent,
                'lineGap'             : self.face.get('hhea').lineGap,
                'descent'             : self.face.get('hhea').descent,
                'linespace'           : linespace,
                'family_name'         : self.face.get('name').names[1].toStr(),
                'glyphs'              : {},
                'glyphTypes'          : [],
                'colored'             : self.colored,
                'kerning_table'       : self.kerningTable,
                'fontRevision'        : head.fontRevision,
                'minLeftSideBearing'  : self.face.get(
                        'hhea').minLeftSideBearing,
                'minRightSideBearing' : self.face.get(
                        'hhea').minRightSideBearing,
                'fontDirection'       : fontDirection,
                'layout'              : layout,
                'max_advance_width'   : self.face.get('hhea').advanceWidthMax,
                'num_glyphs'          : self.face.get('maxp').numGlyphs,
                'style_name'          : self.face.get('name').names[2].toStr(),
                'underline_position'  : self.face.get(
                        'post').underlinePosition,
                'underline_thickness' : self.face.get(
                        'post').underlineThickness,
                'box'                 : (
                        self.face.get('head').xMin,
                        self.face.get('head').xMax,
                        self.face.get('head').yMin,
                        self.face.get('head').yMax
                        ),
                'ySubscriptXSize'     : self.face.get('OS/2').ySubscriptXSize,
                'ySubscriptYSize'     : self.face.get('OS/2').ySubscriptYSize,
                'ySubscriptXOffset'   : self.face.get(
                        'OS/2').ySubscriptXOffset,
                'ySubscriptYOffset'   : self.face.get(
                        'OS/2').ySubscriptYOffset,
                'ySuperscriptXSize'   : self.face.get(
                        'OS/2').ySuperscriptXSize,
                'ySuperscriptYSize'   : self.face.get(
                        'OS/2').ySuperscriptYSize,
                'ySuperscriptXOffset' : self.face.get(
                        'OS/2').ySuperscriptXOffset,
                'ySuperscriptYOffset' : self.face.get(
                        'OS/2').ySuperscriptYOffset,
                'yStrikeoutSize'      : self.face.get('OS/2').yStrikeoutSize,
                'yStrikeoutPosition'  : self.face.get(
                        'OS/2').yStrikeoutPosition,
                'fsFirstCharIndex'    : self.face.get('OS/2').fsFirstCharIndex,
                'fsLastCharIndex'     : self.face.get('OS/2').fsLastCharIndex,
                'units_per_EM'        : self.face.get('head').unitsPerEm,
                'height'              : (self.face.get('head').yMax
                        - self.face.get('head').yMin),
                'width'               : (self.face.get('head').xMax
                        - self.face.get('head').xMin),
                }

        if self.face.get('head').flags == 4:
            self.logger.critical('Check LTSH table!!')

        t0 = time.time()
        glyf = self.face.get('glyf')
        if not glyf:
            #######################################################
            # If glyf table is missing there are some alternatives:
            # CFF/CFF2 Adobe glyph table or Bitmap glyphs
            self.logger.info(f'No glyf table available')
            # Try CFF
            cff = self.face.get('CFF ') or self.face.get('CFF2')
            if cff:
                cff_to_glyf(self.face)
                # extract the builded glyf table
                glyf = self.face['glyf']
                self.logger.debug(f'Glyphs table from CFF/CFF2')

            if 'CBDT' in self.face or 'EBDT' in self.face:
                self.logger.debug(f'CBDT/EBDT table is available')

        if glyf:
            if self.extract(glyf):
                self.fontTable['glyphTypes'].append(GlyphTypes.BASE)

        if self.colored == GlyphTypes.LAYER_COLOR:
            self.buildColorsLayerArray()
            self.fontTable['glyphTypes'].append(GlyphTypes.LAYER_COLOR)
            self.logger.debug(f'Array of colored layer created')
        elif self.colored == GlyphTypes.CBDT_COLOR:
            self.buildColorCBDTArray()
            self.fontTable['glyphTypes'].append(GlyphTypes.CBDT_COLOR)
            cblc = self.face['CBLC']
            hori = cblc.strikes[0].bitmapSizeTable.hori
            self.logger.debug(f'Array of colored bitmap created')

        if self.fontTable['glyphTypes']:
            self.logger.info(f'Font {self.fontFile} processed in'
                    f' {(time.time() - t0):.2f} s.')

    def makeGlyphBands(self, g, nBands, layered=False):
        """ Calculate glyph curves intersection with defined horizontal and
            vertical bands, arrange curves array to be saved in the curves
            texture.

            Args:
                g (:class:`fontTools.ttLib.tables._g_l_y_f.Glyph`): glyph
                nBands (int): number of horizontal and vertical bands,
                              0 means automatic selection
                layered (bool): True if the glyph is a colored layer one

            Returns:
                :class:`numpy.ndarray`: intersections of the glyph Bezier
                curves with the bands
        """
        # Build curve array
        newData = remove_cubic(g.data)
        if not layered:
            normalize_outlines(newData, g.width, g.height)

        scale = np.asarray([1.0, 1.0], np.float32)

        # Build bandsData
        nCurves = sum(g.bezierCount)
        if nBands == 0:
            nBands = min(
                    int(np.round(nCurves / self.curves_x_bands + 0.5)), 16)
        self.logger.debug(f"{self.count}: character <{g.charCode}> ({g.name})"
            f" codePoint {g.codePoint} nCurves = {nCurves}"
            f" curves_x_bands = {self.curves_x_bands}.")
        g.nBands = (nBands, nBands)
        for contour in newData:
            for elem in contour:
                self.curvesData.append([np.concatenate(
                            (elem['p1'] / scale, elem['p2'] / scale))])
                if (np.fmod(1 + len(self.curvesData), self.width) == 0
                        and len(self.curvesData) > 0):
                    # sposta tutto alla riga successiva
                    # metti la p3 e passa oltre
                    self.curvesData.append([np.concatenate(
                        (elem['p3'] / scale, self.a_00))])
                    self.skipIndex = len(self.curvesData) - 1
                    self.logger.debug(
                            f'A curve was shifted to the next row at'
                            f' -> {len(self.curvesData)} for glyph'
                            f' -> {g.name} count: {self.count} <')
                    continue
            self.curvesData.append([np.concatenate(
                (elem['p3']/scale, self.a_00))])
            if (np.fmod(1 + len(self.curvesData), self.width) == 0
                    and len(self.curvesData) > 0):
                self.logger.debug(
                        f'Strange A curve need to be shifted to next row at'
                        f' ->{len(self.curvesData)} for glyph > {g.name}'
                        f' count: {self.count} <')
                self.curvesData.append([self.a_0000])
                self.skipIndex = len(self.curvesData) - 1

        # Glyph Header
        # calculate curves vs bands
        horzBands, vertBands = bandIntersections(np.hstack(newData),
                (nBands, nBands))

        # numero bande orizzontali offset all'indice delle curve
        # numero bande vertricali offset all'indice delle curve
        # h1(bande), h2(offset)
        # Bande orizzontali
        h = [len(iBand) for iBand in horzBands]
        # Bande verticali
        v = [len(iBand) for iBand in vertBands]
        # Appendi alla lista delle bande per ogni glifo
        glyphBands = np.hstack((h, v)).tolist()
        self.iBandCount.extend(glyphBands)

        g.glyphParam = [self.x0, self.y0, nBands, nBands]

        self.x0 += len(glyphBands)
        self.y0 += 0

        # Wrap is x exceeed texture wifdth
        self.y0 += self.x0 >> 12
        self.x0 &= 4095

        g.vertices['gp'] = g.glyphParam

        # Fix header horixontal and vertical offset accounting for index
        # jump at each contour end
        jumps = np.cumsum(g.bezierCount[:-1])
        if len(g.bezierCount) > 1:
            for i, jump in enumerate(jumps):
                for hb in horzBands:
                    ind = np.flatnonzero((hb >= (jump + i)))
                    hb[ind] += 1

                for vb in vertBands:
                    ind = np.flatnonzero((vb >= (jump + i)))
                    vb[ind] += 1

        horzBands = np.hstack(horzBands)
        vertBands = np.hstack(vertBands)
        # Curve index in the band data for the current glyph
        glyphBands = np.hstack((horzBands, vertBands))
        # Add the curve offset for the current glyph
        glyphBands += self.curvesOffset

        # Manage shift of curve index at width limit
        if self.skipIndex is not None:
            ind = glyphBands >= self.skipIndex
            glyphBands[ind] += 1
            ## Reset the skipIndex
            self.skipIndex = None

        self.curvesOffset = len(self.curvesData)
        return glyphBands

    def compositItem(self, g, name, item, glyf):
        """ Process a composite glyph

            Args:
                g (:class:`litGL.glyph.Glyph`): glyph
                name (str): glyph name
                item (:class:`fontTools.ttLib.tables._g_l_y_f.Glyph`):
                    component glyph
                glyf (:class:`fontTools.ttLib.tables._g_l_y_f.table__g_l_y_f`):
                    father glyph
        """
        g.setMetric(item, self.metric[name])
        # Build compound contours and create the outline arrays
        outlines = []

        for c in item.components:
            # trasform (xx, xy, yx, yy, x, y). [[xx, xy],[yx, yy]] =
            # 2x2 matrix transform
            # x, y affine transform
            componentGlyphName, transform = c.getComponentInfo()
            # Two possible options:
            if componentGlyphName in self.glyphs.keys():
                # Component already extracted
                contour, bezierCount, scale = Glyph.buildOutlineArray(
                        self.glyphs[componentGlyphName].contours)
                g.bezierCount.extend(bezierCount)
            else:
                # Glyph componennt must be extracted
                componentItem = glyf.glyphs[componentGlyphName]
                cg = glyferize(glyf, componentGlyphName)
                cg.setMetric(componentItem, self.metric[componentGlyphName])
                if componentItem.numberOfContours != 0:
                    cg.extractContours(componentItem.getCoordinates(glyf))
                    cg.contours2array(
                            x0=cg.xMin, y0=cg.yMin)
                    cg.buildVertices()
                    glyphBands = self.makeGlyphBands(cg, self.nBands)

                    # Add the glyph band curves index to the band data
                    self.bandsData.extend(glyphBands)

                self.count += 1
                self.glyphs[componentGlyphName] = cg
                self.fontTable['glyphs'][cg.cp()] = cg.dataTable
                g.bezierCount.extend(cg.bezierCount)
                contour, bezierCount, scale = Glyph.buildOutlineArray(
                        self.glyphs[componentGlyphName].contours)

            # Apply transform
            rescaling = np.asarray(transform[:4]).reshape(2, 2)
            offset = np.asarray(transform[4:])-np.array([g.xMin, g.yMin])
            for array in contour:
                for field in array.dtype.fields:
                    array[field] = ((np.dot(array[field], rescaling) + offset))
            outlines.append(contour)
        g.data = list(chain(*outlines))

    def extract(self, glyf):
        """ Extract glyphs from a font file.

            Args:
                glyf (:class:`fontTools.ttLib.tables._g_l_y_f.table__g_l_y_f`):
                    glyf table
        """
        self.count = 0
        glyf.compile(self.face)

        # Glyph metric
        if self.face.has_key('hmtx'):
            self.metric = self.face.get('hmtx')
        elif self.face.has_key('vmtx'):
            self.metric = self.face.get('vmtx')
        else:
            raise ValueError('Metric information is missing')

        if self.cmap is None:
            self.logger.critical(f"Font {self.fontFile} has no CMAP,"
                    " cannot distill!")
            return
        self.reversed_cmap = dict(map(reversed, self.cmap.items()))

        # Loop over all glyphs in the glyf table extracting also not Unicode
        # glyph. Char map has only unicode glyphs
        composite = {}

        self.a_00 = np.zeros((2, ), np.float16)
        self.a_0000 = np.zeros((4, ), np.float16)
        self.width = 4096
        self.curves_x_bands = 2048 // 16
        self.skipIndex = None
        self.curvesData = []
        self.bandsData = []
        self.iBandCount = []
        iBandOffset = []
        self.x0 = 0
        self.y0 = 0
        self.curvesOffset = 0

        xMin, xMax, yMin, yMax = self.fontTable['box']
        g_width = xMax-xMin
        g_height = yMax-yMin
        for code, name in self.cmap.items():
            item = glyf[name]

            g = glyferize(glyf, name, code)
            g.setMetric(item, self.metric[name])
            #g.buildVertices()

            # Manage composite glyphs, composite glyphs has a number of
            # contours = -1
            if item.isComposite():
                self.logger.debug(
                        f'Composite glyph -> {g.name} count: {self.count} <')
                self.compositItem(g, name, item, glyf)
                #g.buildVertices()

            else:
                # Create glyph instance and process the font glyph item
                g.extractContours(item.getCoordinates(glyf))
                g.contours2array(x0=g.xMin, y0=g.yMin)
            g.buildVertices()
            # Composit glyph
            if item.isComposite() and len(g.data):
                glyphBands = self.makeGlyphBands(g, self.nBands)
                self.bandsData.extend(glyphBands)
            # Glyph has contours
            if item.numberOfContours > 0:
                glyphBands = self.makeGlyphBands(g, self.nBands)
                self.bandsData.extend(glyphBands)

            self.count += 1

            # Save the glyph data
            self.glyphs[code] = g
            g.dataTable['glyphTypes'].append(GlyphTypes.BASE)
            self.fontTable['glyphs'][code] = g.dataTable
            if name in self.coloredGlyphs:
                g.dataTable['glyphTypes'].append(GlyphTypes.LAYER_COLOR)

        # Extract layered colored glyphs if any
        if self.colored == GlyphTypes.LAYER_COLOR:
            toDelete = []
            for unic, layers in self.coloredGlyphs.items():
                father = glyf[unic]
                if father.numberOfContours == 0:
                    toDelete.append(unic)
                    continue
                g_width = father.xMax - father.xMin
                g_height = father.yMax - father.yMin
                signs = dict()
                for entry in layers:
                    name = entry['name']
                    item = glyf[name]

                    code = key_from_value(name, self.cmap)
                    g = glyferize(glyf, name, code)
                    g.setMetric(item, self.metric[name])
                    g.extractContours(item.getCoordinates(glyf))
                    g.contours2array(x0=father.xMin, y0=father.yMin,
                            xs=g_width, ys=g_height)
                    g.buildVertices()
                    glyphBands = self.makeGlyphBands(g, self.nBands,
                            layered=True)
                    self.bandsData.extend(glyphBands)
                    signs[name] = g
                self.cglyphs[unic] = signs

            for item in toDelete:
                del self.coloredGlyphs[item]

        # Hack to extract .notdef glyph if it has curves
        name = '.notdef'
        item = glyf.get(name)
        if item and item.numberOfContours > 0:
            notdef = glyferize(glyf, name)
            notdef.setMetric(item, self.metric[name])
            notdef.extractContours(item.getCoordinates(glyf))
            notdef.contours2array(x0=notdef.xMin, y0=notdef.yMin)
            notdef.buildVertices()
            glyphBands = self.makeGlyphBands(notdef, self.nBands)
            self.bandsData.extend(glyphBands)
            self.glyphs[-1] = notdef
            g.dataTable['glyphTypes'].append(GlyphTypes.BASE)
            self.fontTable['glyphs'][-1] = notdef.dataTable

        iBandOffset = len(self.iBandCount) + np.concatenate(
                [[0], np.cumsum(self.iBandCount[:-1])])
        self.iBandCount = np.asarray(self.iBandCount)
        # Values that come out the width in the shader must be corrected in
        # the following while because at avery adjustment all subsequent
        # data are moved.
        ss = 0
        while True:
            ind = np.flatnonzero(
                    ((self.iBandCount + iBandOffset) >> 12)
                    != (iBandOffset >> 12))
            if ind.size == 0:
                break
            toAdd = self.width - (iBandOffset[ind] & 4095)
            #NOTE: non capisco il commento, qui non metti zeri.
            # Andrebbero messi? E dove?
            # Vanno messi gli zeri ai banddata che riflettono lo
            # spostamento
            bandDataIndex = iBandOffset[ind][0] - iBandOffset[0]
            self.bandsData = (self.bandsData[:bandDataIndex]
                    + toAdd[0] * [0] + self.bandsData[bandDataIndex:])
            #NOTE: tradurre
            # Si aggiunge solo il primo offset dal primo valkore di ind in poi
            # e si ricalcola tutto
            iBandOffset[ind[0]:] += toAdd[0]
            ss += 1

        texelWidth = 4
        height = int(1 + len(self.curvesData)/self.width)
        curvesLength = height * self.width * texelWidth
        self.curvesArray = np.zeros((self.width * height * texelWidth),
                np.float16)
        self.curvesArrayShape = (self.width, height, texelWidth)
        data = np.asarray(self.curvesData).astype(np.float16).ravel()
        self.curvesArray[:data.size] = data

        numElements = (len(self.iBandCount) + len(iBandOffset)
                + len(self.bandsData))
        height = int(1 + round(numElements / self.width))
        # array bands
        texelWidth = 2
        bandsArrayLength = self.width * height * texelWidth

        maxInt = iBandOffset.max()
        if maxInt <= np.iinfo(np.uint16).max:
            dtype = np.uint16
            self.logger.debug('Bands Array is of type uint16')
        elif np.iinfo(np.uint16).max < maxInt <= np.iinfo(np.uint32).max:
            dtype = np.uint32
            self.logger.debug('Bands Array is of type uint32')
        else:
            raise ValueError('Index max malue in the bansArray exceded'
                    ' the maximum allowed uint32 maximum value!!!')

        self.bandsArray = np.zeros((self.width * height * texelWidth), dtype)

        # Creates bands array header
        header = np.vstack((self.iBandCount, iBandOffset)).T.astype(dtype)
        self.bandsArray[:header.size] = header.ravel()

        data = np.zeros((len(self.bandsData), 2), dtype)
        data[:, 0] = self.bandsData

        # Wrap bandsData if x exceeeds texture wifth
        data[:, 1] = data[:, 0] >> 12
        data[:, 0] &= 0x0fff

        self.bandsArray[header.size:data.size+header.size] = data.ravel()
        self.bandsArrayShape = (self.width, height, texelWidth)
        self.logger.debug(f'Bands Array shape: {self.bandsArrayShape}')
        return True

    def save(self, output, defaultDir=True):
        """ Save fontTable to pickled file.

            Args:
                outfile (str): pathname of output file

            Returns:
                str: pathname of output file
        """
        if not self.fontTable['glyphTypes']:
            return
        if self.curvesArray is not None:
            self.fontTable['curvesArray'] = self.curvesArray
            self.fontTable['curvesArrayShape'] = self.curvesArrayShape
            self.fontTable['bandsArray'] = self.bandsArray
            self.fontTable['bandsArrayShape'] = self.bandsArrayShape
        self.fontTable['creationTime'] = str(datetime.datetime.now())
        if self.colored == GlyphTypes.LAYER_COLOR:
            self.fontTable['colorsArray'] = self.colorsArray
            self.fontTable['colorsArrayShape'] = self.colorsArrayShape
        elif self.colored == GlyphTypes.CBDT_COLOR:
            self.fontTable['colorsArray'] = self.atlas
        data = gzip.compress(pickle.dumps(self.fontTable))

        if defaultDir:
            Path(FontDistiller.NBF_DIR).mkdir(parents=True, exist_ok=True)
            output = Path.joinpath(FontDistiller.NBF_DIR, output)
        fid = open(output, 'wb')
        fid.write(data)
        fid.close()
        self.logger.info(f"Font file {self.fontFile} has been"
                f" distilled to {output} file")
        return str(output)

    def buildColorsLayerArray(self):
        """ Build array for layered colored glyphs
        """
        # Add Composite Glyph for colored ones
        entries = []
        glyf = self.face.get('glyf')
        for unic, values in self.coloredGlyphs.items():

            if unic not in self.cmap.values():
                continue

            glyhBaseCount = len(values)
            base = []
            for item in values:
                colorId = item['colorID']
                glyphName = item['name']
                baseGlyphLoc = self.cglyphs[unic][glyphName].glyphParam[:3]
                base.append((*baseGlyphLoc, colorId))
            code = key_from_value(unic, self.cmap)
            entries.append((self.glyphs[code].description, glyhBaseCount,
                        code, base))
            self.logger.debug(f'Added colored glyph: {unic}, {len(values)}')
        header = []
        layers = []
        last = 0

        for i, item in enumerate(entries):
            self.fontTable['glyphs'][item[2]]['gpc'] = i
            header.append((item[1], last))
            layers.extend(item[-1])
            last += len(item[-1])

        header = np.asarray(header)
        header[:, 1] += len(header)

        h = np.hstack((header, np.zeros((len(header), 2))))

        layers = np.asarray(layers)
        ## Add curves array colors start index
        offsetToColor = len(h) + len(layers)
        layers[:, -1] += offsetToColor

        data = np.vstack((h, layers, self.palettes[0])).astype(np.uint16)

        texelWidth = 4
        width = 4096
        height = max(round(data.size / (width * texelWidth)), 1)

        self.colorsArray = np.zeros((width * height * texelWidth), np.uint16)
        self.colorsArray[:data.size] = data.ravel()

        self.colorsArrayShape = (width, height, texelWidth)
        self.logger.info(f'Colors Array shape: {self.colorsArrayShape}')

        self.fontTable['coloredGlyphs'] = [key_from_value(item, self.cmap)
                for item in self.coloredGlyphs.keys()]

    def buildColorCBDTArray(self):
        """ Build atlas array for glyphs defined as bitmaps
        """
        def glyferize(name, item):
            g = Glyph(name)
            g.codePoint = reversed_cmap.get(name)
            g.index = self.face.getGlyphID(name)
            g.charCode = chr(g.codePoint) if g.codePoint else None
            g.width = self.fontTable['width']
            g.height = self.fontTable['height']
            if g.charCode is not None:
                g.description = Unicode[ord(g.charCode)]
            tmp = Image.open(io.BytesIO(item.imageData))
            if tmp.getbands() == ('P',):
                g.array = np.asarray(tmp.convert('RGBA'))
            else:
                g.array = np.asarray(tmp)

            g.dataTable['height'] = g.height
            g.dataTable['width'] = g.width
            g.dataTable['horz_advance'] = g.width
            g.dataTable['vAdvance'] = 0
            g.dataTable['index'] = g.index
            offset = (metric.BearingX * (g.width / metric.width),
                    (metric.BearingY - metric.height)
                    * (g.height / metric.height))
            g.dataTable['offset'] = offset
            g.dataTable['xMax'] = g.width
            g.dataTable['xMin'] = 0
            g.dataTable['yMax'] = g.height
            g.dataTable['yMin'] = 0
            g.dataTable['printable'] = True
            g.dataTable['extra'] = False
            g.dataTable['extension'] = item.fileExtension
            g.dataTable['glyphTypes'].append(GlyphTypes.CBDT_COLOR)
            return g

        data = self.face['CBDT'].strikeData[0]
        atlas = BitmapAtlas()
        reversed_cmap = dict(map(reversed, self.cmap.items()))
        for i, name in enumerate(data):
            if not name in reversed_cmap:
                continue
            item = data[name]
            metric = item.metrics
            g = glyferize(name, item)
            g.buildVertices()

            x, y, _, _ = atlas.getFreeSpot(metric.width+1, metric.height+1)
            w = metric.width
            h = metric.height
            if x < 0:
                self.logger.critical('Cannot store glyph in atlas!'
                        ' Increase it! >> %d <<' % i)
                exit()
            atlas.data[y:y + h, x:x + w, :] = np.flipud(g.array)
            u0 = (x +     0.0) / float(atlas.width)
            v0 = (y +     0.0) / float(atlas.height)
            u1 = (x + w - 0.0) / float(atlas.width)
            v1 = (y + h - 0.0) / float(atlas.height)

            texcoords = [[u0, v0], [u1, v0], [u1, v1], [u0, v1]]

            g.vertices['tex'] = texcoords
            self.glyphs[name] = g
            if g.cp() in self.fontTable['glyphs']:
                self.fontTable['glyphs'][g.cp()] = g.dataTable
            else:
                self.fontTable['glyphs'][g.cp()] = g.dataTable
        self.atlas = atlas.data


# =============================================================================
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description=("Distill, i.e. extract glyphs from a ttf"
            " or odf file and save them to an nbf file for later"
            " usage by LitGL."))

    # Font file
    parser.add_argument('font', help='Input font file')

    # Output nbf font file
    parser.add_argument('output', help='Output %s file'
            % FontDistiller.EXT, default='')

    parser.add_argument('-b', '--bands', type=int, help='Number of bands',
            default=0)

    parser.add_argument('-f', '--force', action='store_true',
            help='Force overwriting of output file',
            default=False)

    parser.add_argument('-v', '--verbose', nargs='?', default=None,
            const='', help='Verbose processing (logging) level.'
            " Add more 'v' to increase verbosity. Must be the last option.")

    # Process options
    opts = parser.parse_args()
    if opts.verbose is not None:
        levels = (logging.CRITICAL, logging.ERROR, logging.WARNING,
                logging.INFO, logging.DEBUG, logging.NOTSET)
        try:
            level = levels[len(opts.verbose) + 1]
        except IndexError:
            level = levels[-1]

        try:
            from rich.logging import RichHandler
        except ImportError:
            rootLogger = logging.getLogger()
            console = logging.StreamHandler(sys.stdout)
            console.setFormatter(logging.Formatter(
                '%(asctime)s -  %(levelname)s - %(message)s',
                datefmt='%d-%b-%y %H:%M:%S',))
            rootLogger.addHandler(console)
            rootLogger.setLevel(level)
        else:
            FORMAT = '%(name)s.%(funcName)s: %(message)s'
            DATEFMT = "[%X]"
            logging.basicConfig(
                    level=level, format=FORMAT, datefmt=DATEFMT,
                    handlers=[RichHandler()])
            rootLogger = logging.getLogger("rich")

    if os.path.exists(opts.output) and not opts.force:
        raise SystemExit(f"\nOutput file {opts.output} exists,"
                " distillation aborted.")
    FontDistiller(opts.font, opts.bands).save(opts.output, defaultDir=False)
