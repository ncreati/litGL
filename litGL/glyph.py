""" Font glyphs extractor.

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
import math
import numpy as np
import sys

# =============================================================================
class BitmapAtlas:
    """ Group multiple small data regions into a larger texture.

        The algorithm is based on the article by Jukka Jylänki:
        "A Thousand Ways to Pack the Bin - A Practical Approach to
        Two-Dimensional Rectangle Bin Packing", February 27, 2010.
        More precisely, this is an implementation of
        the Skyline Bottom-Left algorithm based on C++ sources provided by
        Jukka Jylänki at: http://clb.demon.fi/files/RectangleBinPack/

        ::

            atlas = Atlas(512, 512, 3)
            region = atlas.get_region(20, 20)
            ...
            atlas.set_region(region, data)
    """
    def __init__(self, width=8192, height=8192):
        """ Initialize a new atlas of given size.

            Args:
                width (int): Width of the underlying texture
                height (int): Height of the underlying texture
        """
        self.width  = int(math.pow(2, int(math.log(width, 2) + 0.5)))
        self.height = int(math.pow(2, int(math.log(height, 2) + 0.5)))
        self.nodes  = [(0, 0, self.width),]
        self.data   = np.zeros((self.height, self.width, 4),
                               dtype=np.ubyte)
        self.atlasId  = 0
        self.used   = 0

    def getFreeSpot(self, width, height):
        """ Allocate a free region of given size and return it.

            Args:
                width (int): Width of region to allocate
                height (int): Height of region to allocate

            Returns:
                tuple: A newly allocated region as (x, y, width, height)
                or (-1, -1, 0, 0)
        """
        best_height = sys.maxsize
        best_index = -1
        best_width = sys.maxsize
        region = 0, 0, width, height

        for i in range(len(self.nodes)):
            y = self._fit(i, width, height)
            if y >= 0:
                node = self.nodes[i]
                if (y+height < best_height or
                    (y+height == best_height and node[2] < best_width)):
                    best_height = y+height
                    best_index = i
                    best_width = node[2]
                    region = node[0], y, width, height

        if best_index == -1:
            return -1,-1,0,0

        node = region[0], region[1]+height, width
        self.nodes.insert(best_index, node)

        i = best_index+1
        while i < len(self.nodes):
            node = self.nodes[i]
            prev_node = self.nodes[i-1]
            if node[0] < prev_node[0]+prev_node[2]:
                shrink = prev_node[0]+prev_node[2] - node[0]
                x,y,w = self.nodes[i]
                self.nodes[i] = x+shrink, y, w-shrink
                if self.nodes[i][2] <= 0:
                    del self.nodes[i]
                    i -= 1
                else:
                    break
            else:
                break
            i += 1

        self._merge()
        self.used += width*height
        return region

    def _fit(self, index, width, height):
        """ Test if region (width, height) fits into self.nodes[index],
            return -1 if it does not.

            Args:
                index (int): Index of the internal node to be tested
                width (int): Width or the region to be tested
                height (int): Height or the region to be tested

            Returns:
                int: offset or -1
        """
        node = self.nodes[index]
        x,y = node[0], node[1]
        width_left = width

        if x+width > self.width:
            return -1

        i = index
        while width_left > 0:
            node = self.nodes[i]
            y = max(y, node[1])
            if y+height > self.height:
                return -1
            width_left -= node[2]
            i += 1
        return y

    def _merge(self):
        """ Merge nodes
        """
        i = 0
        while i < len(self.nodes)-1:
            node = self.nodes[i]
            next_node = self.nodes[i+1]
            if node[1] == next_node[1]:
                self.nodes[i] = node[0], node[1], node[2]+next_node[2]
                del self.nodes[i+1]
            else:
                i += 1

# =============================================================================
class Glyph:
    """ Abstraction of TTF glyf Glyph table. It stores the glyph metric,
        extract vectorial Bezier curves, creates the vertices of the bounding
        box used by OpenGL to render the glyph.
    """
    FT_CURVE_TAG_CONIC, FT_CURVE_TAG_ON, FT_CURVE_TAG_CUBIC = range(3)

    def __init__(self, name):
        """ Create new instance.

            Args:
                name (str): glyph name
        """
        self.name = name

        # Hex glyf code if available in the charmap, unicode
        self.codePoint = None
        # Progressive index
        self.index = None
        self.description = None

        # Save bezier curves in a list of arrays, one for each contour
        self.contours = None

        # Bezier curves as list of structured arrays
        self.data = None

        self.dataTable = {}
        self.dataTable['glyphTypes'] = []
        self.bezierCount = []
        self.scale = 1.0

    def cp(self):
        """ Return the point code integer = ord(chr(charPoint)).

            Returns:
                int: point code integer
        """
        if self.codePoint is None:
            return self.name
        return self.codePoint

    def setMetric(self, glyphItem, glyphMetric):
        """ Save the glyph metric.

            .. |hmtx| replace::
                :class:`fontTools.ttLib.tables._h_m_t_x.table__h_m_t_x`

            Args:
                glyphItem (:class:`fontTools.ttLib.tables._g_l_y_f.Glyph`):
                    instance
                glyphMetric (|hmtx|): hmtx table
        """
        if glyphItem.numberOfContours != 0:
            # Define some attribute
            self.width = glyphItem.xMax - glyphItem.xMin
            self.height = glyphItem.yMax - glyphItem.yMin
            self.xMax = glyphItem.xMax
            self.yMax = glyphItem.yMax
            self.xMin = glyphItem.xMin
            self.yMin = glyphItem.yMin
        else:
            self.width = 0
            self.height = 0
            self.xMax = 0
            self.yMax = 0
            self.xMin = 0
            self.yMin = 0

        horiBearingY = self.yMax
        advanceX, horiBearingX = glyphMetric

        self.offset = (horiBearingX, horiBearingY-self.height)
        self.hAdvance = advanceX

        self.vAdvance = 0
        self.dataTable['height'] = self.height
        self.dataTable['width'] = self.width
        self.dataTable['horz_advance'] = self.hAdvance
        self.dataTable['vAdvance'] = self.vAdvance
        self.dataTable['index'] = self.index
        self.dataTable['offset'] = self.offset
        self.dataTable['xMax'] = self.xMax
        self.dataTable['xMin'] = self.xMin
        self.dataTable['yMax'] = self.yMax
        self.dataTable['yMin'] = self.yMin

    def hasContours(self):
        """ Return True if the glyph has contours.

            Returns:
                bool: True if the glyph has contours
        """
        return self.numberOfContours != 0

    def numContours(self):
        """ Return the number of countour lines.

            Returns:
                int: the number of countour lines
        """
        return len(self.contours)

    def buildVertices(self):
        """ Build the vertices structured array for the glyph
        """
        self.vertices = np.array(
                [([       0.0,         0.0],  [0, 0], [0,0,0,0], [0,0,0,0]),
                ([ self.width,         0.0],  [1, 0], [0,0,0,0], [0,0,0,0]),
                ([ self.width, self.height],  [1, 1], [0,0,0,0], [0,0,0,0]),
                ([        0.0, self.height],  [0, 1], [0,0,0,0], [0,0,0,0])],
                dtype = [('vtx', '<f4', (2,)), ('tex', '<f4', (2,)),
                    ('gp', np.uint16, (4,)), ('rgba', '<f4', (4,))])
        self.dataTable['vertices'] =  self.vertices


    def extractContours(self, outlineData):
        """ Extract the glyph outline contours.

            Args:
                outlineData (tuple): return value of Glyph.getCoordinates(glyf)

            Returns:
                list: list of strucutured arrays, one for each contour
        """
        if not self.contours:
            self.coordinates, self.endPtsOfContours, self.flags = outlineData
            self.numberOfContours = len(self.endPtsOfContours)

            contours_points = self.coordinates
            contours = self.endPtsOfContours
            count = self.numberOfContours
            flags = self.flags

            start = 0
            self.contours = []

            for end in contours:
                points = contours_points[start:end + 1]
                tags = flags[start:end + 1]

                ## Manage last point
                tag0 = tags[0]
                tag_last = tags[-1]

                if tag0 == self.FT_CURVE_TAG_CUBIC:
                    raise ValueError('A contour cannot start with'
                            ' a cubic control point')

                #NOTE: cosa vuol dire?
                # Start with a CONIC OFF
                if tag0 == self.FT_CURVE_TAG_CONIC:

                    # Start at last point if it is on the curve
                    if tag_last == self.FT_CURVE_TAG_ON:
                        points.insert(0, points[-1])
                        tags.insert(0, tags[-1])
                    else:
                        # If both first and last points are conic, start at
                        # their middle and record its position for closure
                        vp = ((points[0][0] + points[-1][0]) / 2,
                                (points[0][1] + points[-1][1]) / 2)
                        points.insert(0, vp)
                        tags.insert(0, 1)

                points.append(points[0])
                tags.append(tags[0])

                outlines = []

                curve = []
                for i, p in enumerate(points[:-1]):
                    tag = tags[i]
                    if tag == self.FT_CURVE_TAG_ON: # point on curve
                        curve.append(p)
                        if len(curve) >= 2: # closed curve
                            outlines.append(curve)
                            curve = [p]
                        if i == (len(points)-1):
                            curve.append(points[0])
                            outlines.append(curve)
                            break
                    elif tag == self.FT_CURVE_TAG_CONIC:
                        # Conic curve off point
                        if tags[i + 1] == self.FT_CURVE_TAG_ON:
                            curve.append(p)
                        # Interpolate points
                        if tags[i + 1] == self.FT_CURVE_TAG_CONIC:
                            vp = (0.5 * (p[0] + points[i + 1][0]),
                                    0.5 * (p[1] + points[i + 1][1]))
                            curve.append(p)
                            curve.append(vp)
                            outlines.append(curve)
                            curve = [vp]
                    elif tag == self.FT_CURVE_TAG_CUBIC:
                        curve.append(p)
                if tags[i + 1] == self.FT_CURVE_TAG_ON:
                    curve.append(points[i + 1])
                    outlines.append(curve)
                self.contours.append(outlines)
                start = end + 1
        return self.contours

    def contours2array(self, x0=0, y0=0, xs=1.0, ys=1.0):
        """ Convert list of outlines to a structured array.

            Args:
                x0 (float): x offset
                y0 (float): y offset
                xs (float): x scale
                ys (float): y scale

        """
        self.data, self.bezierCount, self.scale = self.buildOutlineArray(
                self.contours, x0, y0, xs, ys)
        self.dataTable['scale'] = self.scale

    @staticmethod
    def buildOutlineArray(contours, x0=0, y0=0, xs=1.0, ys=1.0):
        """ Convert list of outlines to a structured array

            Args:
                contours(list): list of outline contour
                x0 (float): x offset
                y0 (float): y offset
                xs (float): x scale
                ys (float): y scale

            Returns:
                tuple: content:
                       - list of bezier curves,
                       - number of bezier curves,
                       - scale factor
        """
        bezier_dtype = np.dtype([('p1', 'f4', 2), ('p2', 'f4', 2),
                ('p3', 'f4', 2), ('p4', 'f4', 2)])
        data = []
        bezierCount = []

        shift = np.asarray([x0, y0], np.float32)
        scale = np.asarray([xs, ys], np.float32)
        if contours:
            for contour in contours:
                bezierCount.append(len(contour))
                base = np.zeros(len(contour), dtype=bezier_dtype)
                base.fill(np.nan)
                for i, item in enumerate(contour):
                    base[i]['p1'] = item[0]
                    base[i]['p4'] = item[-1]
                    if len(item) >= 3:
                        base[i]['p2'] = item[1]
                    if len(item) == 4:
                        base[i]['p3'] = item[2]
                    if len(item) == 2:
                        base[i]['p2'] = (base[i]['p1'] + base[i]['p4']) / 2
                base['p1'] -= shift
                base['p2'] -= shift
                base['p3'] -= shift
                base['p4'] -= shift
                base['p1'] /= scale
                base['p2'] /= scale
                base['p3'] /= scale
                base['p4'] /= scale
                data.append(base)
        return data, bezierCount, scale
