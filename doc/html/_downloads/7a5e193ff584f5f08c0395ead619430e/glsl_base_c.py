""" GLSL code to render colored glyphs:\
        :download:`[source] <../../../litGL/glsl_base_c.py>`

GLSL shader code based on the following public sources:
 - Journal paper: Eric Lengyel, GPU-Centered Font Rendering Directly from Glyph
   Outlines, Journal of Computer Graphics Techniques (JCGT), vol. 6, no. 2,
   31-47, 2017 (http://jcgt.org/published/0006/02/02/)
 - slide presentation: http://terathon.com/i3d2018_lengyel.pdf
 - slide presentation http://terathon.com/font_rendering_sota_lengyel.pdf

Author:
    2020-2021 Nicola Creati

Copyright:
    2020-2021 Nicola Creati <ncreati@inogs.it>

License:
    MIT/X11 License (see
    :download:`license.txt <../../../license.txt>`)
"""
#-------------------------------------------------------------------------------
#:
VERTEX_SHADER = """
#version 330

layout(location=0) in vec2 position;
layout(location=1) in vec2 tex;
layout(location=2) in vec4 gp;
layout(location=3) in vec4 rgba;

uniform mat4 T_MVP;

out vs_output
{
 vec2 texCoord;
 flat uvec4 glyphParam;
 vec4 color;
} vs_out;

void main()
{
   gl_Position = T_MVP * vec4(position.x, position.y, 0.0f, 1.0f);
    vs_out.texCoord = tex;
    vs_out.glyphParam = uvec4(gp);
    vs_out.color = rgba;
}
"""
#-------------------------------------------------------------------------------
#:
FRAGMENT_SHADER = """
#version 330

precision highp float;
precision highp int;
precision highp sampler2D;
precision highp usampler2D;

out vec4 fragColor;

uniform sampler2DRect u_curvesTex;
uniform usampler2DRect u_bandsTex;

in vs_output
{
    vec2 texCoord;
    flat uvec4 glyphParam;
    vec4 color;
} fs_in;

const float epsilon = 0.0001;

vec2 SolvePoly(vec4 p12, vec2 p3)
{
    // At least one root makes a contribution, so solve for the
    // values of t where the curve crosses y = 0. The quadratic
    // polynomial in t is given by
    //
    //     a t^2 - 2b t + c,
    //
    // where a = p1.y - 2 p2.y + p3.y, b = p1.y - p2.y, and c = p1.y.
    // The discriminant b^2 - ac is clamped to zero, and imaginary
    // roots are treated as a double root at the global minimum
    // where t = b / a.

    // Calculate coefficients
    vec2 a = p12.xy - p12.zw * 2.0 + p3;
    vec2 b = p12.xy - p12.zw;
    float ra = 1.0 / a.y;
    float rb = 0.5 / b.y;

    // Clamp discriminant to zero.
    float d = sqrt(max(b.y * b.y - a.y * p12.y, 0.0));
    float t1 = (b.y - d) * ra;
    float t2 = (b.y + d) * ra;

    // Handle linear case where |a| ≈ 0.
    if (abs(a.y) < epsilon) t1 = t2 = p12.y * rb;

    // Return x coordinates at t1 and t2m where curve(t)=0
    return (vec2((a.x * t1 - b.x * 2.0) * t1 + p12.x, (a.x * t2 - b.x * 2.0) * t2 + p12.x));
}

uint CalcRootCode(float y1, float y2, float y3)
{
    uint i1 = floatBitsToUint(y1) >> 31U;
    uint i2 = floatBitsToUint(y2) >> 30U;
    uint i3 = floatBitsToUint(y3) >> 29U;

    uint shift = (i2 & 2U) | (i1 & ~2U);
    shift = (i3 & 4U) | (shift & ~4U);

    return ((0x2E74U >> shift) & 0x0101U);
}

bool TestCurve(uint code)
{
    return (code != 0U);
}

bool TestRoot1(uint code)
{
    return ((code & 1U) != 0U);
}

bool TestRoot2(uint code)
{
    return (code > 1U);
}

void main()
{
    float xcov = 0.0;
    float coverage;
    vec2 emsPerPixel = fwidth(fs_in.texCoord);

    // Rendered glyph pixel size
    vec2 pixelsPerEm = vec2(1.0 / emsPerPixel);

    // Glyph index in the bands texture (x, y)
    uvec2 glyphLoc = uvec2(fs_in.glyphParam.xy);

    // Size of horizontal and vertical bands
    vec2 nBands = fs_in.glyphParam.zw;

    // Band Index. x=vertical band; y=horizontal band
    ivec2 bandIndex = ivec2(clamp(ivec2(fs_in.texCoord * nBands), ivec2(0, 0), ivec2(nBands-1)));

    // Recover band data for the current pixel reading the texture header, number of curves in
    // the band (-x-)and offset to indices of curves in the curves texture (-y-)
    uvec2 hBandData = uvec2(texelFetch(u_bandsTex, ivec2(glyphLoc.x+uint(bandIndex.y), glyphLoc.y)).xy);
    // Curve indices band offset
    uvec2 hbandLoc = uvec2(hBandData.y, 0U);

    // Wrap x to the next line if the offset exceed the texture width (4096)
    hbandLoc.y += hbandLoc.x >> 12U;
    hbandLoc.x &= 0x0FFFU;
    // Loop over all curves in the horizontal band
    for (uint curve = 0U; curve < hBandData.x; curve++)
    {
        ivec2 shift = ivec2(hbandLoc.x + curve, hbandLoc.y);

        // Index of curve in the curve texture
        ivec2 curveLoc = ivec2(texelFetch(u_bandsTex, shift).xy);

        // Retrieve p1 and p2 control points of the current curve
        vec4 p12 = texelFetch(u_curvesTex, curveLoc) - vec4(fs_in.texCoord, fs_in.texCoord);
        // Retrieve p3 control point of the current curve
        vec2 p3 = texelFetch(u_curvesTex, ivec2(curveLoc.x+1, curveLoc.y)).xy - fs_in.texCoord;
        // If the largest x coordinate among all three control points falls
        // left of the current pixel, then there are no more curves in the
        // horizontal band that can influence the result, so exit the loop.
        // (The curves are sorted in descending order by max x coordinate.)
        if (max(max(p12.x, p12.z), p3.x) * pixelsPerEm.x < -0.5) break;

        // Generate the root contribution code based on the signs of the
        // y coordinates of the three control points.
        uint code = CalcRootCode(p12.y, p12.w, p3.y);

        if (TestCurve(code))
        {
            // solve the quadratic polynomial of the curve
            vec2 r = SolvePoly(p12, p3) * pixelsPerEm.x;

            // Bits in code tell which roots make a contribution.
            // clamp data at pixel center to calculate fractional coverage
            // calculate weigth coverage factor as square of the distace
            if (TestRoot1(code))
            {
                xcov += clamp(r.x + 0.5, 0.0, 1.0);
            }

            if (TestRoot2(code))
            {
                xcov -= clamp(r.y + 0.5, 0.0, 1.0);
            }
        }
    }
    // Vertical ray
    float ycov = 0.0;

    // Recover band data for the current pixel reading the texture header, number of curves in
    // the band (-x-)and offset to indices of curves in the curves texture (-y-)
    uvec2 vBandData = uvec2(texelFetch(u_bandsTex, ivec2(glyphLoc.x + nBands.y + bandIndex.x, glyphLoc.y)).xy);
    // Curve indices band offset
    uvec2 vbandLoc = uvec2(vBandData.y, 0U);

    // Wrap x to the next line if the offset exceed the texture width (4096)
    vbandLoc.y += vbandLoc.x >> 12U;
    vbandLoc.x &= 0x0FFFU;
    // Loop over all curves in the vertical band
    for (uint curve = 0U; curve <  vBandData.x; curve++)
    {
        ivec2 shift = ivec2(vbandLoc.x + curve, vbandLoc.y);
        // Index of curve in the curve texture
        ivec2 curveLoc = ivec2(texelFetch(u_bandsTex, shift).xy);
        // Retrieve p1 and p2 control points of the current curve
        vec4 p12 = texelFetch(u_curvesTex, curveLoc) - vec4(fs_in.texCoord, fs_in.texCoord);
        // Retrieve p3 control point of the current curve
        vec2 p3 = texelFetch(u_curvesTex, ivec2(curveLoc.x+1, curveLoc.y)).xy - fs_in.texCoord;

        // If the largest x coordinate among all three control points falls
        // left of the current pixel, then there are no more curves in the
        // horizontal band that can influence the result, so exit the loop.
        // (The curves are sorted in descending order by max x coordinate.)
        if (max(max(p12.y, p12.w), p3.y) * pixelsPerEm.y < -0.5) break;

        // Generate the root contribution code based on the signs of the
        // x coordinates of the three control points.
        uint code = CalcRootCode(p12.x, p12.z, p3.x);

        if (TestCurve(code))
        {
            vec2 r = SolvePoly(p12.yxwz, p3.yx) * pixelsPerEm.y;

            if (TestRoot1(code))
            {
                ycov -= clamp(r.x + 0.5, 0.0, 1.0);
            }

            if (TestRoot2(code))
            {
                ycov += clamp(r.y + 0.5, 0.0, 1.0);
            }
        }
    }

    float a = abs(xcov+ycov);
    float b =  (1.0-(abs(xcov))+(1.0-abs(ycov)));
    coverage = sqrt(clamp(sqrt((a)/(a+b)), 0.0, 1.0));
    float alpha = coverage * fs_in.color.w;
    fragColor = vec4(fs_in.color.xyz * alpha, alpha);

}
"""
