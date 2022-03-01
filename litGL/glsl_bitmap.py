""" GLSL shader code to render bitmap glyphs.\
        :download:`[source] <../../../litGL/glsl_bitmap.py>`

    Author:
        2020-2021 Nicola Creati

    Copyright:
        2020-2021 Nicola Creati <ncreati@inogs.it>

    License:
        MIT/X11 License (see
        :download:`license.txt <../../../license.txt>`)
"""
#:
VERTEX_SHADER = """
#version 330

layout(location=0) in vec2 position;
layout(location=1) in vec2 tex;
layout(location=2) in vec4 gp;

uniform mat4 T_MVP;

out vs_output
{
 vec2 texCoord;
 flat uvec4 glyphParam;
} vs_out;

void main()
{
   gl_Position = T_MVP * vec4(position.x, position.y, 0.0f, 1.0f);
    vs_out.texCoord = tex;
    vs_out.glyphParam = uvec4(gp);
}

"""
#:
FRAGMENT_SHADER = """
#version 330

in vs_output
{
    vec2 texCoord;
    flat uvec4 glyphParam;
} fs_in;

out vec4 fragColor;

uniform sampler2D u_colorsTex;
uniform vec4 u_color;

void main()
{
    vec4 sampled = texture(u_colorsTex, fs_in.texCoord);
    fragColor = vec4(sampled.xyz, sampled.w * u_color.w);
}
"""
