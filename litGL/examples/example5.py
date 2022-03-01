from pathlib import Path
import numpy as np
import OpenGL.GL as gl
import glm
import logging

# Local imports
from litGL.label import Label
from litGL.fontDistiller import GlyphTypes
try:
    from litGL.examples.glfwBackend import (glfw, glfwApp, glInfo,
            get_monitor_dpi, Cross)
except ImportError as e:
    raise SystemExit("%s:\nAppliaction needs extra packages to run.\n"
            "Please install glfw." % e)

#===============================================================================
class GLApp(glfwApp):
    def __init__(self, title, width, height):
        super(GLApp, self).__init__(title, width, height)

        glversion = glInfo()['glversion']
        self.cross = Cross()
        self.text = []
        # Set screen resolution
        Label.DPI = get_monitor_dpi()

    def renderScene(self):
        super(GLApp, self).renderScene()
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        gl.glClearColor(1.0, 1.0, 1.0, 1.0)
        proj=glm.ortho(0, self.width, self.height, 0)
        self.cross.render()
        for text in self.text:
            text.render(proj)

    def onKeyboard(self, window, key, scancode, action, mode):
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            glfw.set_window_should_close(window, True)
        super(GLApp, self).onKeyboard(window, key, scancode, action, mode)

    def onResize(self, window, width, height):
        gl.glViewport(0, 0, width, height)

    def addText(self, *args, **kargs):
        font_file = kargs.get('font_file', None)
        pos = kargs.pop('pos')
        anchor = kargs.pop('anchor')
        if not font_file is None:
            kargs['font_file'] = Path(__file__).parent.joinpath(font_file)
        text = Label(*args, **kargs)
        text.setPos(*pos, anchor)
        self.text.append(text)


#===============================================================================
if __name__ == '__main__':
    import sys
    import argparse
    import signal
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    #logging.basicConfig()

    app = GLApp('Example 5', 800, 600)
    size = 44

    font = "fonts/SpiderWritten.ttf"
    app.addText('SpiderWritten', pos=(0, 0), anchor='ul', font_file=font,
            size=size)

    font = "fonts/PWScratchy.ttf"
    color = (0, 1, 0, 1)
    app.addText('PWScratchy', pos=(0, 50), anchor='ul', font_file=font,
            color=color, size=size)

    pos = (0, 200)
    font = "fonts/RocherColorGX.ttf"
    color = (0, 1, 1, 1)
    text = 'Rocher BASE' + chr(8505) + "☀☑"
    app.addText(text, pos=pos, anchor='ll', font_file=font,
            color=color, size=size)
    color = (0, 1, 1, 0.5)
    text = 'Rocher LAYERED' + chr(8505) + "☀☑"
    app.addText(text, pos=pos, anchor='ul', font_file=font,
            color=color, glyphs=GlyphTypes.LAYER_COLOR, size=size)

    #pos = (0, 350)
    #font = "fonts/seguiemj.ttf"
    #color = (0, 1, 1, 1)
    #text = "seguiemj BASE" + chr(8505) + "☀☑"
    #app.addText(text, pos=pos, anchor='ll', font_file=font,
            #color=color, size=size)
    #color = (0, 1, 1, 0.5)
    #text = "seguiemj LAYERED" + chr(8505) + "☀☑"
    #app.addText(text, pos=pos, anchor='ul', font_file=font,
            #color=color, glyphs=GlyphTypes.LAYER_COLOR, size=size)

    font = "fonts/NotoColorEmoji.ttf"
    pos = (0, 450)
    color = (0, 1, 0, 1)
    app.addText('Noto😂☀☑\n☑☀', pos=pos, anchor='ul', font_file=font,
            color=color, glyphs=GlyphTypes.CBDT_COLOR, size=size)

    color = (0, 1, 0, 0.1)
    pos = (app.width // 2, app.height)
    app.addText('Default font\n[Second line]', pos=pos, anchor='ll',
            color=color, size=size)

    app.run()

