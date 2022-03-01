import numpy as np
import OpenGL.GL as gl
import glm

# Local imports
from litGL.label import Label
try:
    from litGL.examples.glfwBackend import (glfw, glfwApp, glInfo,
            get_monitor_dpi, Cross, Rectangle)
except ImportError as e:
    raise SystemExit("%s:\nAppliaction needs extra packages to run.\n"
            "Please install glfw." % e)

#===============================================================================
class GLApp(glfwApp):
    def __init__(self, title, width, height):
        super(GLApp, self).__init__(title, width, height)

        glversion = glInfo()['glversion']
        self.cross = Cross()
        self.rectangles = []
        self.labels = []

    def renderScene(self):
        super(GLApp, self).renderScene()
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        gl.glClearColor(1.0, 1.0, 1.0, 1.0)
        proj=glm.ortho(0, self.width, self.height, 0)
        self.cross.render()
        for label in self.labels:
            label.render(proj)
        for rect in self.rectangles:
            rect.render()

    def onKeyboard(self, window, key, scancode, action, mode):
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            glfw.set_window_should_close(window, True)
        super(GLApp, self).onKeyboard(window, key, scancode, action, mode)

    def onResize(self, window, width, height):
        gl.glViewport(0, 0, width, height)

    def addRect(self, x, y, width, height):
        self.rectangles.append(Rectangle((self.width, self.height)))
        self.rectangles[-1].setVertices(x, y, width, height)

    def addText(self, *args, **kargs):
        if not 'size' in kargs:
            kargs['size'] = 24
        label = Label(*args, dpi=get_monitor_dpi(), **kargs)
        self.labels.append(label)
        x, y, w, h = label.boundingBox()
        self.addRect(x, y, w, h)

#===============================================================================
if __name__ == '__main__':
    from pathlib import Path
    import signal

    signal.signal(signal.SIGINT, signal.SIG_DFL)
    spider_font = Path(__file__).parent.joinpath("fonts/SpiderWritten.ttf")

    app = GLApp('Example 3', 1024, 768)
    p0 = (app.width // 2, app.height // 2)
    app.addText('Label 1  ', pos=p0)

    l1_width = app.labels[0].pixwidth()
    p1 = (p0[0] + l1_width, p0[1])
    app.addText('         ', pos=p1)

    l2_width = app.labels[-1].pixwidth()
    p2 = (p0[0] + l1_width + l2_width, p0[1])
    app.addText('Label 3  ', pos=p2)

    app.addText('Label4  ', pos=p0, anchor='lr')

    app.addText('SPIDER LABEL 123', pos=p0, anchor='uc', size=72,
            font_file=spider_font)

    p3 = (p0[0], p0[1] + 200)
    app.addText('Multi\nline\nlabel', size=28, pos=p3, anchor='ll')
    newpos = app.labels[-1].nextCharLowerLeft()

    app.addText('+another label', size=28, pos=newpos, anchor='ll')

    app.run()

