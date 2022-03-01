from pathlib import Path
import argparse
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
        self.rect = Rectangle((self.width, self.height))
        self.ialign = 6
        self.iangle = 0
        self.anchors = "cl cc cr ul uc ur ll lc lr".split()

    def renderScene(self):
        super(GLApp, self).renderScene()
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        gl.glClearColor(1.0, 1.0, 1.0, 1.0)
        proj=glm.ortho(0, self.width, self.height, 0)
        self.cross.render()
        self.text.render(proj)
        self.rect.render()

    def onKeyboard(self, window, key, scancode, action, mode):
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            glfw.set_window_should_close(window, True)
        elif ((key == glfw.KEY_N or key == glfw.KEY_SPACE)
                and action == glfw.PRESS):
            # Cycle alignments
            self.ialign += 1
            self.ialign %= len(self.anchors)
            print("Set alignment to --->", self.anchors[self.ialign])
            self.text.setPos(*self.pos, anchor=self.anchors[self.ialign])
            self.rect.setVertices(*self.text.boundingBox())
        elif key == glfw.KEY_UP and action == glfw.PRESS:
            # Cycle rotations
            self.iangle += 20
            if self.iangle > 180:
                self.iangle = -160
            self.text.setRotation(self.iangle)
            self.rect.setVertices(*self.text.boundingBox())
        elif key == glfw.KEY_DOWN and action == glfw.PRESS:
            # Cycle rotations
            self.iangle -= 20
            if self.iangle < -180:
                self.iangle = 160
            self.text.setRotation(self.iangle)
            self.rect.setVertices(*self.text.boundingBox())
        super(GLApp, self).onKeyboard(window, key, scancode, action, mode)

    def onResize(self, window, width, height):
        gl.glViewport(0, 0, width, height)

    def addText(self, value):
        self.pos = (self.width // 2, self.height // 2)
        self.text = Label(text=value, size=20, pos=self.pos, angle=0,
                dpi=get_monitor_dpi())
        self.text.setPos(*self.pos, 'll')
        rect = self.text.boundingBox()
        self.rect.setVertices(*rect)


#===============================================================================
if __name__ == '__main__':
    import signal
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    app = GLApp('Example 4', 800, 600)
    app.addText('Press n to cycle alignment,\nq to quit')
    app.run()

