import os
import sys
import time
from pathlib import Path
import argparse
import numpy as np
import OpenGL.GL as gl
import glm

# Local imports
from litGL.label import Label
try:
    from litGL.examples.glfwBackend import (glfw, glfwApp, glInfo,
            get_monitor_dpi, Cross)
except ImportError as e:
    raise SystemExit("%s:\nAppliaction needs extra packages to run.\n"
            "Please install glfw." % e)

# =============================================================================
class GLApp(glfwApp):
    def __init__(self, title, width, height):
        super(GLApp, self).__init__(title, width, height, bgColor=(1, 1, 1, 1))

        self.glversion = glInfo()['glversion']
        self.cross = Cross()
        Label.DPI = get_monitor_dpi()

    def renderScene(self):
        super(GLApp, self).renderScene()
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        gl.glClearColor(1.0, 1.0, 1.0, 1.0)
        proj = glm.ortho(0.0, self.width, self.height, 0)
        self.cross.render()
        self.text.render(proj)

    def onKeyboard(self, window, key, scancode, action, mode):
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            glfw.set_window_should_close(window, True)
        super(GLApp, self).onKeyboard(window, key, scancode, action, mode)

    def onResize(self, window, width, height):
        gl.glViewport(0, 0, width, height)

    def addText(self, text):
        self.text = Label(text=text, pos=(400, 300), size=22,
                filterControl=False,
                color=(0, 0, 0, 1))

# =============================================================================
if __name__ == '__main__':
    import sys
    import argparse
    import signal
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    app = GLApp('Example 1', 800, 600)
    for i in range(5):
        app.addText("Hello world!\n - 2nd line\n - 3rd line\nThat's all folks!")
    app.run()

