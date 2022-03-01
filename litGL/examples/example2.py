import numpy as np
import OpenGL.GL as gl
import glm

# Local imports
from litGL.label import Label
from litGL.dlabel import Dlabel
try:
    from litGL.examples.glfwBackend import (glfw, glfwApp, glInfo,
            get_monitor_dpi, Cross)
except ImportError as e:
    raise SystemExit("%s:\nAppliaction needs extra packages to run.\n"
            "Please install glfw." % e)

red = [1, 0, 0, 1]
green = [0, 1, 0, 1]
blue = [0, 0, 1, 1]
white = [1, 1, 1, 1]
yellow = [1, 1, 0, 1]
cyan = [0, 1, 1, 1]
magenta = [1, 0, 1, 1]
#===============================================================================
class GLApp(glfwApp):
    def __init__(self, title, width, height):
        super(GLApp, self).__init__(title, width, height)

        glversion = glInfo()['glversion']
        self.cross = Cross()
        self.text = []

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

    def addText(self):
        text1 = Label(text="Static Label", color=(0, 1, 1, 1), size=24,
                dpi=get_monitor_dpi())
        text1.setPos(self.width // 2, self.height // 2 - 80, 'cc')
        self.text.append(text1)
        text2 = Dlabel(text="Dynamic Label", size=24)
        text2.setPos(self.width // 2, self.height // 2, 'lc')
        colors = [
                red, green, blue, yellow,
                red, green, blue, yellow,
                red, green, blue, yellow,
                red,
                ]
        text2.setColors(colors, indexes=[
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        self.text.append(text2)
        text3 = Dlabel(text="Another Dynamic Label", size=24)
        text3.setPos(self.width // 2, self.height // 2 + 80, 'lc')
        colors = [
                red, green, blue, yellow, cyan, magenta, white,
                red, green, blue, yellow, cyan,
                ]
        text3.setColors(colors, indexes=[
            0, 1, 2, 3, 4, 5, 6, 16, 17, 18, 19, 20])
        self.text.append(text3)
        text4 = Dlabel(text="New Dynamic Label", size=24)
        text4.setPos(self.width // 2, self.height // 2 + 140, 'lc')
        colors = [
            red, green, blue, yellow,
            ]
        text4.setColors(colors, indexes=[0, 1, 2, 3])
        self.text.append(text4)

#===============================================================================
if __name__ == '__main__':
    import signal
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    app = GLApp('Example 2', 800, 600)
    app.addText()
    app.run()

