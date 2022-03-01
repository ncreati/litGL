import sys
from abc import ABCMeta, abstractmethod
import numpy as np
import glfw
glfw.ERROR_REPORTING = True
import OpenGL.GL as gl
import glm
from ..shader import Shader
from ..mesh import Mesh

actionMap = {glfw.PRESS: 'press',
             glfw.RELEASE: 'release',
             glfw.REPEAT: 'repeat'}

#==============================================================================
#:
VERTEX = """
# version 330

layout(location=0) in vec2 position;

uniform mat4 T_MVP;

void main()
{
    gl_Position = T_MVP * vec4(position.x, position.y, 0.0f, 1.0f);
}
"""

#:
FRAGMENT = """
# version 330

uniform vec4 color;
out vec4 fragColor;

void main()
{
    fragColor = color;
}
"""

#==============================================================================
def get_monitor_dpi():
    """ Return monitor resolution in dot per inch.

        Returns:
            float: monitor dot per inch
    """
    monitor = glfw.get_primary_monitor()[0]
    # Size in mm
    video_physical_size = glm.vec2(glfw.get_monitor_physical_size(monitor))
    #Monitor info
    video_resolution = glm.vec2(glfw.get_video_mode(monitor).size)
    # Calculate screen DPI
    inch_to_mm = 0.393701/10.0
    dpi = (video_resolution/(video_physical_size*inch_to_mm))
    return sum(dpi) / 2.0

#------------------------------------------------------------------------------
class Cross:
    """ Draw a cross centered on the OpenGL window.
    """
    def __init__(self, color=[1.0, 0.0, 1.0, 1.0]):
        self.shader = Shader.fromString(vertex=VERTEX, fragment=FRAGMENT)
        self.color = color
        self.vertices = np.zeros(4, dtype=[('vxt', np.float32, 2)])
        self.vertices['vxt'][0] = [0.0, -1.0]
        self.vertices['vxt'][1] = [0.0, 1.0]
        self.vertices['vxt'][2] = [-1.0, 0.0]
        self.vertices['vxt'][3] = [1.0, 0.0]
        indices = np.array([0, 1, 2, 3], np.uint32)
        self.mesh = Mesh(self.vertices, indices)

    def render(self, mvp=glm.mat4(1.0)):
        self.shader.bind()
        self.shader.setUniform('T_MVP', mvp)
        self.shader.setUniform('color', self.color)
        self.mesh.draw(gl.GL_LINES)
        self.shader.unbind()

#------------------------------------------------------------------------------
class Rectangle:
    """ Draw a rectangle on the OpenGL window.
    """
    def __init__(self, window, color=[0, 1, 1, 1]):
        self.shader = Shader.fromString(vertex=VERTEX, fragment=FRAGMENT)
        self.color = color
        self.window = window
        self.vertices = np.zeros(1, dtype=[('vxt', np.float32, 2)])
        self.vertices['vxt'][0] = [0.0, 0.0]
        indices = np.array([0], np.uint32)
        self.mesh = Mesh(self.vertices, indices)

    def win2openGL(self, x, y):
        xg = x / self.window[0] * 2 - 1
        yg = - (y / self.window[1] * 2 - 1)
        return xg, yg

    def setVertices(self, x0, y0, width, height):
        vertices = np.zeros(4, dtype=[('vxt', np.float32, 2)])
        vertices['vxt'][0] = self.win2openGL(x0, y0)
        vertices['vxt'][1] = self.win2openGL(x0, y0 + height)
        vertices['vxt'][2] = self.win2openGL(x0 + width, y0 + height)
        vertices['vxt'][3] = self.win2openGL(x0 + width, y0)
        indices = np.array([0, 1, 1, 2, 2, 3, 3, 0], np.uint32)
        self.mesh.rebuild(vertices, indices)

    def render(self, mvp=glm.mat4(1.0)):
        self.shader.bind()
        self.shader.setUniform('T_MVP', mvp)
        self.shader.setUniform('color', self.color)
        self.mesh.draw(gl.GL_LINES)
        self.shader.unbind()

#------------------------------------------------------------------------------
def createWindow(width=100, height=100, title="", visible=False):
    """ Create on OpenGL window and return it.

        Args:
            width (int): window width
            height (int): window height
            title (str): window title
            visible (bool): make the window visible

        Returns:
            :class:`glfw.LP__GLFWwindow`: window instance
    """
    glfw.window_hint(glfw.VISIBLE, visible)
    if sys.platform == 'darwin':
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    window = glfw.create_window(width, height, title, None, None)
    glfw.make_context_current(window)
    return window

#------------------------------------------------------------------------------
def glInfo():
    """ Return OpenGL information dict.

        .. WARNING:: OpenGL context MUST be initialized before calling
            this function!!!

        Returns:
            OpenGL information dict
    """
    try:
        glcontext = glfw.get_current_context()
    except glfw.GLFWError:
        try:
            glfw.init()
            createWindow()
        except:
            return None
        else:
            wasInitialized = False
    else:
        wasInitialized = True

    major = gl.glGetIntegerv(gl.GL_MAJOR_VERSION)
    minor = gl.glGetIntegerv(gl.GL_MINOR_VERSION)
    version = gl.glGetString(gl.GL_VERSION)
    vendor = gl.glGetString(gl.GL_VENDOR)
    renderer = gl.glGetString(gl.GL_RENDERER)
    glsl = gl.glGetString(gl.GL_SHADING_LANGUAGE_VERSION)
    glversion = float("%d.%d" % (major, minor))
    retval = {
            'glversion': glversion,
            'version': version,
            'vendor': vendor,
            'renderer': renderer,
            'glsl': glsl,
            }
    if glversion >= 4.3:
        count = np.zeros(3, dtype=np.int32)
        size = np.zeros(3, dtype=np.int32)
        count[0] = gl.glGetIntegeri_v(gl.GL_MAX_COMPUTE_WORK_GROUP_COUNT, 0)[0]
        count[1] = gl.glGetIntegeri_v(gl.GL_MAX_COMPUTE_WORK_GROUP_COUNT, 1)[0]
        count[2] = gl.glGetIntegeri_v(gl.GL_MAX_COMPUTE_WORK_GROUP_COUNT, 2)[0]
        size[0] = gl.glGetIntegeri_v(gl.GL_MAX_COMPUTE_WORK_GROUP_SIZE, 0)[0]
        size[1] = gl.glGetIntegeri_v(gl.GL_MAX_COMPUTE_WORK_GROUP_SIZE, 1)[0]
        size[2] = gl.glGetIntegeri_v(gl.GL_MAX_COMPUTE_WORK_GROUP_SIZE, 2)[0]
        retval['maxComputeWorkGroupCount'] = count
        retval['maxComputeWorkGroupSize'] = size

    if not wasInitialized:
        glfw.terminate()
    return retval

#------------------------------------------------------------------------------
def errorCallback(error, description):
    print('Error %s, %s' % (error, description))

#==============================================================================
class glfwApp(metaclass=ABCMeta):

    KEY_G = glfw.KEY_G
    KEY_N = glfw.KEY_N
    KEY_F = glfw.KEY_F
    PRESS = glfw.PRESS
    RELEASE = glfw.RELEASE

    def __init__(self, title='', width=800, height=600, resizable=True,
            bgColor=(0, 0, 0, 0)):
        """ Create a new glfwApp instance.

            Args:
                title (string): window title
                width (integer): window width in pixels
                height (integer): window height in pixels
                resizable (bool): if True the window can be resized
                bgColor (tuple): background color (0..1)
        """
        self.width = width
        self.height = height
        self._title = title
        self.bg_color = bgColor

        glfw.set_error_callback(errorCallback)

        if not glfw.init():
            raise SystemExit("Error initializing GLFW")
        if resizable:
            glfw.window_hint(glfw.RESIZABLE, gl.GL_TRUE)
        else:
            glfw.window_hint(glfw.RESIZABLE, gl.GL_FALSE)
        # Create the window
        self._window = createWindow(self.width, self.height,
                self._title, visible=True)
        glfw.set_framebuffer_size_callback(self._window, self.onResize)

        if not self._window:
            glfw.terminate()
            raise SystemExit

        glfw.make_context_current(self._window)
        glfw.set_key_callback(self._window, self.onKeyboard)
        self.fwidth, self.fheight = glfw.get_framebuffer_size(self._window)
        gl.glViewport(0, 0, self.fwidth, self.fheight)

    @abstractmethod
    def onResize(self, window, width, height):
        """ This method must be implemened. It is called automatically when
            the window gets resized.

            Args:
                window (class:`glfw.LP__GLFWwindow` instance): window
                width (int): window width in pixels
                height (int): window height in pixels
        """
        pass

    def onKeyboard(self, window, key, scancode, action, mode):
        """ Process keybord input. This method is called automatically when
            the user interacts with the keyboard.

            Args:
                window (:class:`glfw.LP__GLFWwindow` instance): window
                key (integer): the key that was pressed
                scancode (integer):
                action (integer): PRESS, RELEASE, REPEAT
                mode (integer): modifier
        """
        if key in (glfw.KEY_ESCAPE, glfw.KEY_Q):
            glfw.set_window_should_close(self._window, 1)
        glfw.poll_events()

    def window(self):
        """ Return the window instance

            Returns:
                the window instance
        """
        return self._window

    def title(self):
        """ Return window title

            Returns:
                the window title
        """
        return self._title

    def setTitle(self, title):
        """ Set window title

            Args:
                title (string): the new window title
        """
        glfw.set_window_title(self._window, title)

    def run(self, close=True):
        """ Start the application main loop
        """
        while not glfw.window_should_close(self._window):
            gl.glClearColor(*self.bg_color)
            glfw.poll_events()
            self.renderScene()
            glfw.swap_buffers(self._window)
        if close:
            self.close()

    def close(self):
        """ Destroy the window and terminate glfw
        """
        glfw.destroy_window(self._window)
        glfw.terminate()

    def renderScene(self):
        """ Render the scene. This method is called automatically in the run
            loop
        """
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

# =============================================================================
if __name__ == "__main__":
    app = glfwApp('glfwApp')
    app.run()

