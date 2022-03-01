from pathlib import Path
import numpy as np
import OpenGL.GL as gl
import glm
import logging

# Local imports
from litGL.label import Label
try:
    from litGL.examples.glfwBackend import (glfw, glfwApp, glInfo,
            get_monitor_dpi, Cross)
except ImportError as e:
    raise SystemExit("%s:\nAppliaction needs extra packages to run.\n"
            "Please install glfw." % e)
from litGL.fontDistiller import FontDistiller

#===============================================================================
class GLApp(glfwApp):
    def __init__(self, fontlist, width, height):
        super(GLApp, self).__init__("", width, height)

        glversion = glInfo()['glversion']
        self.cross = Cross()
        self.text = ""
        # Set screen resolution
        Label.DPI = get_monitor_dpi()
        self.fontlist = fontlist
        self.ifont = 0
        self.itype = 0
        self.makeLabel()

    def makeLabel(self):
        font_file = self.fontlist[self.ifont]
        print(font_file)
        self.glyphTypes = Font.getGlyphTypes(font_file)
        gtype =  self.glyphTypes[self.itype]
        title = (f"Font {font_file.name} ({self.ifont} of"
                f" {len(self.fontlist)}): types {self.glyphTypes},"
                f" showing type {gtype}")
        self.setTitle(title)
        font = Font(font_file)
        chars = font.chars(gtype)
        text = ""
        n = 0
        for c in chars:
            text += c
            n += 1
            if n % 45 == 0:
                text += '\n'
        self.text = Label(text, pos=(0,0), anchor='ul',
                font_file=font_file, glyphs=gtype, color=(1, 1, 1, 1))

    def renderScene(self):
        super(GLApp, self).renderScene()
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        gl.glClearColor(1.0, 1.0, 1.0, 1.0)
        proj = glm.ortho(0, self.width, self.height, 0)
        self.cross.render()
        if self.text:
            self.text.render(proj)

    def onKeyboard(self, window, key, scancode, action, mode):
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            glfw.set_window_should_close(window, True)
        elif key in (glfw.KEY_UP , glfw.KEY_DOWN, glfw.KEY_LEFT,
                glfw.KEY_RIGHT) and action == glfw.PRESS:

            if key == glfw.KEY_LEFT and action == glfw.PRESS:
                self.itype -= 1
                if self.itype < 0:
                    self.itype = 0
            elif key == glfw.KEY_RIGHT and action == glfw.PRESS:
                self.itype += 1
                self.itype %= len(self.glyphTypes)
            if key == glfw.KEY_UP and action == glfw.PRESS:
                self.ifont -= 1
                if self.ifont < 0:
                    self.ifont = 0
            elif key == glfw.KEY_DOWN and action == glfw.PRESS:
                self.ifont += 1
                self.ifont %= len(self.fontlist)

            self.makeLabel()
        super(GLApp, self).onKeyboard(window, key, scancode, action, mode)

    def onResize(self, window, width, height):
        gl.glViewport(0, 0, width, height)

# =============================================================================
def displayDistilled(distilled):
    app = GLApp(distilled, 1024, 768)
    app.run()

# -----------------------------------------------------------------------------
def main(opts):
    ttdir = Path(opts.fontdir)
    files = [pn for pn in ttdir.glob("**/*.*tf")]
    print("Number of files:", len(files))

    distilled = []
    nbfdir = Path(FontDistiller.NBF_DIR)
    nbfdir.mkdir(parents=True, exist_ok=True)
    existing = [pn.name for pn in nbfdir.glob("*.nbf")]
    for i, ttf in enumerate(files):
        output = ttf.parent.joinpath(ttf.stem + FontDistiller.EXT)
        nbf = ttf.with_suffix(".nbf").name
        if not (nbf in existing) or opts.force :
            print(f"Distilling {ttf.name}...")
            out = FontDistiller(ttf, 0).save(output.name, defaultDir=True)
            if out:
                print(f"{i:03}--> {ttf.name} has been distilled to {out}")
                distilled.append(Path(out))
        else:
            print(f"{i:03}--> {ttf.name} already distilled")
            distilled.append(nbfdir.joinpath(nbf))

    print("Number of distilled files:", len(distilled))
    displayDistilled(distilled)

# =============================================================================
if __name__ == '__main__':
    import sys
    import argparse
    import signal
    from fontTools.ttLib import TTFont
    import argparse

    from ..font import Font

    signal.signal(signal.SIGINT, signal.SIG_DFL)
    parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description=("Distill, i.e. extract glyphs from ttf"
            " or odf files, save them to nbf file for later"
            " usage by LitGL and display the character map."
            " Use left, right arrow keys to cycle between glyph types"
            " and up down keys to move between fonts."))

    # Font file
    parser.add_argument('fontdir', nargs='?', help='Font directory',
            default=Path(__file__).parent.parent.joinpath(
            "examples/fonts"))
            #default="/usr/share/fonts/fonts-go/")

    parser.add_argument('-f', '--force', action='store_true',
            help='Force distillation of output file',
            default=False)

    parser.add_argument('-v', '--verbose', nargs='?', default=None,
            const='', help='Verbose processing (logging) level.'
            " Add more 'v' to increase verbosity. Must be the last option.")

    # Process options
    opts = parser.parse_args()
    if opts.verbose is not None:
        try:
            from rich.logging import RichHandler
        except ImportError:
            rootLogger = logging.getLogger()
            console = logging.StreamHandler(sys.stdout)
            console.setFormatter(logging.Formatter(
                '%(asctime)s -  %(levelname)s - %(message)s',
                datefmt='%d-%b-%y %H:%M:%S',))
            rootLogger.addHandler(console)
            rootLogger.setLevel(logging.INFO)
        else:
            FORMAT = '%(name)s.%(funcName)s: %(message)s'
            DATEFMT = "[%X]"
            logging.basicConfig(
                    level=logging.WARNING, format=FORMAT, datefmt=DATEFMT,
                    handlers=[RichHandler()])
            rootLogger = logging.getLogger("rich")

    #logging.basicConfig()
    #logging.getLogger().setLevel(logging.INFO)

    main(opts)
