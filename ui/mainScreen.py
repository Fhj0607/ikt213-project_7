import cv2

from kivy.core.clipboard import Clipboard
from kivy.graphics import Color, RoundedRectangle
from kivy.graphics.texture import Texture
from kivy.metrics import dp
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.gridlayout import GridLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.textinput import TextInput


BACKGROUND = (0.035, 0.055, 0.09, 1)
SURFACE = (0.065, 0.095, 0.14, 1)
TEXT = (0.94, 0.96, 0.99, 1)
MUTED = (0.65, 0.72, 0.81, 1)
ACCENT = (0.27, 0.87, 0.73, 1)

# Sizes are in dp. They are converted at use time (not at import time) so
# Kivy's window/metrics are fully initialised first.
BUTTON_HEIGHT = 52
BUTTON_GAP = 10
BUTTON_MIN_WIDTH = 240
START_BUTTON_WIDTH = 320

PREVIEW_HINT = "Your page appears here\n\nChoose a capture mode to begin."


def rounded_background(widget, color, radius=16):
    with widget.canvas.before:
        tint = Color(*color)
        shape = RoundedRectangle(pos=widget.pos, size=widget.size, radius=[dp(radius)])

    def update(*_):
        shape.pos = widget.pos
        shape.size = widget.size

    widget.bind(pos=update, size=update)
    return tint


class MenuButton(Button):
    def __init__(self, text, emphasis=False, danger=False, **kwargs):
        super().__init__(text=text, font_size=dp(15), bold=True,
                         size_hint_y=None, height=dp(BUTTON_HEIGHT),
                         background_normal="", background_down="",
                         background_color=(0, 0, 0, 0),
                         halign="center", valign="middle", **kwargs)
        self.emphasis = emphasis
        self.base_color = ACCENT if emphasis else ((0.24, 0.10, 0.14, 1) if danger else (0.12, 0.17, 0.23, 1))
        self.color = BACKGROUND if emphasis else ((1, 0.74, 0.76, 1) if danger else TEXT)
        self.disabled_color = MUTED
        self.tint = rounded_background(self, self.base_color, 10)
        self.bind(state=self.refresh, disabled=self.refresh, size=self.wrap)

    def wrap(self, *_):
        self.text_size = (max(1, self.width - dp(20)), self.height)

    def refresh(self, *_):
        factor = 0.5 if self.disabled else (0.78 if self.state == "down" else 1)
        self.tint.rgba = tuple(c * factor for c in self.base_color[:3]) + (1,)


class MainScreen(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(orientation="vertical", padding=dp(24), spacing=dp(16), **kwargs)
        rounded_background(self, BACKGROUND, 0)

        # --- Header: brand on the left, stage on the right ------------------
        header = BoxLayout(size_hint_y=None, height=dp(56), spacing=dp(12))
        brand = BoxLayout(orientation="vertical")
        self.title = self.label("SnapPrint", font_size=30, bold=True,
                                size_hint_y=None, height=dp(38))
        brand.add_widget(self.title)
        brand.add_widget(self.label("YOUR WORDS, WITHIN REACH", font_size=10, color=ACCENT,
                                    size_hint_y=None, height=dp(16)))
        header.add_widget(brand)
        self.stage = self.label("READY", font_size=12, bold=True, color=ACCENT,
                                halign="right", size_hint_x=None, width=dp(160))
        header.add_widget(self.stage)
        self.add_widget(header)

        # --- Status line: grows with its text instead of reserving 64dp -----
        self.status = Label(text="", font_size=dp(17), color=TEXT, halign="left",
                            valign="middle", size_hint_y=None, height=dp(32))
        self.status.bind(
            width=lambda widget, width: setattr(widget, "text_size", (width, None)),
            texture_size=lambda widget, size: setattr(widget, "height", max(dp(32), size[1] + dp(8))))
        self.add_widget(self.status)

        # --- Actions ---------------------------------------------------------
        self.actions = GridLayout(cols=2, spacing=dp(BUTTON_GAP), size_hint_y=None)
        self.add_widget(self.actions)

        # --- Workspace: two cards --------------------------------------------
        self.workspace = BoxLayout(spacing=dp(16))
        self.add_widget(self.workspace)

        # RelativeLayout (not FloatLayout): children without a pos_hint are
        # positioned relative to the layout itself, so they stay inside the card.
        self.preview_card = self.card("01  /  CAPTURE", "Camera preview")
        preview = RelativeLayout()
        self.preview_hint = self.hint(PREVIEW_HINT)
        self.image_display = Image(opacity=0)  # an Image with no texture paints a white box
        self.image_display.bind(texture=self.onPreviewTexture)
        preview.add_widget(self.preview_hint)
        preview.add_widget(self.image_display)
        self.preview_card.add_widget(preview)
        self.workspace.add_widget(self.preview_card)

        self.copy_button = self.button("Copy")
        self.copy_button.size_hint_x = None
        self.copy_button.width = dp(84)
        self.copy_button.height = dp(36)
        self.copy_button.disabled = True
        self.copy_button.bind(on_release=self.on_copy_click)
        self.text_card = self.card("02  /  RECOGNIZE", "Collected text", self.copy_button)
        text_area = RelativeLayout()
        self.ocr_text = TextInput(
            text="", multiline=True, font_size=dp(18),
            foreground_color=TEXT, cursor_color=ACCENT,
            selection_color=(0.27, 0.87, 0.73, 0.3),
            background_normal="", background_active="",
            background_color=(0, 0, 0, 0),
            hint_text="Recognized pages appear here. Click to edit your notes.",
            hint_text_color=MUTED, padding=(dp(4), dp(8)))
        self.ocr_text.bind(text=lambda _, text: setattr(self.copy_button, "disabled", not text.strip()))
        text_area.add_widget(self.ocr_text)
        self.text_card.add_widget(text_area)
        self.workspace.add_widget(self.text_card)

        # --- Footer ----------------------------------------------------------
        self.footer = self.label("LOCAL OCR", font_size=11,
                                 color=MUTED, size_hint_y=None, height=dp(20))
        self.add_widget(self.footer)

        self.start_button = self.button("Start session (S)", emphasis=True)
        self.live_button = self.button("Live capture with spoken guidance (L)", emphasis=True)
        self.photos_button = self.button("Take photos or upload images (P)")
        self.capture_button = self.button("Take photo (C)", emphasis=True)
        self.upload_button = self.button("Upload image (U)")
        self.force_capture_button = self.button("Capture now (F)")
        self.scan_again_button = self.button("Scan next page (C)", emphasis=True)
        self.end_button = self.button("End session (Esc)", danger=True)
        self.pages_button = self.button("Single page (2 to toggle)")
        self.bind(size=self.reflow)
        self.showHome()

    # ------------------------------------------------------------------ helpers
    def label(self, text, font_size=18, color=TEXT, halign="left", **kwargs):
        label = Label(text=text, font_size=dp(font_size), color=color,
                      halign=halign, valign="middle", **kwargs)
        label.bind(size=lambda widget, size: setattr(widget, "text_size", size))
        return label

    def hint(self, text):
        label = Label(text=text, color=MUTED, font_size=dp(16), halign="center", valign="middle")
        label.bind(size=lambda widget, size: setattr(widget, "text_size", size))
        return label

    def card(self, eyebrow, title, action=None):
        card = BoxLayout(orientation="vertical", padding=dp(20), spacing=dp(8))
        rounded_background(card, SURFACE)
        card.add_widget(self.label(eyebrow, font_size=10, color=ACCENT,
                                   size_hint_y=None, height=dp(16)))
        heading = BoxLayout(size_hint_y=None, height=dp(36), spacing=dp(8))
        heading.add_widget(self.label(title, font_size=20, bold=True))
        if action is not None:
            heading.add_widget(action)
        card.add_widget(heading)
        return card

    def button(self, text, **kwargs):
        return MenuButton(text=text, **kwargs)

    def on_copy_click(self, *_):
        text = self.ocr_text.text
        if not text.strip():
            return
        try:
            Clipboard.copy(text)
        except Exception as error:
            self.status.text = "Could not copy notes: " + str(error)
        else:
            self.status.text = "Notes copied to clipboard."

    def onPreviewTexture(self, _, texture):
        self.image_display.opacity = 1 if texture else 0
        self.preview_hint.opacity = 0 if texture else 1

    # ------------------------------------------------------------------- layout
    def reflow(self, *_):
        width = self.width
        padding = dp(24) if width >= dp(580) else dp(12)
        self.padding = padding
        self.spacing = dp(16) if self.height >= dp(680) else dp(8)
        self.workspace.orientation = "horizontal" if width >= dp(760) else "vertical"

        count = len(self.actions.children)
        if count <= 1:
            # A lone primary action stays compact instead of a full-width slab.
            cols = 1
            self.actions.size_hint_x = None
            self.actions.width = min(dp(START_BUTTON_WIDTH), max(dp(100), width - 2 * padding))
        else:
            # As many columns as fit at a comfortable button width (one row when wide).
            fit = int((width - 2 * padding) // dp(BUTTON_MIN_WIDTH))
            cols = max(1, min(count, fit))
            self.actions.size_hint_x = 1
        self.actions.cols = cols

        rows = -(-count // cols) if count else 0
        self.actions.height = rows * dp(BUTTON_HEIGHT) + max(0, rows - 1) * dp(BUTTON_GAP)

    def setActions(self, buttons):
        self.actions.clear_widgets()
        for button in buttons:
            self.actions.add_widget(button)
        self.reflow()

    # ------------------------------------------------------------------- states
    def showLoading(self, message):
        self.showHome(message)
        self.stage.text = "INITIALIZING"
        self.start_button.text = "Loading models..."
        self.start_button.disabled = True

    def showLoadingError(self, message):
        self.showHome("Could not initialize recognition models: " + message)
        self.stage.text = "LOAD FAILED"
        self.start_button.text = "Retry initialization (S)"
        self.start_button.disabled = False

    def showHome(self, text="Start a session to collect pages for a document."):
        self.footer.text = "LOCAL OCR"
        self.stage.text = "READY"
        self.status.text = text
        self.preview_hint.text = PREVIEW_HINT
        self.image_display.texture = None
        self.setActions([self.start_button])

    def showModes(self):
        self.footer.text = "LOCAL OCR"
        self.stage.text = "CHOOSE A MODE"
        self.status.text = "How would you like to capture your pages? You can end the session at any time."
        self.preview_hint.text = PREVIEW_HINT
        self.ocr_text.text = ""
        self.setActions([self.live_button, self.photos_button, self.end_button])

    def showCapture(self, mode):
        self.footer.text = "LOCAL OCR    /    Space to stop speech" if mode == "live" else "LOCAL OCR"
        self.stage.text = "LIVE CAPTURE" if mode == "live" else "PHOTOS & FILES"
        self.status.text = ("Hold your page steady. Spoken guidance will help you position it."
                            if mode == "live" else "Take a photo or choose an image to add its text to your session.")
        self.preview_hint.text = ("Waiting for the camera..." if mode == "live"
                                  else "Photos you take appear here.")
        self.setActions(([self.scan_again_button, self.force_capture_button] if mode == "live" else
                         [self.capture_button, self.upload_button]) +
                        [self.pages_button, self.end_button])

    def displayFrame(self, frame):

        display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        display_frame = cv2.flip(display_frame, 0)

        height, width, channels = (display_frame.shape)

        texture = Texture.create(size=(width, height), colorfmt="rgb")

        texture.blit_buffer(
            display_frame.tobytes(),
            colorfmt="rgb",
            bufferfmt="ubyte"
        )

        self.image_display.texture = texture

    def displayOcrOutput(self, text):
        self.ocr_text.text = text
    def appendOcrText(self, text):
        current = self.ocr_text.text
        self.ocr_text.text = current + ("\n\n" if current else "") + text
