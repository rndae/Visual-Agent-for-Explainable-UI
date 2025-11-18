# requires PyQt6 to be installed

import sys
import time
import webbrowser
import os
import pyautogui
import pyscreeze
import PIL
from datetime import datetime
from PyQt6.QtWidgets import QApplication, QMainWindow
from PyQt6.QtCore import Qt, QTimer, QRect, QPoint
from PyQt6.QtGui import QPainter, QColor, QPen, QFont, QScreen

# --- Configuration ---
LOG_FILE = 'actions.log'
SCREENSHOT_DIR = 'screenshots'
os.makedirs(SCREENSHOT_DIR, exist_ok=True)
# Make sure you have a 'login_demo.html' or similar form for it to interact with
LOGIN_PAGE_PATH = 'login_demo.html' 

class OverlayWindow(QMainWindow):
    """
    A transparent, frameless, full-screen window to draw visualizations on.
    """
    def __init__(self):
        super().__init__()
        self.boxes = []
        self.texts = []
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        screen_geometry = QScreen.availableGeometry(QApplication.primaryScreen())
        self.setGeometry(screen_geometry)

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        for box_info in self.boxes:
            rect, color, width = box_info
            pen = QPen(color, width)
            pen.setStyle(Qt.PenStyle.SolidLine)
            painter.setPen(pen)
            painter.drawRect(rect)
        for text_info in self.texts:
            point, text, color, font_size = text_info
            pen = QPen(color)
            font = QFont('Arial', font_size)
            painter.setPen(pen)
            painter.setFont(font)
            painter.drawText(point, text)

    def draw_box(self, x, y, width, height, color=QColor(255, 0, 0, 200), line_width=2):
        self.boxes.append((QRect(x, y, width, height), color, line_width))
        self.update()

    def draw_text(self, x, y, text, color=QColor(0, 100, 255, 220), font_size=12):
        self.texts.append((QPoint(x, y), text, color, font_size))
        self.update()

    def clear_visuals(self):
        self.boxes.clear()
        self.texts.clear()
        self.update()

# --- Agent Action and Logging Functions ---

def log_action(description: str):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    log_entry = f"[{timestamp}] - {description}\n"
    print(log_entry.strip())
    with open(LOG_FILE, 'a') as f:
        f.write(log_entry)

def click_at(x: int, y: int, overlay: OverlayWindow):
    log_action(f"CLICK at ({x}, {y})")
    overlay.draw_box(x - 5, y - 5, 10, 10, QColor(0, 255, 0, 200), 2)
    pyautogui.click(x, y)

def type_text(x: int, y: int, text: str, overlay: OverlayWindow):
    """
    Clicks at a position to focus and then types text.
    Includes a small delay to ensure the window focus is correct before typing.
    """
    log_action(f"TYPE '{text}' at ({x}, {y})")
    pyautogui.click(x, y)
    
    # Pause for a fraction of a second to allow the OS to switch window focus.
    time.sleep(0.2)

    pyautogui.write(text, interval=0.1)

def scroll(amount: int, overlay: OverlayWindow):
    """
    Scrolls the screen. 
    Positive int = Scroll UP
    Negative int = Scroll DOWN
    """
    direction = "UP" if amount > 0 else "DOWN"
    log_action(f"SCROLL {direction} amount: {abs(amount)}")
    
    # Get center of screen for notification
    screen_geo = QScreen.availableGeometry(QApplication.primaryScreen())
    center_x = screen_geo.width() // 2
    center_y = screen_geo.height() // 2
    
    # Draw a large visible indicator
    overlay.draw_text(center_x - 100, center_y, f"SCROLLING {direction}...", QColor(255, 165, 0, 255), 30)
    
    time.sleep(0.5) # Pause so the visual is seen
    pyautogui.scroll(amount)

def take_screenshot(overlay: OverlayWindow) -> str:
    """
    Captures the screen and saves it to the screenshots directory.
    Returns the path to the saved file.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(SCREENSHOT_DIR, f"capture_{timestamp}.png")
    screenshot = pyautogui.screenshot()
    screenshot.save(filename)
    log_action(f"Saved screenshot to '{filename}'")
    return filename

# --- Main Demonstration ---

def run_demonstration(app, overlay):
    """
    Executes a predefined sequence of actions with more reliable timing.
    """
    log_action("Starting agent demonstration...")
    
    # Example: open a local HTML file for the demo
    # if os.path.exists(LOGIN_PAGE_PATH):
    #     abs_path = os.path.abspath(LOGIN_PAGE_PATH)
    #     webbrowser.open(f'file://{abs_path}')

    # --- Step 1: Identify and type in the first field ---
    QTimer.singleShot(1500, lambda: take_screenshot(overlay))
    QTimer.singleShot(2000, lambda: overlay.draw_box(755, 500, 400, 90, QColor("red")))
    QTimer.singleShot(2500, lambda: overlay.draw_text(755, 495, "Targeting 'Username' field", QColor("red")))
    QTimer.singleShot(4000, lambda: type_text(855, 545, "John Doe", overlay))

    # --- Step 2: Clear visuals and move to the next field ---
    QTimer.singleShot(6500, lambda: overlay.clear_visuals())
    QTimer.singleShot(7000, lambda: overlay.draw_box(755, 590, 400, 90, QColor("blue")))
    QTimer.singleShot(7500, lambda: overlay.draw_text(755, 585, "Targeting 'Password' field", QColor("blue")))

    # --- Step 3: Type in the second field ---
    QTimer.singleShot(9000, lambda: type_text(855, 635, "12345678", overlay))

    # --- Step 4: Scroll Down (NEW STEP) ---
    # Simulating scrolling down to find the button
    QTimer.singleShot(11000, lambda: overlay.clear_visuals())
    # Scroll down by sending negative integer
    QTimer.singleShot(11500, lambda: scroll(-500, overlay)) 

    # --- Step 5: Click the submit button ---
    QTimer.singleShot(12500, lambda: overlay.clear_visuals()) # Clear the scroll text
    QTimer.singleShot(13000, lambda: click_at(955, 715, overlay))

    # --- Step 6: Finalize and close ---
    QTimer.singleShot(14000, lambda: overlay.clear_visuals())
    QTimer.singleShot(14500, lambda: log_action("Demonstration finished."))
    QTimer.singleShot(15000, app.quit)


if __name__ == '__main__':
    with open(LOG_FILE, 'w') as f:
        f.write("--- Agent Action Log ---\n")

    app = QApplication(sys.argv)
    overlay = OverlayWindow()
    overlay.show()

    # Start the demonstration after the GUI is up and running
    QTimer.singleShot(500, lambda: run_demonstration(app, overlay))

    sys.exit(app.exec())