"""Example following OmniParser tutorial structure."""

from pathlib import Path
from PIL import Image
from src import UIParser, GraniteVLM, ScreenCapture, setup_directories
from src.config import OUTPUT_DIR, SCREENSHOT_DIR


def quick_start_example():
    """Quick start example following the tutorial."""
    setup_directories()
    
    parser = UIParser()
    capture = ScreenCapture()
    
    print("Capturing screenshot...")
    img = capture.capture_active_window()
    
    print("Parsing UI elements...")
    result = parser.parse(img)
    
    for element in result.elements:
        print(f"{element.type}: {element.text} at {element.bbox}")
    
    login_button = result.find(text_contains="login")
    if login_button:
        print(f"Found login element: {login_button}")
    
    print("Creating visualization...")
    vis = result.visualize(img, show_labels=True)
    vis.save(OUTPUT_DIR / "parsed_screenshot.png")
    print("Visualization saved!")


def vlm_integration_example():
    """Example with VLM integration."""
    setup_directories()
    
    vlm = GraniteVLM()
    capture = ScreenCapture()
    
    print("Capturing and analyzing UI...")
    img = capture.capture_active_window()
    
    analysis = vlm.analyze_ui(img, "What buttons are available in this interface?")
    print(f"VLM Analysis: {analysis}")
    
    action_suggestion = vlm.suggest_action(img, "log into the application")
    print(f"Action suggestion: {action_suggestion}")


def agent_step_example(goal: str):
    """Agent step example from the tutorial."""
    capture = ScreenCapture()
    parser = UIParser()
    vlm = GraniteVLM()
    
    img = capture.capture_active_window()
    res = parser.parse(img)
    
    if goal == "submit form":
        btn = res.find(text_contains="submit")
        if btn:
            print(f"Found submit button at: {btn.center}")
            return btn.center
    
    analysis = vlm.suggest_action(img, goal)
    print(f"VLM suggests: {analysis}")
    
    return None


if __name__ == "__main__":
    print("Running quick start example...")
    quick_start_example()
    
    print("\nRunning VLM integration example...")
    vlm_integration_example()
    
    print("\nRunning agent step example...")
    agent_step_example("submit form")
