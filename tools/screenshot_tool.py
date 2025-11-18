"""
Website Screenshot Capture Tool
Uses Playwright to capture screenshots of websites
"""

import os
from datetime import datetime
from typing import Optional

def capture_website_screenshot(
    url: str,
    output_dir: str = "screenshots",
    viewport_width: int = 1920,
    viewport_height: int = 1080,
    wait_time: int = 2000,
    full_page: bool = False
) -> str:
    """
    Capture a screenshot of a website URL
    
    Args:
        url: Website URL to capture
        output_dir: Directory to save screenshots
        viewport_width: Browser viewport width
        viewport_height: Browser viewport height
        wait_time: Time to wait for page load (milliseconds)
        full_page: Capture full scrollable page
    
    Returns:
        Path to saved screenshot
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise ImportError(
            "Playwright not installed. Run: pip install playwright && playwright install chromium"
        )
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_url = url.replace('://', '_').replace('/', '_').replace('?', '_')[:50]
    filename = f"screenshot_{safe_url}_{timestamp}.png"
    filepath = os.path.join(output_dir, filename)
    
    # Capture screenshot
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={'width': viewport_width, 'height': viewport_height}
        )
        page = context.new_page()
        
        # Navigate to URL
        page.goto(url, wait_until='networkidle', timeout=30000)
        
        # Wait additional time for dynamic content
        page.wait_for_timeout(wait_time)
        
        # Take screenshot
        page.screenshot(path=filepath, full_page=full_page)
        
        browser.close()
    
    print(f"✓ Screenshot saved: {filepath}")
    return filepath


if __name__ == "__main__":
    # Test the screenshot tool
    test_url = "https://example.com"
    screenshot_path = capture_website_screenshot(test_url)
    print(f"Screenshot captured: {screenshot_path}")
