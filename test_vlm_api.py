#!/usr/bin/env python3
"""
Professional test suite for VLM API
Tests health check and online VLM endpoints
"""

import requests
import base64
import os
import sys
from pathlib import Path

# Load config to get correct API path
sys.path.insert(0, str(Path.cwd() / "src"))
try:
    from api_config import get_config
    config = get_config()
    API_BASE = config.api.base_path
except Exception:
    # Fallback if config not available
    API_BASE = "/api"

API_URL = f"http://localhost:5000{API_BASE}"

def test_health():
    """Test health endpoint"""
    print("=" * 70)
    print(f"Testing {API_BASE}/health...")
    print("=" * 70)
    
    try:
        response = requests.get(f"http://localhost:5000{API_BASE}/health", timeout=5)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"\nService: {data.get('service')} v{data.get('version')}")
            print("\nModels:")
            for model_type, info in data.get('models', {}).items():
                status = "✓ Ready" if info.get('ready') else "✗ Not Ready"
                print(f"  • {model_type.capitalize()}: {status}")
            print("\n" + "=" * 70 + "\n")
            return True
        else:
            print(f"Error: {response.text}")
            return False
    
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API")
        print("Make sure the server is running: python vlm_api.py\n")
        return False


def test_online_vlm():
    """Test online VLM with Qwen3-VL-Flash"""
    print("=" * 70)
    print(f"Testing {API_BASE}/vlm/online...")
    print("=" * 70)
    
    # Check for analysis file
    analysis_path = Path("data/outputs/form-example-2_analysis.txt")
    if not analysis_path.exists():
        print(f"⚠️  Analysis file not found: {analysis_path}")
        print("Run OmniParser first: python run_omniparser.py images/form-example-2.png\n")
        return False
    
    with open(analysis_path) as f:
        analysis_text = f.read()
    
    # Check for image
    image_path = Path("images/form-example-2.png")
    if not image_path.exists():
        print(f"❌ Image not found: {image_path}\n")
        return False
    
    # Encode image
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode()
    
    # Prepare request
    user_command = "Fill form with: Jose de la Rosa, 242343111, josedelarosaroja@jose.com, 2444 pine st, Seal Beach, CA, 90740"
    
    print(f"\n📝 Command: {user_command[:60]}...")
    print(f"🖼️  Image: {image_path.name}")
    print(f"📄 Analysis: {len(analysis_text)} chars")
    print("\n⏳ Sending request...")
    
    try:
        response = requests.post(
            f"{API_URL}/vlm/online",
            json={
                "image_url": f"data:image/png;base64,{image_b64}",
                "analysis_text": analysis_text,
                "user_command": user_command
            },
            timeout=60
        )
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            data = result.get('data', {})
            
            print("\n" + "=" * 70)
            print("✅ ACTION PLAN GENERATED")
            print("=" * 70)
            print(f"Model: {data.get('model', 'N/A')}")
            print(f"Provider: {data.get('provider', 'N/A')}")
            print("-" * 70)
            print(data.get('action_plan', 'No action plan'))
            print("=" * 70 + "\n")
            return True
        else:
            error_data = response.json()
            print(f"\n❌ Error: {error_data.get('error', 'Unknown error')}\n")
            return False
    
    except requests.exceptions.Timeout:
        print("\n❌ Request timed out (60s)\n")
        return False
    except Exception as e:
        print(f"\n❌ Exception: {str(e)}\n")
        return False


def test_llm():
    """Test LLM text-only endpoint (fast, no image required)"""
    print("=" * 70)
    print(f"Testing {API_BASE}/llm/parse...")
    print("=" * 70)
    
    # Check for analysis file
    analysis_path = Path("data/outputs/form-example-2_analysis.txt")
    if not analysis_path.exists():
        print(f"⚠️  Analysis file not found: {analysis_path}")
        print("Run OmniParser first: python run_omniparser.py images/form-example-2.png\n")
        return False
    
    with open(analysis_path) as f:
        analysis_text = f.read()
    
    # Prepare request (no image needed for LLM!)
    user_command = "Fill form with: Jose de la Rosa, 242343111, josedelarosaroja@jose.com, 2444 pine st, Seal Beach, CA, 90740"
    
    print(f"\n📝 Command: {user_command[:60]}...")
    print(f"📄 Analysis: {len(analysis_text)} chars")
    print(f"🚀 Mode: Text-only (no image processing)")
    print("\n⏳ Sending request...")
    
    try:
        response = requests.post(
            f"{API_URL}/llm/parse",
            json={
                "analysis_text": analysis_text,
                "user_command": user_command
            },
            timeout=60
        )
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            data = result.get('data', {})
            
            print("\n" + "=" * 70)
            print("✅ ACTION PLAN GENERATED (LLM)")
            print("=" * 70)
            print(f"Model: {data.get('model', 'N/A')}")
            print(f"Mode: {data.get('mode', 'N/A')}")
            print("-" * 70)
            print(data.get('action_plan', 'No action plan'))
            print("=" * 70 + "\n")
            return True
        else:
            error_data = response.json()
            print(f"\n❌ Error: {error_data.get('error', 'Unknown error')}\n")
            return False
    
    except requests.exceptions.Timeout:
        print("\n❌ Request timed out (60s)\n")
        return False
    except Exception as e:
        print(f"\n❌ Exception: {str(e)}\n")
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("VLM API Test Suite")
    print("=" * 70 + "\n")
    
    # Note: API key check moved to individual test
    # Run tests
    results = []
    
    # Test 1: Health check
    results.append(("Health Check", test_health()))
    
    # Test 2: LLM (fast, no image)
    print("⚡ Testing LLM endpoint (fast, text-only)...")
    results.append(("LLM Text-Only", test_llm()))
    
    # Test 3: Online VLM (requires API key)
    if os.getenv("DASHSCOPE_API_KEY"):
        print("🌐 Testing Online VLM endpoint (with image)...")
        results.append(("Online VLM", test_online_vlm()))
    else:
        print("⚠️  Skipping Online VLM test (no DASHSCOPE_API_KEY)\n")
    
    # Summary
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print("-" * 70)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 70 + "\n")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
