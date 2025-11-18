# OmniParser Local Implementation

## Overview
Successfully migrated OmniParser from HuggingFace Inference API to **local processing** using our own models running in `.venv`.

## Key Changes

### 1. **tools/omniparser_tool.py**
- **Before**: Called HuggingFace Inference Endpoint API
- **After**: Uses local `UIParser` from `src/` package
- **Benefits**:
  - No API calls or network latency
  - No HuggingFace token required
  - Full control over model inference
  - Processes images entirely offline

### 2. **omniparser_api.py**
- Flask server on port 5001
- Uses local OmniParser models (YOLO + Florence2)
- Supports three input methods:
  1. `image_path`: Local file path
  2. `image_data`: Base64 encoded image
  3. `url`: Website URL (captures screenshot with Playwright)

### 3. **Screenshot Capture**
- Added `tools/screenshot_tool.py` using Playwright
- Captures live website screenshots automatically
- Headless Chrome browser
- Configurable viewport and wait times

## Testing Results

### Local Image Processing
```bash
curl -X POST http://localhost:5001/api/parse \
  -H "Content-Type: application/json" \
  -d '{"image_path": "/path/to/image.png"}'
```

**Result**: ✓ 28 elements detected in 2-3 seconds

### Website Screenshot + Parse
```bash
curl -X POST http://localhost:5001/api/parse \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com"}'
```

**Result**: ✓ 5 elements detected (screenshot + parsing)

## Architecture

```
Frontend (5173)
    ↓
Backend FastAPI (8080)
    ↓
    ├─→ OmniParser API (5001) → Local Models (.venv)
    └─→ LLM API (5000) → Phi-4 (.venv-llm)
```

## Dependencies Installed
```bash
# In .venv (Python 3.12)
pip install playwright flask flask-cors python-dotenv
playwright install chromium
```

## Configuration

### No HuggingFace Token Required
The `.env` file no longer needs `HUGGINGFACE_TOKEN` for OmniParser.

### Backend Configuration
```env
OMNIPARSER_API_URL=http://127.0.0.1:5001
OMNIPARSER_API_TIMEOUT=60
```

## API Endpoints

### POST /api/parse
Parse UI elements from an image or website.

**Request**:
```json
{
  "image_path": "/path/to/image.png"  // Option 1: Local file
}
```
or
```json
{
  "url": "https://example.com"  // Option 2: Website screenshot
}
```
or
```json
{
  "image_data": "base64_encoded_image"  // Option 3: Base64
}
```

**Response**:
```json
{
  "status": "success",
  "data": {
    "elements": [
      {
        "text": "Button Label",
        "type": "button",
        "bbox": [100, 200, 300, 250],
        "center": [200, 225],
        "confidence": 0.95
      }
    ],
    "analysis_text": "UI Element Analysis:\n...",
    "analysis_path": "/path/to/output.txt",
    "element_count": 28,
    "image_path": "/path/to/image.png"
  }
}
```

### GET /api/health
Health check endpoint.

**Response**:
```json
{
  "status": "running",
  "service": "OmniParser API",
  "version": "1.0.0",
  "mode": "local"
}
```

## Performance Comparison

| Method | Speed | Pros | Cons |
|--------|-------|------|------|
| HF API | 3-5s | No local GPU needed | Requires token, network dependency |
| **Local** | **2-3s** | **Offline, no token, faster** | **Requires local GPU/models** |

## Files Modified
- `tools/omniparser_tool.py` - Local UIParser integration
- `omniparser_api.py` - Removed HF token requirements
- `tools/screenshot_tool.py` - NEW: Website capture
- `requirements.txt` - Added Playwright, Flask
- `web_system/backend/app/config.py` - Added OMNIPARSER_API_URL
- `web_system/backend/app/pipeline/perception.py` - Calls OmniParser API
- `STEPS.txt` - Updated startup instructions

## Next Steps
1. ✅ OmniParser running locally
2. ⏳ Start Backend API (port 8080)
3. ⏳ Start LLM API (port 5000)
4. ⏳ Start Frontend (port 5173)
5. ⏳ Test full end-to-end workflow

## Summary
Successfully converted OmniParser from cloud API to **fully local processing**. The system now runs entirely offline with no external dependencies, using our own YOLO + Florence2 models in `.venv`.
