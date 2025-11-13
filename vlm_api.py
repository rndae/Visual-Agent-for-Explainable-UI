"""
VLM Parsing REST API
Professional Flask API for UI automation with vision-language models
"""

import sys
import logging
from pathlib import Path
from typing import Optional

# Add to path
sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path.cwd() / "src"))

from vlm_client import QwenVLMClient
from llm_client import LLMClient
from api_config import get_config
from flask import Flask, request, jsonify, g
from flask_cors import CORS
from openai import OpenAI
from functools import wraps

# Initialize configuration
config = get_config()

# Validate configuration
is_valid, errors = config.validate()
if not is_valid:
    print("❌ Configuration validation failed:")
    for error in errors:
        print(f"  - {error}")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=getattr(logging, config.logging.level),
    format=config.logging.format
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = config.security.max_content_length
CORS(app, origins=config.security.allowed_origins)

# Initialize clients
qwen_online_client: Optional[OpenAI] = None
vlm_client: Optional[QwenVLMClient] = None
llm_client: Optional[LLMClient] = None

if config.online_model.enabled:
    if config.dashscope_api_key:
        logger.info("Initializing online VLM client...")
        qwen_online_client = OpenAI(
            api_key=config.dashscope_api_key,
            base_url=config.online_model.base_url,
            timeout=config.online_model.timeout,
            max_retries=config.online_model.max_retries
        )
        logger.info("✓ Online VLM client ready")
    else:
        logger.warning("Online model enabled but DASHSCOPE_API_KEY not set")

if config.local_model.enabled:
    logger.info("Initializing local VLM client...")
    vlm_client = QwenVLMClient(auto_load=config.local_model.auto_load)
    logger.info("✓ Local VLM client ready")

if config.llm.enabled:
    logger.info("Initializing LLM client...")
    llm_client = LLMClient(
        model_name=config.llm.model_name,
        device=config.llm.device,
        torch_dtype=config.llm.torch_dtype,
        max_new_tokens=config.llm.max_new_tokens,
        quantization=config.llm.quantization,
        auto_load=config.llm.auto_load
    )
    logger.info("✓ LLM client ready")


def require_api_key(f):
    """Decorator to require API key authentication"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not config.security.require_api_key:
            return f(*args, **kwargs)
        
        api_key = request.headers.get('X-API-Key')
        if not api_key or api_key != config.api_key:
            logger.warning(f"Unauthorized API access attempt from {request.remote_addr}")
            return jsonify({"status": "error", "error": "Invalid or missing API key"}), 401
        
        return f(*args, **kwargs)
    return decorated_function


@app.route(f'{config.api.base_path}/vlm/parse', methods=['POST'])
@require_api_key
def parse_ui():
    """Local VLM parsing (uses local Qwen3-VL-8B model)"""
    if not config.local_model.enabled:
        return jsonify({"status": "error", "error": "Local model is disabled"}), 503
    
    if vlm_client is None:
        return jsonify({"status": "error", "error": "Local VLM client not initialized"}), 503
    
    try:
        data = request.json
        image_path = data.get('image_path')
        analysis_path = data.get('analysis_path')
        user_command = data.get('user_command')
        
        if not all([image_path, analysis_path, user_command]):
            return jsonify({
                "status": "error", 
                "error": "Missing required fields: image_path, analysis_path, user_command"
            }), 400
        
        logger.info(f"Processing local VLM request: {user_command[:50]}...")
        
        result = vlm_client.run_with_omniparser(
            image_path=image_path,
            analysis_txt_path=analysis_path,
            user_command=user_command
        )
        
        logger.info("Local VLM request completed successfully")
        return jsonify({"status": "success", "data": result})
    
    except Exception as e:
        logger.error(f"Local VLM error: {str(e)}", exc_info=True)
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route(f'{config.api.base_path}/vlm/online', methods=['POST'])
@require_api_key
def parse_ui_online():
    """Online VLM parsing using Qwen3-VL-Flash (faster, requires API key)"""
    if not config.online_model.enabled:
        return jsonify({"status": "error", "error": "Online model is disabled"}), 503
    
    if qwen_online_client is None:
        return jsonify({"status": "error", "error": "Online VLM client not initialized"}), 503
    
    try:
        data = request.json
        image_url = data.get('image_url')  # URL or base64
        analysis_text = data.get('analysis_text', '')
        user_command = data.get('user_command', '')
        
        if not image_url:
            return jsonify({"status": "error", "error": "image_url is required"}), 400
        
        logger.info(f"Processing online VLM request: {user_command[:50]}...")
        
        # Build prompt using config templates
        system_prompt = config.get_prompt_template("system_message")
        action_format = config.get_prompt_template("action_format")
        
        prompt = f"""{system_prompt}

UI Elements Detected:
{analysis_text}

User Task: {user_command}

{action_format}

Provide the action sequence:"""
        
        # Call online model
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        completion = qwen_online_client.chat.completions.create(
            model=config.online_model.model_name,
            messages=messages,
            stream=False
        )
        
        action_plan = completion.choices[0].message.content
        
        result = {
            "action_plan": action_plan,
            "model": config.online_model.model_name,
            "provider": config.online_model.provider
        }
        
        logger.info("Online VLM request completed successfully")
        return jsonify({"status": "success", "data": result})
        
    except Exception as e:
        logger.error(f"Online VLM error: {str(e)}", exc_info=True)
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route(f'{config.api.base_path}/vlm/online/stream', methods=['POST'])
@require_api_key
def parse_ui_online_stream():
    """Streaming version of online VLM parsing"""
    if not config.online_model.enabled:
        return jsonify({"status": "error", "error": "Online model is disabled"}), 503
    
    if qwen_online_client is None:
        return jsonify({"status": "error", "error": "Online VLM client not initialized"}), 503
    
    try:
        data = request.json
        image_url = data.get('image_url')
        analysis_text = data.get('analysis_text', '')
        user_command = data.get('user_command', '')
        
        if not image_url:
            return jsonify({"status": "error", "error": "image_url is required"}), 400
        
        logger.info(f"Processing streaming online VLM request: {user_command[:50]}...")
        
        # Compact prompt for streaming
        prompt = f"""UI automation assistant. Generate executable commands.

Elements: {analysis_text}
Task: {user_command}

Format: Click(x,y,id,"desc") | Type(x,y,id,"field","text") | Submit(x,y,id,"btn")"""
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        completion = qwen_online_client.chat.completions.create(
            model=config.online_model.model_name,
            messages=messages,
            stream=True
        )
        
        full_content = ""
        for chunk in completion:
            if chunk.choices[0].delta.content is None:
                continue
            full_content += chunk.choices[0].delta.content
        
        result = {
            "action_plan": full_content,
            "model": config.online_model.model_name,
            "provider": config.online_model.provider
        }
        
        logger.info("Streaming online VLM request completed successfully")
        return jsonify({"status": "success", "data": result})
        
    except Exception as e:
        logger.error(f"Streaming online VLM error: {str(e)}", exc_info=True)
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route(f'{config.api.base_path}/llm/parse', methods=['POST'])
@require_api_key
def parse_ui_llm():
    """
    Text-only LLM parsing using Llama 3.1 8B (fast, no image required)
    Uses only OmniParser text analysis for reasoning
    """
    if not config.llm.enabled:
        return jsonify({"status": "error", "error": "LLM model is disabled"}), 503
    
    if llm_client is None:
        return jsonify({"status": "error", "error": "LLM client not initialized"}), 503
    
    try:
        data = request.json
        analysis_text = data.get('analysis_text', '')
        user_command = data.get('user_command', '')
        
        # Optional: can also accept analysis_path
        analysis_path = data.get('analysis_path')
        
        if not analysis_text and not analysis_path:
            return jsonify({
                "status": "error", 
                "error": "Either analysis_text or analysis_path is required"
            }), 400
        
        if not user_command:
            return jsonify({
                "status": "error", 
                "error": "user_command is required"
            }), 400
        
        logger.info(f"Processing LLM request: {user_command[:50]}...")
        
        result = llm_client.run_with_omniparser(
            analysis_txt_path=analysis_path,
            analysis_text=analysis_text if analysis_text else None,
            user_command=user_command
        )
        
        logger.info("LLM request completed successfully")
        return jsonify({"status": "success", "data": result})
    
    except Exception as e:
        logger.error(f"LLM error: {str(e)}", exc_info=True)
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route(f'{config.api.base_path}/health', methods=['GET'])
def health():
    """Health check endpoint with service status"""
    status_info = {
        "status": "running",
        "service": config.api.name,
        "version": config.api.version,
        "models": {
            "local": {
                "enabled": config.local_model.enabled,
                "ready": vlm_client is not None
            },
            "online": {
                "enabled": config.online_model.enabled,
                "ready": qwen_online_client is not None,
                "provider": config.online_model.provider if config.online_model.enabled else None
            },
            "llm": {
                "enabled": config.llm.enabled,
                "ready": llm_client is not None and llm_client.is_loaded(),
                "model": config.llm.model_name if config.llm.enabled else None
            }
        },
        "endpoints": {
            f"{config.api.base_path}/vlm/parse": "Local VLM inference (with image)",
            f"{config.api.base_path}/vlm/online": "Online VLM inference (with image)",
            f"{config.api.base_path}/vlm/online/stream": "Streaming online VLM (with image)",
            f"{config.api.base_path}/llm/parse": "LLM inference (text-only, fast)",
            f"{config.api.base_path}/health": "Health check"
        }
    }
    
    return jsonify(status_info)


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle request too large errors"""
    logger.warning(f"Request too large from {request.remote_addr}")
    return jsonify({
        "status": "error",
        "error": f"Request too large. Max size: {config.security.max_content_length} bytes"
    }), 413


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        "status": "error",
        "error": "Endpoint not found",
        "available_endpoints": list({
            f"{config.api.base_path}/vlm/parse": "POST",
            f"{config.api.base_path}/vlm/online": "POST",
            f"{config.api.base_path}/vlm/online/stream": "POST",
            f"{config.api.base_path}/llm/parse": "POST",
            f"{config.api.base_path}/health": "GET"
        }.keys())
    }), 404


if __name__ == "__main__":
    print("\n" + "="*70)
    print(f"🚀 {config.api.name} v{config.api.version}")
    print("="*70)
    print(f"Server: http://{config.server.host}:{config.server.port}")
    print(f"\nConfiguration:")
    print(f"  • Local Model:  {'✓ Enabled' if config.local_model.enabled else '✗ Disabled'}")
    print(f"  • Online Model: {'✓ Enabled' if config.online_model.enabled else '✗ Disabled'}")
    if config.online_model.enabled:
        print(f"    - Provider: {config.online_model.provider}")
        print(f"    - Model: {config.online_model.model_name}")
    print(f"  • LLM (Text):   {'✓ Enabled' if config.llm.enabled else '✗ Disabled'}")
    if config.llm.enabled:
        print(f"    - Model: {config.llm.model_name}")
        print(f"    - Device: {config.llm.device}")
    print(f"  • Authentication: {'✓ Required' if config.security.require_api_key else '✗ Optional'}")
    print(f"  • Rate Limiting: {'✓ Enabled' if config.rate_limit.enabled else '✗ Disabled'}")
    print(f"\nEndpoints:")
    print(f"  POST {config.api.base_path}/vlm/parse         - Local VLM (with image)")
    print(f"  POST {config.api.base_path}/vlm/online        - Online VLM (with image)")
    print(f"  POST {config.api.base_path}/vlm/online/stream - Online VLM streaming")
    print(f"  POST {config.api.base_path}/llm/parse         - LLM text-only (fast)")
    print(f"  GET  {config.api.base_path}/health            - Health check")
    print("="*70 + "\n")
    
    logger.info(f"Starting {config.api.name} on {config.server.host}:{config.server.port}")
    
    app.run(
        host=config.server.host,
        port=config.server.port,
        debug=config.server.debug
    )

