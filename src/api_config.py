"""
Configuration management for VLM API
Follows 12-factor app principles with environment variable overrides
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass


@dataclass
class ServerConfig:
    """Server configuration"""
    host: str
    port: int
    debug: bool


@dataclass
class LocalModelConfig:
    """Local model configuration"""
    enabled: bool
    model_name: str
    device: str
    auto_load: bool


@dataclass
class OnlineModelConfig:
    """Online model configuration"""
    enabled: bool
    provider: str
    model_name: str
    base_url: str
    timeout: int
    max_retries: int


@dataclass
class LLMConfig:
    """LLM (text-only) model configuration"""
    enabled: bool
    model_name: str
    device: str
    torch_dtype: str
    max_new_tokens: int
    auto_load: bool
    quantization: Optional[str]


@dataclass
class AzureOpenAIConfig:
    """Azure OpenAI configuration"""
    enabled: bool
    endpoint: Optional[str]
    api_key: Optional[str]
    api_version: str
    deployment_name: str
    max_tokens: int
    temperature: float


@dataclass
class APIConfig:
    """API metadata configuration"""
    name: str
    version: str
    base_path: str


@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str
    format: str


@dataclass
class SecurityConfig:
    """Security configuration"""
    require_api_key: bool
    allowed_origins: list
    max_content_length: int


@dataclass
class RateLimitConfig:
    """Rate limiting configuration"""
    enabled: bool
    requests_per_minute: int


class Config:
    """Main configuration class with environment variable overrides"""
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize configuration from YAML file with env var overrides
        
        Args:
            config_path: Path to YAML config file. Defaults to config.yaml
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config.yaml"
        
        self.config_path = config_path
        self._config_data = self._load_config()
        
        # Initialize config objects
        self.server = self._init_server_config()
        self.api = self._init_api_config()
        self.local_model = self._init_local_model_config()
        self.online_model = self._init_online_model_config()
        self.llm = self._init_llm_config()
        self.azure_openai = self._init_azure_openai_config()
        self.logging = self._init_logging_config()
        self.security = self._init_security_config()
        self.rate_limit = self._init_rate_limit_config()
        
        # Special: API keys from environment only (never in config file)
        self.dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")
        self.api_key = os.getenv("VLM_API_KEY")  # For API authentication
    
    def _load_config(self) -> Dict[str, Any]:
        """Load YAML configuration file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _get_env_or_config(self, env_key: str, config_path: list, default: Any = None) -> Any:
        """
        Get value from environment variable or config file
        Environment variables take precedence
        
        Args:
            env_key: Environment variable name
            config_path: List of keys to traverse config dict
            default: Default value if not found
        """
        # Try environment variable first
        env_value = os.getenv(env_key)
        if env_value is not None:
            # Convert string to proper type
            if isinstance(default, bool):
                return env_value.lower() in ('true', '1', 'yes')
            elif isinstance(default, int):
                return int(env_value)
            return env_value
        
        # Fall back to config file
        value = self._config_data
        for key in config_path:
            if isinstance(value, dict):
                value = value.get(key, default)
            else:
                return default
        return value if value is not None else default
    
    def _init_server_config(self) -> ServerConfig:
        """Initialize server configuration"""
        return ServerConfig(
            host=self._get_env_or_config("VLM_HOST", ["server", "host"], "0.0.0.0"),
            port=int(self._get_env_or_config("VLM_PORT", ["server", "port"], 5000)),
            debug=self._get_env_or_config("VLM_DEBUG", ["server", "debug"], False)
        )
    
    def _init_api_config(self) -> APIConfig:
        """Initialize API configuration"""
        return APIConfig(
            name=self._config_data.get("api", {}).get("name", "VLM API"),
            version=self._config_data.get("api", {}).get("version", "1.0.0"),
            base_path=self._config_data.get("api", {}).get("base_path", "/api")
        )
    
    def _init_local_model_config(self) -> LocalModelConfig:
        """Initialize local model configuration"""
        models = self._config_data.get("models", {})
        local = models.get("local", {})
        
        return LocalModelConfig(
            enabled=self._get_env_or_config("VLM_LOCAL_ENABLED", ["models", "local", "enabled"], True),
            model_name=self._get_env_or_config("VLM_LOCAL_MODEL", ["models", "local", "model_name"], "Qwen/Qwen2-VL-7B-Instruct"),
            device=self._get_env_or_config("VLM_DEVICE", ["models", "local", "device"], "cuda"),
            auto_load=local.get("auto_load", True)
        )
    
    def _init_online_model_config(self) -> OnlineModelConfig:
        """Initialize online model configuration"""
        models = self._config_data.get("models", {})
        online = models.get("online", {})
        
        return OnlineModelConfig(
            enabled=self._get_env_or_config("VLM_ONLINE_ENABLED", ["models", "online", "enabled"], True),
            provider=online.get("provider", "dashscope"),
            model_name=self._get_env_or_config("VLM_ONLINE_MODEL", ["models", "online", "model_name"], "qwen-vl-plus"),
            base_url=self._get_env_or_config("VLM_ONLINE_BASE_URL", ["models", "online", "base_url"], "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"),
            timeout=int(self._get_env_or_config("VLM_TIMEOUT", ["models", "online", "timeout"], 30)),
            max_retries=int(online.get("max_retries", 3))
        )
    
    def _init_llm_config(self) -> LLMConfig:
        """Initialize LLM (text-only) model configuration"""
        models = self._config_data.get("models", {})
        llm = models.get("llm", {})
        
        return LLMConfig(
            enabled=self._get_env_or_config("VLM_LLM_ENABLED", ["models", "llm", "enabled"], True),
            model_name=self._get_env_or_config("VLM_LLM_MODEL", ["models", "llm", "model_name"], "google/gemma-2-9b"),
            device=self._get_env_or_config("VLM_LLM_DEVICE", ["models", "llm", "device"], "auto"),
            torch_dtype=llm.get("torch_dtype", "bfloat16"),
            max_new_tokens=int(llm.get("max_new_tokens", 512)),
            auto_load=llm.get("auto_load", True),
            quantization=llm.get("quantization")
        )
    
    def _init_azure_openai_config(self) -> AzureOpenAIConfig:
        """Initialize Azure OpenAI configuration"""
        models = self._config_data.get("models", {})
        azure = models.get("azure_openai", {})
        
        return AzureOpenAIConfig(
            enabled=self._get_env_or_config("AZURE_OPENAI_ENABLED", ["models", "azure_openai", "enabled"], False),
            endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", azure.get("api_version", "2024-02-15-preview")),
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", azure.get("deployment_name", "o1-mini")),
            max_tokens=int(azure.get("max_tokens", 512)),
            temperature=float(azure.get("temperature", 0.7))
        )
    
    def _init_logging_config(self) -> LoggingConfig:
        """Initialize logging configuration"""
        logging_cfg = self._config_data.get("logging", {})
        
        return LoggingConfig(
            level=self._get_env_or_config("VLM_LOG_LEVEL", ["logging", "level"], "INFO"),
            format=logging_cfg.get("format", "[%(asctime)s] %(levelname)s - %(message)s")
        )
    
    def _init_security_config(self) -> SecurityConfig:
        """Initialize security configuration"""
        security = self._config_data.get("security", {})
        
        return SecurityConfig(
            require_api_key=self._get_env_or_config("VLM_REQUIRE_API_KEY", ["security", "require_api_key"], False),
            allowed_origins=security.get("allowed_origins", ["*"]),
            max_content_length=int(security.get("max_content_length", 16777216))
        )
    
    def _init_rate_limit_config(self) -> RateLimitConfig:
        """Initialize rate limiting configuration"""
        rate_limit = self._config_data.get("rate_limiting", {})
        
        return RateLimitConfig(
            enabled=self._get_env_or_config("VLM_RATE_LIMIT", ["rate_limiting", "enabled"], False),
            requests_per_minute=int(rate_limit.get("requests_per_minute", 60))
        )
    
    def get_prompt_template(self, prompt_type: str) -> str:
        """Get prompt template from config"""
        prompts = self._config_data.get("prompts", {})
        return prompts.get(prompt_type, "")
    
    def validate(self) -> tuple[bool, list[str]]:
        """
        Validate configuration
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check online API key if online model is enabled
        if self.online_model.enabled and not self.dashscope_api_key:
            errors.append("DASHSCOPE_API_KEY environment variable not set but online model is enabled")
        
        # Check API key if authentication is required
        if self.security.require_api_key and not self.api_key:
            errors.append("VLM_API_KEY environment variable not set but API authentication is required")
        
        # Validate port range
        if not (1 <= self.server.port <= 65535):
            errors.append(f"Invalid port number: {self.server.port}")
        
        # Validate device
        valid_devices = ["cuda", "cpu", "mps"]
        if self.local_model.device not in valid_devices:
            errors.append(f"Invalid device: {self.local_model.device}. Must be one of {valid_devices}")
        
        return (len(errors) == 0, errors)
    
    def __repr__(self) -> str:
        """String representation of config"""
        return f"Config(server={self.server.host}:{self.server.port}, local={self.local_model.enabled}, online={self.online_model.enabled}, llm={self.llm.enabled})"


# Singleton instance
_config_instance: Optional[Config] = None


def get_config(config_path: Optional[Path] = None) -> Config:
    """
    Get configuration singleton
    
    Args:
        config_path: Path to config file (only used on first call)
    
    Returns:
        Config instance
    """
    global _config_instance
    
    if _config_instance is None:
        _config_instance = Config(config_path)
    
    return _config_instance


def reload_config(config_path: Optional[Path] = None) -> Config:
    """
    Force reload configuration
    
    Args:
        config_path: Path to config file
    
    Returns:
        New Config instance
    """
    global _config_instance
    _config_instance = Config(config_path)
    return _config_instance
