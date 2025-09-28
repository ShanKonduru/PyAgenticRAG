"""
Configuration management for PyAgenticRAG
"""
import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv
import yaml

load_dotenv()

@dataclass
class LLMConfig:
    """LLM Configuration"""
    provider: str = "ollama"  # ollama, openai, anthropic, etc.
    model: str = "llama3"
    temperature: float = 0.7
    max_tokens: int = 2000
    api_key: Optional[str] = None
    base_url: Optional[str] = None

@dataclass
class VectorDBConfig:
    """Vector Database Configuration"""
    provider: str = "chroma"  # chroma, pinecone, weaviate, qdrant
    connection_string: Optional[str] = None
    collection_name: str = "pyagentic_rag"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 1000
    chunk_overlap: int = 200

@dataclass
class DataSourceConfig:
    """Data Source Configuration"""
    enabled_sources: list = field(default_factory=lambda: ["local_files", "web_search"])
    local_data_path: str = "./data"
    web_search_api_key: Optional[str] = None
    database_configs: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AgentConfig:
    """Agent Configuration"""
    max_iterations: int = 5
    timeout_seconds: int = 60
    enable_memory: bool = True
    memory_window: int = 10

@dataclass
class APIConfig:
    """API Configuration"""
    host: str = "localhost"
    port: int = 8000
    debug: bool = False
    cors_origins: list = field(default_factory=lambda: ["*"])

@dataclass
class LoggingConfig:
    """Logging Configuration"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None
    max_file_size: str = "10MB"
    backup_count: int = 5

@dataclass
class AppConfig:
    """Main Application Configuration"""
    llm: LLMConfig = field(default_factory=LLMConfig)
    vector_db: VectorDBConfig = field(default_factory=VectorDBConfig)
    data_sources: DataSourceConfig = field(default_factory=DataSourceConfig)
    agents: AgentConfig = field(default_factory=AgentConfig)
    api: APIConfig = field(default_factory=APIConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def from_env(cls) -> 'AppConfig':
        """Load configuration from environment variables"""
        config = cls()
        
        # LLM Configuration
        config.llm.provider = os.getenv("LLM_PROVIDER", config.llm.provider)
        config.llm.model = os.getenv("LLM_MODEL", config.llm.model)
        config.llm.temperature = float(os.getenv("LLM_TEMPERATURE", config.llm.temperature))
        config.llm.api_key = os.getenv("LLM_API_KEY")
        config.llm.base_url = os.getenv("LLM_BASE_URL")
        
        # Vector DB Configuration
        config.vector_db.provider = os.getenv("VECTOR_DB_PROVIDER", config.vector_db.provider)
        config.vector_db.connection_string = os.getenv("VECTOR_DB_CONNECTION_STRING")
        config.vector_db.collection_name = os.getenv("VECTOR_DB_COLLECTION", config.vector_db.collection_name)
        
        # API Configuration
        config.api.host = os.getenv("API_HOST", config.api.host)
        config.api.port = int(os.getenv("API_PORT", config.api.port))
        config.api.debug = os.getenv("API_DEBUG", "false").lower() == "true"
        
        # Data Sources
        config.data_sources.local_data_path = os.getenv("LOCAL_DATA_PATH", config.data_sources.local_data_path)
        config.data_sources.web_search_api_key = os.getenv("WEB_SEARCH_API_KEY")
        
        return config
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'AppConfig':
        """Load configuration from YAML file"""
        with open(config_path, 'r') as file:
            data = yaml.safe_load(file)
        
        config = cls()
        
        # Update configuration with YAML data
        if 'llm' in data:
            for key, value in data['llm'].items():
                setattr(config.llm, key, value)
        
        if 'vector_db' in data:
            for key, value in data['vector_db'].items():
                setattr(config.vector_db, key, value)
        
        if 'data_sources' in data:
            for key, value in data['data_sources'].items():
                setattr(config.data_sources, key, value)
        
        if 'agents' in data:
            for key, value in data['agents'].items():
                setattr(config.agents, key, value)
        
        if 'api' in data:
            for key, value in data['api'].items():
                setattr(config.api, key, value)
        
        if 'logging' in data:
            for key, value in data['logging'].items():
                setattr(config.logging, key, value)
        
        return config

# Global configuration instance
_config: Optional[AppConfig] = None

def get_config() -> AppConfig:
    """Get the global configuration instance"""
    global _config
    if _config is None:
        # Try to load from config file first, then environment
        config_path = Path("config/config.yaml")
        if config_path.exists():
            _config = AppConfig.from_yaml(str(config_path))
        else:
            _config = AppConfig.from_env()
    return _config

def set_config(config: AppConfig):
    """Set the global configuration instance"""
    global _config
    _config = config