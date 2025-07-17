from pydantic import Field
from pydantic_settings import BaseSettings

# Define the environment variables

class Config(BaseSettings):
    port: str = Field(..., env="PORT")
    azure_openai_deployment: str = Field(..., env="AZURE_OPENAI_DEPLOYMENT")
    azure_openai_model: str = Field(..., env="AZURE_OPENAI_MODEL")
    azure_openai_api_version: str = Field(..., env="AZURE_OPENAI_API_VERSION")
    azure_openai_endpoint: str = Field(..., env="AZURE_OPENAI_ENDPOINT")
    azure_openai_api_key: str = Field(..., env="AZURE_OPENAI_API_KEY")
    llm_model_scope: str = Field(..., env="LLM_MODEL_SCOPE")
    environment: str = Field(..., env="ENVIRONMENT")
    azure_agent_api_key: str = Field(..., env="AZURE_AGENT_API_KEY")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

config = Config()