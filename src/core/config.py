"""运行配置：从环境变量读取，提供合理默认值。"""
import os


class Settings:
    def __init__(self):
        self.model = os.getenv("SENTINEL_MODEL", "gpt-4o-mini")
        self.base_url = os.getenv("OPENAI_BASE_URL", "")
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        self.timeout = float(os.getenv("SENTINEL_TIMEOUT", "15"))


settings = Settings()
