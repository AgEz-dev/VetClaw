"""运行配置：从环境变量读取，提供合理默认值。"""
import os
from dotenv import load_dotenv

load_dotenv()
class Settings:
    def __init__(self):
        self.model = os.getenv("VETCLAW_MODEL", "gpt-4o-mini")
        self.base_url = os.getenv("OPENAI_BASE_URL", "")
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        # SDK 层 read timeout：单次模型请求两次字节间的最大间隔，防连接挂死
        self.timeout = float(os.getenv("VETCLAW_TIMEOUT", "15"))
        # Agent 层总时长守卫：整个 ReAct 生命周期（含多轮调用+工具执行）上限
        self.total_timeout = float(os.getenv("VETCLAW_TOTAL_TIMEOUT", "30"))
        # CORS 白名单，逗号分隔。allow_credentials=True 时不能用 "*"，下面自动防御。
        raw_origins = os.getenv(
            "CORS_ORIGINS",
            "http://localhost:5173,http://127.0.0.1:5173",
        )
        self.cors_origins = [o.strip() for o in raw_origins.split(",") if o.strip()]


settings = Settings()
