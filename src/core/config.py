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
        # ReAct 最大步数（工具循环与补正循环共用同一份预算）。默认 5，沿用历史行为。
        self.max_steps = int(os.getenv("VETCLAW_MAX_STEPS", "5"))
        # 引擎开关：react（默认，零新增依赖）| langgraph（可选依赖，见 pyproject 的 [graph] extra）
        # 非法值 fail-fast —— 不静默回落 react，否则 CI 会"假绿"（评测跑的其实是另一套引擎）。
        self.engine = os.getenv("VETCLAW_ENGINE", "react").strip().lower()
        if self.engine not in ("react", "langgraph"):
            raise ValueError(
                f"VETCLAW_ENGINE 非法值：{self.engine!r}（只允许 react | langgraph）"
            )
        # CORS 白名单，逗号分隔。allow_credentials=True 时不能用 "*"，下面自动防御。
        raw_origins = os.getenv(
            "CORS_ORIGINS",
            "http://localhost:5173,http://127.0.0.1:5173",
        )
        self.cors_origins = [o.strip() for o in raw_origins.split(",") if o.strip()]


settings = Settings()
