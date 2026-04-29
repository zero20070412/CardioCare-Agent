from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"

# Load local overrides when a developer creates a .env file from .env.example.
load_dotenv(ENV_FILE, override=False)


def _get_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer.") from exc


@dataclass(frozen=True)
class Settings:
    app_env: str = os.getenv("APP_ENV", "dev")
    debug: bool = _get_bool("DEBUG", True)

    use_mock_model: bool = _get_bool("USE_MOCK_MODEL", True)
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "").strip()
    model_name: str = os.getenv("MODEL_NAME", "gpt-4o-mini")
    model_temperature: float = float(os.getenv("MODEL_TEMPERATURE", "0.3"))
    model_max_tokens: int = _get_int("MODEL_MAX_TOKENS", 800)
    model_timeout_seconds: int = _get_int("MODEL_TIMEOUT_SECONDS", 30)

    max_history_rounds: int = _get_int("MAX_HISTORY_ROUNDS", 6)


settings = Settings()
