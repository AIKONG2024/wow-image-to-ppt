from pathlib import Path

from pydantic import BaseModel


class Settings(BaseModel):
    data_dir: Path = Path(__file__).resolve().parents[2] / "data"
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]

