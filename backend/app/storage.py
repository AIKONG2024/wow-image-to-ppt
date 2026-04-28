from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

from PIL import Image

from .models import Project
from .settings import Settings


class ProjectStore:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.settings.data_dir.mkdir(parents=True, exist_ok=True)

    def create_project(self, filename: str, source_file) -> Project:
        project_id = uuid.uuid4().hex
        project_dir = self.project_dir(project_id)
        project_dir.mkdir(parents=True, exist_ok=True)
        suffix = Path(filename).suffix.lower() or ".png"
        image_path = project_dir / f"source{suffix}"
        with image_path.open("wb") as handle:
            shutil.copyfileobj(source_file, handle)

        with Image.open(image_path) as image:
            width, height = image.size

        project = Project(
            id=project_id,
            image_path=str(image_path),
            width=width,
            height=height,
            status="uploaded",
        )
        self.save(project)
        return project

    def project_dir(self, project_id: str) -> Path:
        return self.settings.data_dir / "projects" / project_id

    def metadata_path(self, project_id: str) -> Path:
        return self.project_dir(project_id) / "project.json"

    def load(self, project_id: str) -> Project:
        path = self.metadata_path(project_id)
        if not path.exists():
            raise FileNotFoundError(project_id)
        return Project.model_validate_json(path.read_text(encoding="utf-8"))

    def save(self, project: Project) -> None:
        path = self.metadata_path(project.id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(project.model_dump_json(indent=2), encoding="utf-8")

    def asset_dir(self, project_id: str) -> Path:
        path = self.project_dir(project_id) / "assets"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def export_dir(self, project_id: str) -> Path:
        path = self.project_dir(project_id) / "exports"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def scene_path(self, project_id: str) -> Path:
        return self.export_dir(project_id) / "scene.svg"
