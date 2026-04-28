from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image

from .analysis import Analyzer, runtime_status
from .exporter import export_pptx
from .models import ComponentPatch, Project
from .operations import apply_component_patch
from .scene import build_scene_graph, render_scene_svg
from .settings import Settings
from .storage import ProjectStore


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    store = ProjectStore(settings)
    analyzer = Analyzer()

    app = FastAPI(title="PPT Agent Studio", version="0.1.0")
    static_dir = Path(__file__).resolve().parents[1] / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

        @app.get("/")
        def index():
            return FileResponse(static_dir / "index.html")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/runtime")
    def get_runtime():
        return runtime_status()

    @app.post("/api/projects", response_model=Project)
    def create_project(file: UploadFile = File(...)):
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Upload an image file.")
        return store.create_project(file.filename or "slide.png", file.file)

    @app.get("/api/projects/{project_id}", response_model=Project)
    def get_project(project_id: str):
        return _load_or_404(store, project_id)

    @app.get("/api/projects/{project_id}/image")
    def get_project_image(project_id: str):
        project = _load_or_404(store, project_id)
        return FileResponse(project.image_path)

    @app.post("/api/projects/{project_id}/analyze", response_model=Project)
    def analyze_project(project_id: str):
        project = _load_or_404(store, project_id)
        project.status = "analyzing"
        store.save(project)
        try:
            project = analyzer.analyze(project, store)
        except Exception as exc:
            project.status = "error"
            project.error = str(exc)
        store.save(project)
        return project

    @app.patch("/api/projects/{project_id}/components", response_model=Project)
    def patch_components(project_id: str, patch: ComponentPatch):
        project = _load_or_404(store, project_id)
        project = apply_component_patch(project, patch)
        store.save(project)
        return project

    @app.post("/api/projects/{project_id}/export/pptx")
    def export_project(project_id: str):
        project = _load_or_404(store, project_id)
        output_path = export_pptx(project, store)
        return FileResponse(
            output_path,
            media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
            filename=f"{project.id}.pptx",
        )

    @app.get("/api/projects/{project_id}/scene.svg")
    def get_project_scene(project_id: str):
        project = _load_or_404(store, project_id)
        with Image.open(project.image_path) as opened:
            source = opened.convert("RGBA")
        scene = build_scene_graph(project, source)
        svg = render_scene_svg(scene, source)
        store.scene_path(project.id).write_text(svg, encoding="utf-8")
        return Response(content=svg, media_type="image/svg+xml")

    return app


def _load_or_404(store: ProjectStore, project_id: str) -> Project:
    try:
        return store.load(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Project not found.") from exc


app = create_app()
