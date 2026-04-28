from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


ComponentType = Literal[
    "text",
    "icon",
    "chart",
    "diagram",
    "table",
    "arrow",
    "line",
    "image",
    "shape",
    "unknown",
]

PrimitiveKind = Literal["textbox", "shape", "line", "arrow", "picture"]
SceneNodeKind = Literal["rect", "text", "line", "arrow", "image"]


class BBox(BaseModel):
    x: float
    y: float
    width: float
    height: float


class Component(BaseModel):
    id: str
    type: ComponentType = "unknown"
    bbox: BBox
    mask_path: str | None = None
    asset_path: str | None = None
    text: str | None = None
    confidence: float | None = None
    source: str = "unknown"
    hidden: bool = False


class PptPrimitive(BaseModel):
    id: str
    kind: PrimitiveKind
    bbox: BBox
    source_component_id: str | None = None
    source_component_type: ComponentType | None = None
    text: str | None = None
    text_color: str | None = None
    fill_color: str | None = None
    line_color: str | None = None
    line_width: float | None = None
    asset_path: str | None = None
    mask_path: str | None = None
    x1: float | None = None
    y1: float | None = None
    x2: float | None = None
    y2: float | None = None
    z_index: int = 0


class SceneNode(BaseModel):
    id: str
    kind: SceneNodeKind
    bbox: BBox
    source_component_id: str | None = None
    source_component_type: ComponentType | None = None
    text: str | None = None
    text_color: str | None = None
    fill_color: str | None = None
    line_color: str | None = None
    line_width: float | None = None
    asset_path: str | None = None
    mask_path: str | None = None
    x1: float | None = None
    y1: float | None = None
    x2: float | None = None
    y2: float | None = None
    z_index: int = 0
    erase_boxes: list[BBox] = Field(default_factory=list)


class SceneGraph(BaseModel):
    width: int
    height: int
    nodes: list[SceneNode] = Field(default_factory=list)


class Project(BaseModel):
    id: str
    image_path: str
    width: int
    height: int
    status: Literal["uploaded", "analyzing", "analyzed", "error"] = "uploaded"
    components: list[Component] = Field(default_factory=list)
    analysis_notes: list[str] = Field(default_factory=list)
    error: str | None = None


class ComponentPatch(BaseModel):
    operation: Literal["merge", "split", "delete"]
    component_ids: list[str]
    boxes: list[BBox] = Field(default_factory=list)
