from __future__ import annotations

import uuid

from .models import BBox, Component, ComponentPatch, Project


def apply_component_patch(project: Project, patch: ComponentPatch) -> Project:
    if patch.operation == "delete":
        return _delete_components(project, patch.component_ids)
    if patch.operation == "merge":
        return _merge_components(project, patch.component_ids)
    if patch.operation == "split":
        return _split_component(project, patch.component_ids, patch.boxes)
    return project


def _delete_components(project: Project, component_ids: list[str]) -> Project:
    selected = set(component_ids)
    for component in project.components:
        if component.id in selected:
            component.hidden = True
    return project


def _merge_components(project: Project, component_ids: list[str]) -> Project:
    selected = [component for component in project.components if component.id in set(component_ids)]
    if len(selected) < 2:
        return project

    for component in selected:
        component.hidden = True

    x1 = min(component.bbox.x for component in selected)
    y1 = min(component.bbox.y for component in selected)
    x2 = max(component.bbox.x + component.bbox.width for component in selected)
    y2 = max(component.bbox.y + component.bbox.height for component in selected)
    text_only = all(component.type == "text" for component in selected)
    merged_type = "text" if text_only else "image"
    merged_text = "\n".join(component.text for component in selected if component.text) if text_only else None
    project.components.append(
        Component(
            id=f"component-{uuid.uuid4().hex[:10]}",
            type=merged_type,
            bbox=BBox(x=x1, y=y1, width=x2 - x1, height=y2 - y1),
            text=merged_text,
            confidence=None,
            source="user-merge",
        )
    )
    return project


def _split_component(project: Project, component_ids: list[str], boxes: list[BBox]) -> Project:
    if not component_ids or not boxes:
        return project
    parent = next((component for component in project.components if component.id == component_ids[0]), None)
    if parent is None:
        return project
    parent.hidden = True
    for box in boxes:
        project.components.append(
            Component(
                id=f"component-{uuid.uuid4().hex[:10]}",
                type=parent.type if parent.type != "text" else "unknown",
                bbox=box,
                confidence=None,
                source="user-split",
            )
        )
    return project
