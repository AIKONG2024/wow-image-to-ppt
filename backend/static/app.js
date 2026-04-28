import React, { useEffect, useMemo, useRef, useState } from 'https://esm.sh/react@19.2.3';
import { createRoot } from 'https://esm.sh/react-dom@19.2.3/client';
import {
  Download,
  EyeOff,
  FileCode2,
  Layers,
  Merge,
  Scissors,
  SplitSquareHorizontal,
  Upload,
  WandSparkles,
} from 'https://esm.sh/lucide-react@0.562.0?deps=react@19.2.3';

const h = React.createElement;
const labels = {
  text: 'Text',
  icon: 'Icon',
  chart: 'Chart',
  diagram: 'Diagram',
  table: 'Table',
  arrow: 'Arrow',
  line: 'Line',
  image: 'Image',
  shape: 'Shape',
  unknown: 'Unknown',
};

function App() {
  const [runtime, setRuntime] = useState(null);
  const [project, setProject] = useState(null);
  const [selected, setSelected] = useState([]);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState('');
  const [splitMode, setSplitMode] = useState(false);
  const [splitBoxes, setSplitBoxes] = useState([]);
  const [draft, setDraft] = useState(null);
  const [selectionDraft, setSelectionDraft] = useState(null);
  const [viewMode, setViewMode] = useState('overlay');
  const stageRef = useRef(null);
  const suppressClickRef = useRef(false);

  useEffect(() => {
    api('/api/runtime').then(setRuntime).catch((error) => setMessage(error.message));
  }, []);

  const visible = useMemo(() => (project?.components ?? []).filter((item) => !item.hidden), [project]);
  const orderedVisible = useMemo(() => [...visible].sort(componentDrawOrder), [visible]);
  const inspectionVisible = useMemo(() => [...visible].sort(componentInspectionOrder), [visible]);

  async function busyRun(callback) {
    setBusy(true);
    setMessage('');
    try {
      await callback();
    } catch (error) {
      setMessage(error.message);
    } finally {
      setBusy(false);
    }
  }

  async function uploadFile(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    await busyRun(async () => {
      const form = new FormData();
      form.append('file', file);
      const next = await api('/api/projects', { method: 'POST', body: form });
      setProject(next);
      setSelected([]);
      setSplitBoxes([]);
      setSelectionDraft(null);
      setMessage('이미지가 업로드되었습니다.');
    });
  }

  async function analyze() {
    if (!project) return;
    await busyRun(async () => {
      const next = await api(`/api/projects/${project.id}/analyze`, { method: 'POST' });
      setProject(next);
      setSelected([]);
      setSelectionDraft(null);
      const noteSuffix = next.analysis_notes?.length ? ` · notes ${next.analysis_notes.length}` : '';
      setMessage(`분석 완료: ${next.components.length}개 컴포넌트${noteSuffix}`);
    });
  }

  async function patch(payload) {
    await busyRun(async () => {
      const next = await api(`/api/projects/${project.id}/components`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      setProject(next);
      setSelected([]);
      setSelectionDraft(null);
      setMessage('컴포넌트 그래프가 갱신되었습니다.');
    });
  }

  async function exportPptx() {
    if (!project) return;
    await busyRun(async () => {
      const response = await fetch(`/api/projects/${project.id}/export/pptx`, { method: 'POST' });
      if (!response.ok) throw new Error(await response.text());
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `${project.id}.pptx`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
      setMessage('PPTX export가 완료되었습니다.');
    });
  }

  function openSceneSvg() {
    if (!project) return;
    window.open(`/api/projects/${project.id}/scene.svg`, '_blank', 'noopener,noreferrer');
  }

  function toggle(id) {
    if (splitMode) return;
    setSelected((current) => (current.includes(id) ? current.filter((item) => item !== id) : [...current, id]));
  }

  function point(event) {
    const rect = stageRef.current.getBoundingClientRect();
    return {
      x: clamp(((event.clientX - rect.left) / rect.width) * project.width, 0, project.width),
      y: clamp(((event.clientY - rect.top) / rect.height) * project.height, 0, project.height),
    };
  }

  function down(event) {
    if (!project) return;
    if (splitMode && selected.length !== 1) return;
    const p = point(event);
    if (splitMode) {
      setDraft({ x: p.x, y: p.y, width: 0, height: 0, originX: p.x, originY: p.y });
      return;
    }
    setSelectionDraft({ x: p.x, y: p.y, width: 0, height: 0, originX: p.x, originY: p.y });
    event.currentTarget.setPointerCapture?.(event.pointerId);
  }

  function move(event) {
    const p = point(event);
    if (draft) {
      setDraft({
        ...draft,
        x: Math.min(draft.originX, p.x),
        y: Math.min(draft.originY, p.y),
        width: Math.abs(p.x - draft.originX),
        height: Math.abs(p.y - draft.originY),
      });
      return;
    }
    if (selectionDraft) {
      setSelectionDraft({
        ...selectionDraft,
        x: Math.min(selectionDraft.originX, p.x),
        y: Math.min(selectionDraft.originY, p.y),
        width: Math.abs(p.x - selectionDraft.originX),
        height: Math.abs(p.y - selectionDraft.originY),
      });
    }
  }

  function up() {
    if (draft) {
      if (draft.width > 4 && draft.height > 4) {
        const { originX, originY, ...box } = draft;
        setSplitBoxes((current) => [...current, roundBox(box)]);
      }
      setDraft(null);
      return;
    }
    if (!selectionDraft) return;
    if (selectionDraft.width > 4 && selectionDraft.height > 4) {
      setSelected(visible.filter((component) => intersects(component.bbox, selectionDraft)).map((component) => component.id));
      suppressClickRef.current = true;
    } else {
      setSelected([]);
      suppressClickRef.current = false;
    }
    setSelectionDraft(null);
  }

  function cancelDrafts() {
    if (draft) {
      const { originX, originY, ...box } = draft;
      if (box.width > 4 && box.height > 4) {
        setSplitBoxes((current) => [...current, roundBox(box)]);
      }
      setDraft(null);
    }
    if (selectionDraft) {
      setSelectionDraft(null);
    }
    suppressClickRef.current = false;
  }

  const samReady = runtime?.sam3?.ready;
  const ocrReady = runtime?.paddleocr?.ready;
  const runtimeIssues = runtime?.issues ?? [];

  return h('main', { className: 'appShell' }, [
    h('header', { className: 'topbar', key: 'topbar' }, [
      h('div', { key: 'title' }, [
        h('h1', { key: 'h1' }, 'PPT Agent Studio'),
        h('p', { key: 'p' }, '한국어 보고서 슬라이드를 editable PPT 컴포넌트로 변환합니다.'),
      ]),
      h('div', { className: 'runtime', key: 'runtime' }, [
        h('span', { className: samReady ? 'ok' : 'warn', key: 'sam', title: runtimeIssues.join('\n') }, `SAM3 ${samReady ? 'ready' : 'fallback'}`),
        h('span', { className: ocrReady ? 'ok' : 'warn', key: 'ocr', title: runtimeIssues.join('\n') }, `PaddleOCR ${ocrReady ? 'ready' : 'missing'}`),
      ]),
    ]),
    h('section', { className: 'workspace', key: 'workspace' }, [
      h('aside', { className: 'sidebar', key: 'sidebar' }, [
        h('label', { className: 'uploadButton', key: 'upload' }, [
          h(Upload, { size: 18, key: 'icon' }),
          h('span', { key: 'span' }, '이미지 업로드'),
          h('input', { key: 'input', type: 'file', accept: 'image/*', onChange: uploadFile }),
        ]),
        actionButton(WandSparkles, '분석 실행', !project || busy, analyze),
        actionButton(Merge, '병합', selected.length < 2 || busy, () => patch({ operation: 'merge', component_ids: selected })),
        actionButton(SplitSquareHorizontal, '분리 영역 그리기', selected.length !== 1 || busy, () => setSplitMode(!splitMode), splitMode),
        actionButton(Scissors, '분리 적용', !splitMode || splitBoxes.length === 0 || busy, () => patch({ operation: 'split', component_ids: selected, boxes: splitBoxes })),
        actionButton(EyeOff, '제외', selected.length === 0 || busy, () => patch({ operation: 'delete', component_ids: selected })),
        actionButton(FileCode2, 'SVG scene', !project || busy, openSceneSvg),
        actionButton(Download, 'PPTX export', !project || busy, exportPptx),
        h('div', { className: 'statusBlock', key: 'status' }, [
          h('strong', { key: 'label' }, '상태'),
          h('span', { key: 'project' }, project?.status ?? '대기 중'),
          h('span', { key: 'mode' }, `analysis: ${runtime?.analysis_mode ?? 'checking'}`),
          h('span', { key: 'count' }, `${visible.length} visible components`),
          h('span', { key: 'selected' }, `${selected.length} selected`),
        ]),
        project?.analysis_notes?.length ? h('div', { className: 'notes', key: 'notes' }, project.analysis_notes.map((note, index) =>
          h('span', { key: index }, note),
        )) : null,
        message ? h('div', { className: 'message', key: 'message' }, message) : null,
        h('div', { className: 'componentList', key: 'components' }, visible.map((component) =>
          h('button', {
            key: component.id,
            className: selected.includes(component.id) ? 'row selected' : 'row',
            onClick: () => toggle(component.id),
          }, [
            h('span', { key: 'type' }, labels[component.type] ?? component.type),
            h('small', { key: 'source' }, component.source),
          ]),
        )),
      ]),
      h('section', { className: 'canvasPane', key: 'canvas' }, project
        ? h('div', { className: 'canvasStack' }, [
          h('div', { className: 'viewToolbar', key: 'toolbar' }, [
            h('div', { className: 'segmented', role: 'group', 'aria-label': 'component view mode', key: 'modes' }, [
              h('button', {
                key: 'overlay',
                type: 'button',
                className: viewMode === 'overlay' ? 'active' : '',
                onClick: () => setViewMode('overlay'),
              }, 'Overlay'),
              h('button', {
                key: 'exploded',
                type: 'button',
                className: viewMode === 'exploded' ? 'active' : '',
                onClick: () => setViewMode('exploded'),
              }, 'Exploded'),
            ]),
            h('span', { className: 'viewCount', key: 'count' }, `${inspectionVisible.length} components`),
          ]),
          viewMode === 'overlay' ? h('div', { className: 'stageWrap', key: 'overlay-view' }, h('div', {
            className: `stage ${splitMode ? 'splitMode' : ''} ${selectionDraft ? 'selecting' : ''}`,
            ref: stageRef,
            onPointerDown: down,
            onPointerMove: move,
            onPointerUp: up,
            onPointerCancel: cancelDrafts,
          }, [
            h('img', { key: 'image', src: `/api/projects/${project.id}/image`, alt: 'uploaded slide', draggable: false }),
            ...orderedVisible.map((component) => h('button', {
              key: component.id,
              type: 'button',
              className: `overlay ${component.type} ${selected.includes(component.id) ? 'selected' : ''}`,
              style: { ...boxStyle(component.bbox, project), zIndex: componentZIndex(component) },
              title: `${labels[component.type] ?? component.type} · ${component.source}`,
              onClick: (event) => {
                event.stopPropagation();
                if (suppressClickRef.current) {
                  suppressClickRef.current = false;
                  return;
                }
                toggle(component.id);
              },
            }, h('span', null, labels[component.type] ?? component.type))),
            ...splitBoxes.map((box, index) => h('div', { key: `split-${index}`, className: 'splitBox', style: boxStyle(box, project) })),
            draft ? h('div', { key: 'draft', className: 'splitBox draft', style: boxStyle(draft, project) }) : null,
            selectionDraft ? h('div', { key: 'selection', className: 'selectionBox', style: boxStyle(selectionDraft, project) }) : null,
          ])) : h('div', { className: 'explodedGrid', key: 'exploded-view' }, inspectionVisible.map((component) =>
            componentCard(component, project, selected.includes(component.id), toggle),
          )),
        ])
        : h('div', { className: 'emptyState' }, [
            h(Layers, { size: 42, key: 'icon' }),
            h('strong', { key: 'strong' }, '이미지를 업로드하세요'),
            h('span', { key: 'span' }, '분석 후 컴포넌트를 병합, 분리, 제외할 수 있습니다.'),
          ])),
    ]),
  ]);
}

function componentCard(component, project, selected, onSelect) {
  const label = labels[component.type] ?? component.type;
  return h('button', {
    key: component.id,
    type: 'button',
    className: `componentCard ${component.type} ${selected ? 'selected' : ''}`,
    onClick: () => onSelect(component.id),
    title: `${label} · ${component.source} · ${component.id}`,
  }, [
    h('div', { className: 'cropViewport', style: componentCropFrameStyle(component), key: 'crop' },
      h('img', {
        src: `/api/projects/${project.id}/image`,
        alt: '',
        draggable: false,
        style: componentCropImageStyle(component, project),
      }),
    ),
    h('div', { className: 'cardMeta', key: 'meta' }, [
      h('div', { className: 'cardTitle', key: 'title' }, [
        h('span', { className: `typePill ${component.type}`, key: 'type' }, label),
        h('small', { key: 'id' }, component.id.slice(-6)),
      ]),
      h('span', { className: 'sourceText', key: 'source' }, component.source),
      h('span', { className: 'bboxText', key: 'bbox' }, bboxLabel(component.bbox)),
    ]),
  ]);
}

function actionButton(Icon, label, disabled, onClick, active = false) {
  return h('button', { key: label, disabled, onClick, className: active ? 'active' : '' }, [
    h(Icon, { size: 18, key: 'icon' }),
    label,
  ]);
}

async function api(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

function boxStyle(bbox, project) {
  return {
    left: `${(bbox.x / project.width) * 100}%`,
    top: `${(bbox.y / project.height) * 100}%`,
    width: `${(bbox.width / project.width) * 100}%`,
    height: `${(bbox.height / project.height) * 100}%`,
  };
}

function componentCropFrameStyle(component) {
  const ratio = clamp(component.bbox.width / Math.max(component.bbox.height, 1), 0.65, 2.4);
  return { aspectRatio: `${ratio}` };
}

function componentCropImageStyle(component, project) {
  const width = Math.max(component.bbox.width, 1);
  const height = Math.max(component.bbox.height, 1);
  return {
    position: 'absolute',
    left: `${-(component.bbox.x / width) * 100}%`,
    top: `${-(component.bbox.y / height) * 100}%`,
    width: `${(project.width / width) * 100}%`,
    height: `${(project.height / height) * 100}%`,
    maxWidth: 'none',
  };
}

function bboxLabel(bbox) {
  return `${Math.round(bbox.x)}, ${Math.round(bbox.y)} · ${Math.round(bbox.width)}×${Math.round(bbox.height)}`;
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function roundBox(box) {
  return {
    x: Math.round(box.x),
    y: Math.round(box.y),
    width: Math.round(box.width),
    height: Math.round(box.height),
  };
}

function componentDrawOrder(left, right) {
  const zDelta = componentZIndex(left) - componentZIndex(right);
  if (zDelta !== 0) return zDelta;
  return componentArea(right) - componentArea(left);
}

function componentInspectionOrder(left, right) {
  const yDelta = left.bbox.y - right.bbox.y;
  if (Math.abs(yDelta) > 8) return yDelta;
  const xDelta = left.bbox.x - right.bbox.x;
  if (Math.abs(xDelta) > 8) return xDelta;
  return componentZIndex(left) - componentZIndex(right);
}

function componentZIndex(component) {
  const layer = {
    shape: 10,
    image: 25,
    chart: 30,
    table: 30,
    diagram: 35,
    unknown: 40,
    line: 50,
    arrow: 55,
    icon: 65,
    text: 75,
  }[component.type] ?? 40;
  return component.source?.startsWith('synthetic-') ? layer - 2 : layer;
}

function componentArea(component) {
  return component.bbox.width * component.bbox.height;
}

function intersects(left, right) {
  return !(
    left.x + left.width < right.x
    || right.x + right.width < left.x
    || left.y + left.height < right.y
    || right.y + right.height < left.y
  );
}

createRoot(document.getElementById('root')).render(h(App));
