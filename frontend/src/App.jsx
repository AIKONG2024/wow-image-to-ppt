import { useEffect, useMemo, useRef, useState } from 'react';
import {
  Download,
  EyeOff,
  Layers,
  Merge,
  MousePointer2,
  Scissors,
  SplitSquareHorizontal,
  Upload,
  WandSparkles,
} from 'lucide-react';
import { analyzeProject, exportPptx, getRuntime, patchComponents, uploadProject } from './api.js';

const typeLabels = {
  text: 'Text',
  icon: 'Icon',
  chart: 'Chart',
  diagram: 'Diagram',
  table: 'Table',
  arrow: 'Arrow',
  image: 'Image',
  shape: 'Shape',
  unknown: 'Unknown',
};

export function App() {
  const [runtime, setRuntime] = useState(null);
  const [project, setProject] = useState(null);
  const [selected, setSelected] = useState([]);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState('');
  const [splitMode, setSplitMode] = useState(false);
  const [splitBoxes, setSplitBoxes] = useState([]);
  const [draftBox, setDraftBox] = useState(null);
  const [viewMode, setViewMode] = useState('overlay');
  const stageRef = useRef(null);

  useEffect(() => {
    getRuntime().then(setRuntime).catch((error) => setMessage(error.message));
  }, []);

  const visibleComponents = useMemo(
    () => (project?.components ?? []).filter((component) => !component.hidden),
    [project],
  );
  const inspectionComponents = useMemo(
    () => [...visibleComponents].sort(componentInspectionOrder),
    [visibleComponents],
  );

  const selectedComponents = useMemo(
    () => visibleComponents.filter((component) => selected.includes(component.id)),
    [visibleComponents, selected],
  );

  async function handleUpload(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    await runBusy(async () => {
      const nextProject = await uploadProject(file);
      setProject(nextProject);
      setSelected([]);
      setSplitBoxes([]);
      setMessage('이미지가 업로드되었습니다.');
    });
  }

  async function handleAnalyze() {
    if (!project) return;
    await runBusy(async () => {
      const nextProject = await analyzeProject(project.id);
      setProject(nextProject);
      setSelected([]);
      const noteSuffix = nextProject.analysis_notes?.length ? ` · notes ${nextProject.analysis_notes.length}` : '';
      setMessage(`분석 완료: ${nextProject.components.length}개 컴포넌트${noteSuffix}`);
    });
  }

  async function handleMerge() {
    if (!project || selected.length < 2) return;
    await updateComponents({ operation: 'merge', component_ids: selected });
  }

  async function handleDelete() {
    if (!project || selected.length === 0) return;
    await updateComponents({ operation: 'delete', component_ids: selected });
  }

  async function handleApplySplit() {
    if (!project || selected.length !== 1 || splitBoxes.length === 0) return;
    await updateComponents({ operation: 'split', component_ids: selected, boxes: splitBoxes });
    setSplitMode(false);
    setSplitBoxes([]);
  }

  async function updateComponents(payload) {
    await runBusy(async () => {
      const nextProject = await patchComponents(project.id, payload);
      setProject(nextProject);
      setSelected([]);
      setMessage('컴포넌트 그래프가 갱신되었습니다.');
    });
  }

  async function handleExport() {
    if (!project) return;
    await runBusy(async () => {
      const blob = await exportPptx(project.id);
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

  async function runBusy(callback) {
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

  function toggleSelect(id) {
    if (splitMode) return;
    setSelected((current) =>
      current.includes(id) ? current.filter((item) => item !== id) : [...current, id],
    );
  }

  function stagePoint(event) {
    const rect = stageRef.current.getBoundingClientRect();
    return {
      x: clamp(((event.clientX - rect.left) / rect.width) * project.width, 0, project.width),
      y: clamp(((event.clientY - rect.top) / rect.height) * project.height, 0, project.height),
    };
  }

  function onStagePointerDown(event) {
    if (!splitMode || !project || selected.length !== 1) return;
    const point = stagePoint(event);
    setDraftBox({ x: point.x, y: point.y, width: 0, height: 0, originX: point.x, originY: point.y });
  }

  function onStagePointerMove(event) {
    if (!draftBox || !splitMode) return;
    const point = stagePoint(event);
    setDraftBox({
      ...draftBox,
      x: Math.min(draftBox.originX, point.x),
      y: Math.min(draftBox.originY, point.y),
      width: Math.abs(point.x - draftBox.originX),
      height: Math.abs(point.y - draftBox.originY),
    });
  }

  function onStagePointerUp() {
    if (!draftBox) return;
    if (draftBox.width > 4 && draftBox.height > 4) {
      const { originX, originY, ...box } = draftBox;
      setSplitBoxes((current) => [...current, roundBox(box)]);
    }
    setDraftBox(null);
  }

  const imageUrl = project ? `/api/projects/${project.id}/image` : null;

  return (
    <main className="appShell">
      <header className="topbar">
        <div>
          <h1>PPT Agent Studio</h1>
          <p>한국어 보고서 슬라이드를 editable PPT 컴포넌트로 변환합니다.</p>
        </div>
        <RuntimeStatus runtime={runtime} />
      </header>

      <section className="workspace">
        <aside className="sidebar">
          <label className="uploadButton">
            <Upload size={18} />
            <span>이미지 업로드</span>
            <input type="file" accept="image/*" onChange={handleUpload} />
          </label>

          <button disabled={!project || busy} onClick={handleAnalyze}>
            <WandSparkles size={18} />
            분석 실행
          </button>
          <button disabled={selected.length < 2 || busy} onClick={handleMerge}>
            <Merge size={18} />
            병합
          </button>
          <button
            disabled={selected.length !== 1 || busy}
            className={splitMode ? 'active' : ''}
            onClick={() => setSplitMode((value) => !value)}
          >
            <SplitSquareHorizontal size={18} />
            분리 영역 그리기
          </button>
          <button disabled={!splitMode || splitBoxes.length === 0 || busy} onClick={handleApplySplit}>
            <Scissors size={18} />
            분리 적용
          </button>
          <button disabled={selected.length === 0 || busy} onClick={handleDelete}>
            <EyeOff size={18} />
            제외
          </button>
          <button disabled={!project || busy} onClick={handleExport}>
            <Download size={18} />
            PPTX export
          </button>

          <div className="statusBlock">
            <strong>상태</strong>
            <span>{project?.status ?? '대기 중'}</span>
            <span>analysis: {runtime?.analysis_mode ?? 'checking'}</span>
            <span>{visibleComponents.length} visible components</span>
            <span>{selectedComponents.length} selected</span>
          </div>

          {project?.analysis_notes?.length ? (
            <div className="notes">
              {project.analysis_notes.map((note, index) => (
                <span key={index}>{note}</span>
              ))}
            </div>
          ) : null}

          {message && <div className="message">{message}</div>}

          <ComponentList
            components={visibleComponents}
            selected={selected}
            onSelect={toggleSelect}
          />
        </aside>

        <section className="canvasPane">
          {!project && (
            <div className="emptyState">
              <Layers size={42} />
              <strong>이미지를 업로드하세요</strong>
              <span>분석 후 컴포넌트를 선택해 병합, 분리, 제외할 수 있습니다.</span>
            </div>
          )}
          {project && (
            <div className="canvasStack">
              <div className="viewToolbar">
                <div className="segmented" role="group" aria-label="component view mode">
                  <button
                    type="button"
                    className={viewMode === 'overlay' ? 'active' : ''}
                    onClick={() => setViewMode('overlay')}
                  >
                    Overlay
                  </button>
                  <button
                    type="button"
                    className={viewMode === 'exploded' ? 'active' : ''}
                    onClick={() => setViewMode('exploded')}
                  >
                    Exploded
                  </button>
                </div>
                <span className="viewCount">{inspectionComponents.length} components</span>
              </div>

              {viewMode === 'overlay' ? (
                <div className="stageWrap">
                  <div
                    className={`stage ${splitMode ? 'splitMode' : ''}`}
                    ref={stageRef}
                    onPointerDown={onStagePointerDown}
                    onPointerMove={onStagePointerMove}
                    onPointerUp={onStagePointerUp}
                  >
                    <img src={imageUrl} alt="uploaded slide" draggable={false} />
                    {visibleComponents.map((component) => (
                      <button
                        key={component.id}
                        type="button"
                        className={`overlay ${component.type} ${
                          selected.includes(component.id) ? 'selected' : ''
                        }`}
                        style={boxStyle(component.bbox, project)}
                        onClick={(event) => {
                          event.stopPropagation();
                          toggleSelect(component.id);
                        }}
                        title={`${typeLabels[component.type] ?? component.type} · ${component.source}`}
                      >
                        <span>{typeLabels[component.type] ?? component.type}</span>
                      </button>
                    ))}
                    {splitBoxes.map((box, index) => (
                      <div className="splitBox" style={boxStyle(box, project)} key={`${box.x}-${index}`} />
                    ))}
                    {draftBox && <div className="splitBox draft" style={boxStyle(draftBox, project)} />}
                  </div>
                </div>
              ) : (
                <div className="explodedGrid">
                  {inspectionComponents.map((component) => (
                    <ComponentCard
                      key={component.id}
                      component={component}
                      project={project}
                      selected={selected.includes(component.id)}
                      imageUrl={imageUrl}
                      onSelect={toggleSelect}
                    />
                  ))}
                </div>
              )}
            </div>
          )}
        </section>
      </section>
    </main>
  );
}

function ComponentCard({ component, project, selected, imageUrl, onSelect }) {
  const label = typeLabels[component.type] ?? component.type;
  return (
    <button
      type="button"
      className={`componentCard ${component.type} ${selected ? 'selected' : ''}`}
      onClick={() => onSelect(component.id)}
      title={`${label} · ${component.source} · ${component.id}`}
    >
      <div className="cropViewport" style={componentCropFrameStyle(component)}>
        <img src={imageUrl} alt="" draggable={false} style={componentCropImageStyle(component, project)} />
      </div>
      <div className="cardMeta">
        <div className="cardTitle">
          <span className={`typePill ${component.type}`}>{label}</span>
          <small>{component.id.slice(-6)}</small>
        </div>
        <span className="sourceText">{component.source}</span>
        <span className="bboxText">{bboxLabel(component.bbox)}</span>
      </div>
    </button>
  );
}

function RuntimeStatus({ runtime }) {
  const samReady = runtime?.sam3?.ready;
  const ocrReady = runtime?.paddleocr?.ready;
  const issues = runtime?.issues?.join('\n') ?? '';
  return (
    <div className="runtime">
      <span className={samReady ? 'ok' : 'warn'} title={issues}>SAM3 {samReady ? 'ready' : 'fallback'}</span>
      <span className={ocrReady ? 'ok' : 'warn'} title={issues}>PaddleOCR {ocrReady ? 'ready' : 'missing'}</span>
    </div>
  );
}

function ComponentList({ components, selected, onSelect }) {
  return (
    <div className="componentList">
      <div className="listHeader">
        <MousePointer2 size={15} />
        <span>컴포넌트</span>
      </div>
      {components.map((component) => (
        <button
          key={component.id}
          className={selected.includes(component.id) ? 'row selected' : 'row'}
          onClick={() => onSelect(component.id)}
        >
          <span>{typeLabels[component.type] ?? component.type}</span>
          <small>{component.source}</small>
        </button>
      ))}
    </div>
  );
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
