# PPT Agent Studio

PPT Agent Studio is a local web app for converting Korean report/PPT slide images into editable PowerPoint decks.

## Run

```powershell
cd C:\Users\ust21\Documents\ppt-agent-studio
python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`.

## Use

1. Upload a slide image.
2. Click `분석 실행`.
3. Use drag selection or the component list to select components.
4. Use `병합`, `분리 영역 그리기` / `분리 적용`, and `제외` to revise the component graph.
5. Click `SVG scene` to inspect the intermediate editable scene graph. Rectangles render as SVG rects, text as SVG text, and visual components as embedded cropped PNG images.
6. Click `PPTX export` to generate the editable PowerPoint deck from the same scene graph.

The export path is intentionally `image -> components -> scene graph -> PPTX`. The SVG scene is a preview/debug layer for the reconstructed PowerPoint structure, not a whole-slide bitmap trace.

Shortcut:

```powershell
C:\Users\ust21\Documents\ppt-agent-studio\scripts\start.ps1
```

An optional Vite React development app is also included. On this machine, npm/esbuild process execution may be blocked by Windows permissions; the FastAPI-served React UI above is the verified local path. The verified UI loads React modules from `esm.sh`, so the browser needs internet access unless the frontend is later bundled locally.

```powershell
cd C:\Users\ust21\Documents\ppt-agent-studio\frontend
npm install
npm run dev
```

Open `http://127.0.0.1:5173` for the Vite dev app when available.

## Current Runtime

The app exposes `/api/runtime` and falls back to OpenCV segmentation if SAM3 or PaddleOCR are not installed.

For the intended local AI runtime, use the setup script. It creates two local environments: `.venv-ai` for FastAPI/SAM3/PyTorch GPU and `.venv-ocr` for PaddleOCR/Paddle CPU so the CUDA DLL stacks do not collide in one Python process.

```powershell
cd C:\Users\ust21\Documents\ppt-agent-studio
.\scripts\setup-ai-runtime.ps1 -HfToken "YOUR_HUGGING_FACE_TOKEN"
.\scripts\start-ai.ps1
```

Then open `http://127.0.0.1:8000`. The runtime badges should change from `SAM3 fallback` and `PaddleOCR missing` to ready states after the AI packages, CUDA PyTorch, and checkpoint access are active.

You can re-check the runtime without starting the server:

```powershell
.\scripts\check-ai-runtime.ps1 -HfToken "YOUR_HUGGING_FACE_TOKEN"
```

The AI environment expects:

- Python 3.12+
- CUDA-enabled PyTorch
- PaddleOCR
- SAM3 and its checkpoint access

The UI and export flow remain usable without those optional AI packages, but OCR and SAM quality will be limited until they are installed. SAM3 checkpoint access requires accepted Hugging Face access for the SAM3 model repository and an authenticated token.
