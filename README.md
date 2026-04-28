# Image to PPT

Convert AI-generated slide images into editable PowerPoint files.

Nanobanana, Duct Tape 같은 이미지 생성 모델로 만든 16:9 슬라이드 이미지를 업로드하면 텍스트는 PowerPoint 텍스트박스로, 도형/차트/아이콘/일러스트는 이동 가능한 컴포넌트로 재구성해 PPTX로 내보냅니다.

이 프로젝트의 목표는 원본 이미지를 PPT 위에 통째로 붙이는 것이 아닙니다. 사람이 PPT를 만들 때처럼 `text box`, `shape`, `line`, `picture` 레이어로 분해해서, 결과물이 원본과 비슷하게 보이면서도 수정 가능하도록 만드는 것입니다.

## What It Does

- PaddleOCR로 텍스트를 검출하고 편집 가능한 PPT 텍스트박스로 내보냅니다.
- SAM3가 활성화되어 있으면 아이콘, 그림, 차트, 다이어그램 같은 시각 요소를 더 의미 단위에 가깝게 분리합니다.
- SAM3가 없거나 실패하면 OpenCV fallback으로 기본 컴포넌트를 추출합니다.
- 단순 사각형/프레임은 PPT shape로 재구성하고, 복잡한 차트/일러스트/효과음은 picture 컴포넌트로 유지합니다.
- 텍스트는 항상 이미지/도형보다 위 레이어에 배치해 도형에 가려지지 않게 합니다.
- 브라우저 UI에서 컴포넌트를 드래그 선택하고 `merge`, `split`, `delete`로 보정할 수 있습니다.
- 현재 컴포넌트 그래프를 SVG scene으로 확인하고 PPTX로 export합니다.

## Verified Example

Input slide image:

![Input slide](docs/examples/one-pun-input.png)

Detected components:

![Detected components](docs/examples/one-pun-components.png)

Reconstructed editable scene preview:

![Reconstructed scene](docs/examples/one-pun-scene.png)

Generated files:

- [Example PPTX](docs/examples/one-pun-editable.pptx)
- [Scene SVG](docs/examples/one-pun-scene.svg)
- [Analysis summary](docs/examples/one-pun-analysis-summary.json)

Latest local verification:

- Backend tests: `109 passed`
- PPTX structure: `53` editable text shapes, `13` picture shapes
- Duplicate stylized `BAM` text: `0` editable text shapes, kept as artwork
- Text layer order: all text shapes are above non-text shapes
- Example scene/PPTX regenerated from the checked-in `one-pun` sample

Note: the checked-in example was generated with PaddleOCR active. If SAM3 is not available, visual segmentation uses OpenCV fallback and may require more manual merge/split cleanup.

## Quick Start

```powershell
git clone https://github.com/AIKONG2024/wow-image-to-ppt.git
cd wow-image-to-ppt
pip install -r requirements.txt
python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000
```

Windows shortcut:

```powershell
.\scripts\start.ps1
```

## Optional AI Runtime

For better OCR and semantic image/icon separation, use the local AI setup:

```powershell
cd wow-image-to-ppt
.\scripts\setup-ai-runtime.ps1 -HfToken "YOUR_HUGGING_FACE_TOKEN"
.\scripts\start-ai.ps1
```

Expected runtime:

- Python 3.12+
- PaddleOCR for text detection
- CUDA PyTorch for GPU inference
- SAM3 checkpoint access through Hugging Face

The app still runs without SAM3, but icon/image/chart separation will be less semantic.

## If You Do Not Know How To Set It Up

Ask an AI assistant this:

```text
I want to run this GitHub project on Windows:
https://github.com/AIKONG2024/wow-image-to-ppt

Please give me PowerShell commands to:
1. install Python dependencies,
2. start the FastAPI server,
3. open http://127.0.0.1:8000,
4. optionally enable SAM3 with my Hugging Face token.

Explain each step briefly and tell me how to check whether PaddleOCR and SAM3 are active.
```

## How To Use

1. Create or download a 16:9 slide image from Nanobanana, Duct Tape, or another image model.
2. Open the local web app.
3. Upload the slide image.
4. Click `분석 실행`.
5. Check the component boxes on the canvas.
6. Drag-select multiple components if needed.
7. Use `병합`, `분리 영역 그리기`, `분리 적용`, and `제외` to clean up the structure.
8. Click `SVG scene` to inspect the reconstructed scene.
9. Click `PPTX export` to download the editable PowerPoint.

## Prompt For Slide Image Models

Use prompts like this when generating source slide images:

```text
Create a clean 16:9 infographic slide with clear text, simple rectangular sections,
icons, charts, and strong visual hierarchy. Use high contrast, avoid tiny text,
avoid overlapping labels, and keep each section visually separated so it can be
converted into editable PowerPoint components later.
```

Korean version:

```text
16:9 비율의 인포그래픽 슬라이드 이미지를 만들어줘. 텍스트는 선명하게,
섹션은 사각형 도형 중심으로 구분하고, 아이콘/차트/이미지 요소는 서로 겹치지 않게 배치해줘.
나중에 편집 가능한 PPT 컴포넌트로 분리할 수 있도록 작은 글씨와 복잡한 배경은 피하고,
도형 단위가 명확하게 보이도록 만들어줘.
```

## Current Scope

- OCR text becomes editable PPT text whenever it is detected.
- Simple shapes and frames are reconstructed as PPT shapes when the source region is clean enough.
- Complex charts, illustrations, icons, stylized labels, and photos are exported as movable picture components.
- Large stylized artwork text, such as comic sound effects, is preserved as artwork instead of duplicated as editable OCR text.
- Native chart recreation is not implemented yet.
- Highly stylized image-model slides may still need manual merge/split cleanup before export.

## Development Checks

```powershell
$env:PYTHONPATH='backend'
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
pytest -p no:cacheprovider backend/tests -q
node --check backend/static/app.js
git diff --check
```

The latest local verification produced:

```text
109 passed, 2 warnings
```

Frontend `npm run build` can fail on some locked-down Windows environments with `spawn EPERM` from Vite/esbuild process spawning. In that case, run it from a normal user PowerShell session after dependencies are installed:

```powershell
cd frontend
npm install
npm run build
```

## Architecture

```text
image -> OCR/SAM3/OpenCV components -> component graph -> SVG scene -> PPTX
```

Backend:

- FastAPI
- PaddleOCR
- SAM3 optional runtime
- OpenCV fallback
- python-pptx export

Frontend:

- Browser canvas UI
- Component overlay
- Drag selection
- Merge/split/delete tools
