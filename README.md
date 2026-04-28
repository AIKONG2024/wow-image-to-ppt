# Image to PPT

이미지 슬라이드를 웹에서 업로드하면 화면 요소를 컴포넌트로 나누고 편집 가능한 PPTX로 내보내는 웹서비스입니다.

English README: [README_EN.md](README_EN.md)

## Example

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

## 사용 방법

### 1. 기본 실행

```powershell
git clone https://github.com/AIKONG2024/wow-image-to-ppt.git
cd wow-image-to-ppt
pip install -r requirements.txt
python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
```

브라우저에서 아래 주소를 엽니다.

```text
http://127.0.0.1:8000
```

Windows에서는 아래 스크립트로 실행할 수도 있습니다.

```powershell
.\scripts\start.ps1
```

### 2. 웹에서 PPT 만들기

1. 16:9 슬라이드 이미지를 업로드합니다.
2. `분석 실행`을 누릅니다.
3. 화면에 잡힌 컴포넌트를 확인합니다.
4. 필요하면 `병합`, `분리`, `제외`로 컴포넌트를 정리합니다.
5. `PPTX export`를 눌러 PowerPoint 파일을 받습니다.

## SAM3를 이용하면 더 좋은 성능을 냅니다

SAM3를 연결하면 아이콘, 그림, 차트, 다이어그램 같은 시각 요소를 더 잘 나눌 수 있습니다. NVIDIA GPU가 있으면 SAM3 처리 속도도 더 좋아집니다.

### GPU 사용 및 Hugging Face 연결

1. NVIDIA GPU 드라이버와 CUDA가 설치되어 있는지 확인합니다.
2. Hugging Face에서 SAM3 모델에 접근 가능한 토큰을 준비합니다.
3. 아래 명령을 실행합니다.

```powershell
cd wow-image-to-ppt
.\scripts\setup-ai-runtime.ps1 -HfToken "YOUR_HUGGING_FACE_TOKEN"
.\scripts\start-ai.ps1
```

설정이 잘 되었는지 확인하려면 아래 명령을 실행합니다.

```powershell
.\scripts\check-ai-runtime.ps1
```

## SAM3가 없는 사용자

SAM3가 없어도 사용할 수 있습니다. 이 경우 OpenCV 방식으로 컴포넌트를 나누기 때문에 결과가 조금 덜 정확할 수 있습니다.

SAM3 없이 사용할 때는 이렇게 하면 됩니다.

1. 기본 실행 방법으로 서버를 켭니다.
2. 이미지를 업로드하고 분석합니다.
3. 자동으로 나뉜 컴포넌트를 확인합니다.
4. 잘못 나뉜 부분은 웹 화면에서 직접 `병합`, `분리`, `제외`로 정리합니다.
5. PPTX로 내보냅니다.

GPU나 Hugging Face 토큰이 없어도 기본 기능은 실행됩니다.
