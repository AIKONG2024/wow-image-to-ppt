[한국어](README.md) | [EN](README_EN.md)

# WOW Image to PPT

A web tool that uploads slide images generated with Duct Tape (GPT-Image2), Nano Banana, or similar image models, separates visual elements into components, and exports them as editable PPTX files.

## Example

Input slide image:

![Input slide](docs/examples/one-pun-input.png)

Detected components:

![Detected components](docs/examples/one-pun-components.png)

Reconstructed editable scene preview:

![Reconstructed scene](docs/examples/one-pun-scene.png)

## How to Use

### 1. Run the app

```powershell
git clone https://github.com/AIKONG2024/wow-image-to-ppt.git
cd wow-image-to-ppt
pip install -r requirements.txt
python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
```

Open this address in your browser.

```text
http://127.0.0.1:8000
```

On Windows, you can also run:

```powershell
.\scripts\start.ps1
```

### 2. Create a PPT from the web page

1. Upload a slide image generated with Duct Tape (GPT-Image2), Nano Banana, or a similar image model.
2. Click `분석 실행`.
3. Check the detected components on the screen.
4. Use `병합`, `분리`, or `제외` if you need to clean up the components.
5. Click `PPTX export` to download the PowerPoint file.

## SAM3 Gives Better Results

SAM3 can split visual elements such as icons, illustrations, charts, and diagrams more accurately. If you have an NVIDIA GPU, SAM3 will also run faster.

### GPU and Hugging Face Setup

1. Check that your NVIDIA GPU driver and CUDA are installed.
2. Create a Hugging Face Access Token.
   - Log in to Hugging Face.
   - Go to `Settings` → `Access Tokens` → `New token`.
   - Choose `Read` permission and copy the generated token.
   - If the SAM3 model page requires approval, request access on the model page first.
3. Run the commands below.

```powershell
cd wow-image-to-ppt
.\scripts\setup-ai-runtime.ps1 -HfToken "YOUR_HUGGING_FACE_TOKEN"
.\scripts\start-ai.ps1
```

To check whether the setup worked, run:

```powershell
.\scripts\check-ai-runtime.ps1
```

## If You Do Not Have SAM3

You can still use the app without SAM3. In this mode, the app uses OpenCV to split components, so the result may be less accurate.

Use it like this:

1. Start the server with the basic run command.
2. Upload an image and run analysis.
3. Check the automatically detected components.
4. Fix incorrect components in the web page with `병합`, `분리`, or `제외`.
5. Export the result as PPTX.

The basic features work without a GPU or Hugging Face token.
