import importlib.util


def available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def main():
    modules = ["fastapi", "uvicorn", "cv2", "PIL", "pptx", "paddleocr", "sam3", "torch"]
    for module in modules:
        print(f"{module}: {available(module)}")
    if available("torch"):
        import torch

        print(f"torch version: {torch.__version__}")
        print(f"torch cuda available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"torch device: {torch.cuda.get_device_name(0)}")


if __name__ == "__main__":
    main()
