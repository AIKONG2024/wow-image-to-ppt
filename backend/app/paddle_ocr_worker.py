from __future__ import annotations

import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

from .analysis import (
    _create_paddle_ocr,
    _ensure_local_ai_cache,
    _paddle_results_to_components,
    _run_paddle_ocr,
)


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python -m app.paddle_ocr_worker <image_path>", file=sys.stderr)
        return 2
    _ensure_local_ai_cache()
    image_path = Path(sys.argv[1])
    with redirect_stdout(sys.stderr):
        ocr = _create_paddle_ocr()
        result = _run_paddle_ocr(ocr, image_path)
    components = _paddle_results_to_components(result)
    print(json.dumps([component.model_dump() for component in components], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
