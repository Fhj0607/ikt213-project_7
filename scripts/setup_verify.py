"""Verify an installation without opening the UI or accessing the camera."""
import argparse
import importlib
import os
import sys
import tempfile
from pathlib import Path
from urllib.request import urlopen
import json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("KIVY_HOME", str(ROOT / "results/setup/kivy"))
os.environ.setdefault("KIVY_NO_ARGS", "1")


def verify(args):
    for module in ("cv2", "numpy", "torch", "torchvision", "paddle", "paddleocr",
                   "paddlex", "transformers", "kivy", "win32com.client", "docx",
                   "sklearn", "joblib"):
        importlib.import_module(module)
        print(f"Import OK: {module}", flush=True)

    import torch
    os.environ["SNAPPRINT_OCR_DEVICE"] = "gpu:0" if args.device == "cuda" else "cpu"
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Check the NVIDIA driver or rerun setup with -TorchDevice cpu.")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("This GPU cannot run the app's bfloat16 OCR settings. Rerun setup with -TorchDevice cpu.")
        # Trigger a small allocation/operation to catch driver or architecture issues.
        result = (torch.ones((2, 2), device="cuda", dtype=torch.bfloat16) @
                  torch.ones((2, 2), device="cuda", dtype=torch.bfloat16))
        assert result.cpu().float().sum().item() == 8
        print(f"CUDA OK: {torch.cuda.get_device_name(0)}", flush=True)

    from cnn.textClassifier import CNNTextGate
    cnn = CNNTextGate()
    try:
        cnn.load()
        print("CNN weights loaded.", flush=True)
    finally:
        cnn.close()

    if (ROOT / "hog/models/hog_svm.joblib").is_file():
        from hog.hogTextGate import HogTextGate
        hog = HogTextGate()
        try:
            hog.load()
            print("HOG weights loaded.", flush=True)
        finally:
            hog.close()

    if args.preload_ocr:
        from ocr.model_service import ModelService
        from ocr.vl_reader import VLReader
        models = ModelService()
        models.reader = VLReader(device="gpu:0" if args.device == "cuda" else "cpu")
        try:
            models.load(lambda message: print(message, flush=True))
            print("OCR detector and recognition models loaded.", flush=True)
        finally:
            models.close()
            models.worker.shutdown(wait=True)

    if args.check_llm:
        with urlopen("http://localhost:11434/api/tags", timeout=10) as response:
            names = {model["name"] for model in json.load(response)["models"]}
        if not names.intersection({"snapprint-summary", "snapprint-summary:latest"}):
            raise RuntimeError("Ollama does not contain snapprint-summary. Rerun setup without -SkipLlm.")
        print("Ollama model registered.", flush=True)

    from output.document import writeDocument
    pages = ["The SnapPrint project meeting is on Monday at 10:00. Bring the printed notes."]
    summary = {
        "title": "Setup verification", "summary": "The meeting is on Monday at 10:00.",
        "key_points": ["Bring printed notes."], "sections": [], "unclear_passages": [],
    }
    if args.smoke_llm:
        from llm.summarizer import LocalSummarizer
        summary = LocalSummarizer().summarize(pages)
        print("Qwen summarization request passed schema validation.", flush=True)
    with tempfile.TemporaryDirectory(prefix="snapprint-setup-") as folder:
        from docx import Document
        path = writeDocument(summary, pages, Path(folder) / "verification.docx")
        paragraphs = [paragraph.text for paragraph in Document(path).paragraphs]
        if pages[0] not in paragraphs:
            raise RuntimeError("Word export did not preserve the test source.")
    print("Word export and source preservation verified.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--preload-ocr", action="store_true")
    parser.add_argument("--check-llm", action="store_true")
    parser.add_argument("--smoke-llm", action="store_true")
    arguments = parser.parse_args()
    try:
        verify(arguments)
    except Exception as error:
        print(f"Setup verification failed: {error}", file=sys.stderr)
        sys.exit(1)
