import argparse
import importlib
import subprocess
import sys
import time
from pathlib import Path



PACKAGES = ["cv2", "numpy", "sklearn", "joblib", "torch", "torchvision", "paddleocr"]
CODE = ["hog.hogTextGate", "hog.makeDataset", "hog.trainHog", "hog.benchmarkHog"]

IMAGE_FOLDERS = [Path("cnn/datasets/text"), Path("cnn/datasets/noText")]
COCO_TEXT_JSON = Path("cnn/datasets/raw/cocotext.v2.json")
CNN_MODEL = Path("cnn/textCNN.pth")

QUICK_STEPS = [
    ["hog.makeDataset", "--limit", "300"],
    ["hog.trainHog", "--validation-limit", "200"],
    ["hog.benchmarkHog", "--overlays", "10"]
]

FULL_STEPS = [
    ["hog.makeDataset"],
    ["hog.trainHog"],
    ["hog.benchmarkHog", "--compare", "--overlays", "20"]
]



def checkImport(name):
    try:
        importlib.import_module(name)
        print(f"  OK       {name}")
        return True

    except Exception as error:
        print(f"  MISSING  {name}  ({type(error).__name__}: {error})")
        return False



def checkCode():
    print("Packages and code:")

    results = [checkImport(name) for name in PACKAGES + CODE]

    if CNN_MODEL.exists():
        print(f"  OK       {CNN_MODEL}")
    else:
        print(f"  MISSING  {CNN_MODEL} (pull Teodor's branch)")
        results.append(False)

    return all(results)



def checkData():
    print("Dataset:")

    ok = True

    for folder in IMAGE_FOLDERS:
        count = len(list(folder.glob("*.jpg"))) if folder.exists() else 0

        print(f"  {'OK     ' if count else 'MISSING'}  {folder}  ({count} images)")

        ok = ok and count > 0

    if COCO_TEXT_JSON.exists():
        print(f"  OK       {COCO_TEXT_JSON}")
    else:
        print(f"  MISSING  {COCO_TEXT_JSON}")
        ok = False

    return ok



def runStep(step):
    print(f"\n=== python -m {' '.join(step)} ===", flush=True)

    start = time.perf_counter()

    result = subprocess.run([sys.executable, "-m", *step])

    if result.returncode != 0:
        raise SystemExit(f"\nStopped: {step[0]} failed. The error is above.")

    print(f"--- {step[0]} done in {(time.perf_counter() - start) / 60:.1f} min")



def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="only check packages and code")
    parser.add_argument("--quick", action="store_true", help="small test run")
    args = parser.parse_args()

    code_ok = checkCode()

    if args.check:
        print("\nReady. Run this again without --check once the dataset is in place."
              if code_ok else "\nFix the MISSING items above first.")
        return

    if not code_ok or not checkData():
        raise SystemExit("\nFix the MISSING items above first.")

    start = time.perf_counter()

    for step in QUICK_STEPS if args.quick else FULL_STEPS:
        runStep(step)

    print(f"\nAll done in {(time.perf_counter() - start) / 60:.1f} min.")
    print("Results table: results/hog_benchmark.txt, detection images: results/hog_overlays/")



if __name__ == "__main__":
    main()