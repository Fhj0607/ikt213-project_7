"""Generate a Word document from saved OCR text without starting the camera/UI."""
import argparse
from pathlib import Path

from output.document_service import createDocument


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="UTF-8 OCR text files, in page order")
    parser.add_argument("-o", "--output", type=Path, help="Destination .docx file")
    args = parser.parse_args()
    try:
        summary, path = createDocument(
            [source.read_text(encoding="utf-8-sig") for source in args.inputs], args.output)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Document creation failed: {error}\n")
    print(summary["title"])
    print(f"Saved document: {path}")


if __name__ == "__main__":
    main()
