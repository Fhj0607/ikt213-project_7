Speech feedback (Windows)
=========================

Requires pywin32 in the app environment (`pip install pywin32`).
Uses the installed Windows default SAPI voice with no network service.

Run `python main.py` in the snapprint environment. Quality warnings cover
brightness, blur, contrast, and stability, with an eight-second cooldown.
Warnings do not prevent the existing preprocessing and capture behavior.
After PaddleOCR-VL completes, the controller announces whether readable text
was found. Read text (R) speaks the actual document. No confidence percentage
is announced because VL does not provide the previous OCR line scores.

Space stops speech. R reads the latest document. Scan again (C) explicitly
starts another capture; the app stays paused while the user listens.

Run automated checks with `python -m unittest discover -s tests -v`.
For manual testing, cover the lens, point at an overexposed scene, move the
camera, and capture a printed page. Check warning timing, the completion
announcement, document reading, stop, and closing the app while it is speaking.
