# Local summarization model

The proof-of-concept model is Qwen3.5 2B in Q4_K_M GGUF format.
The weights live in `.models/llm/qwen3.5-2b/`, separate from application code.
The existing `.gitignore` excludes `.models/`; do not commit the model weights.

For automatic download, checksum verification, Ollama installation/import,
and a summarization smoke test, run `powershell -NoProfile -ExecutionPolicy
Bypass -File .\setup.ps1` from the repository root. See [Windows setup](SETUP.md).

## Download provenance

- Original model: https://huggingface.co/Qwen/Qwen3.5-2B
- GGUF conversion: https://huggingface.co/bartowski/Qwen_Qwen3.5-2B-GGUF
- Conversion revision: `7d26695454df6de5fbcce2e58681e62dae06ce43`
- File: `Qwen_Qwen3.5-2B-Q4_K_M.gguf`
- Size: 1,396,198,496 bytes (approximately 1.40 GB / 1.30 GiB)
- SHA-256: `57a1085840f497d764a7fc5d346922dbde961efb54cc792ea81d694fd846a1d8`
- License: Apache 2.0; see the upstream model license.

Only the language model is downloaded. An image projector is unnecessary because
this stage consumes OCR text, not images. Runtime memory exceeds the file size.

To reproduce the download, run these commands from the repository root in PowerShell:

```powershell
New-Item -ItemType Directory -Force -Path .models/llm/qwen3.5-2b
curl.exe --fail --location --retry 3 --continue-at - --output .models/llm/qwen3.5-2b/Qwen_Qwen3.5-2B-Q4_K_M.gguf https://huggingface.co/bartowski/Qwen_Qwen3.5-2B-GGUF/resolve/7d26695454df6de5fbcce2e58681e62dae06ce43/Qwen_Qwen3.5-2B-Q4_K_M.gguf
Get-FileHash .models/llm/qwen3.5-2b/Qwen_Qwen3.5-2B-Q4_K_M.gguf -Algorithm SHA256
```

## Run with Ollama

Ollama 0.40.15 is installed on this development machine. The downloaded GGUF has
been imported as `snapprint-summary` and passed a short CPU-only summarization
smoke test through `http://localhost:11434/api/chat`. The test used `think: false`,
`num_gpu: 0`, and `keep_alive: "0"` to unload the model afterward. Its response
is saved locally in `.models/llm/smoke-test.json`. This does not establish speed
or accuracy on long OCR sessions or on the minimum target hardware.

Open a new terminal if the `ollama` command is not yet on PATH. On this machine,
the executable is also available at `%LOCALAPPDATA%\Programs\Ollama\ollama.exe`.

To chat with the imported model:

```powershell
ollama run snapprint-summary
```

### Set up on another machine

Install a current version of Ollama with Qwen3.5 support from https://ollama.com/download.
With Ollama running, execute from the repository root:

```powershell
ollama create snapprint-summary -f llm/Modelfile
ollama run snapprint-summary
```

Import follows https://docs.ollama.com/import. Ollama manages its own model store;
importing may consume additional disk space outside this project's download folder.
The Modelfile starts with a modest 4,096-token context to limit memory use.
Long sessions are chunked by the summarizer, as described below.

## Create a document

In the desktop app, collect pages, end the session, then choose **Create Word
document**. Generation runs on the background model worker; session controls
are disabled until it finishes. The app displays the saved path and retains
the editable source notes.
If generation fails, the original text remains available and the button retries.
Files are saved with unique names under `results/documents/`, which is excluded
from Git. This directory contains local OCR text; delete exports manually when
they are no longer needed.

For a demo without loading OCR or opening the camera, save UTF-8 text files and run:

```powershell
python summarize_text.py page1.txt page2.txt -o results/documents/my-notes.docx
```

The template always uses the same A4 page size, Calibri font, heading styles,
and section order: title, Summary, Key points, Detailed notes, Unclear passages,
and Original OCR text. Template labels are English; generated content follows
the source language. The submitted source text is copied directly, with only XML-invalid control
characters removed so Word can open the file. In the desktop app this is the
user's edited notes; raw OCR pages remain separately in session memory.

`llm/summarizer.py` contains the task instructions and JSON schema. Each request
uses temperature 0, a fixed seed, CPU inference, a 4,096-token context, and no
thinking output. Responses are validated before document creation. Output-limit
and malformed-response errors are reported instead of exporting partial notes.
Inputs are split into chunks of at most 2,000 UTF-8 bytes. For long sessions,
chunk summaries and topic sections are combined in order; repeated key points
are deduplicated. This simple approach can repeat topics and does not produce
a single polished global summary. Multi-chunk documents use a generic title.

The app waits for any outstanding OCR job on the serial worker before generating
the document. OCR models currently remain resident, so the complete desktop flow
has not been validated within 8 GB RAM. Ollama unloads the summarizer after each
request. Performance tuning, better global summaries, automatic document opening, cloud
inference, and phone support remain future work.

## Validation

Automatic setup checks local model registration, a short summarization request,
response validation, and Word export. See [Windows setup](SETUP.md). Generated
notes still require human review for factual accuracy.
