# SnapPrint

SnapPrint is a local Python desktop application for printed and handwritten text recognition using PaddleOCR-VL, with spoken camera guidance and document playback.

## Requirements

- Windows
- Miniconda or Anaconda
- Python 3.11
- NVIDIA GPU and a CUDA 12.8-compatible driver for the camera app (tested on an RTX 4050 Laptop GPU, 6 GB)

## Installation

### 1. Install Miniconda

If Conda is not already installed, download and install Miniconda:

https://www.anaconda.com/docs/getting-started/miniconda/install

After installation, open **Miniconda Prompt** or **Anaconda Prompt**.

### 2. Clone or Download SnapPrint

Clone the repository:

```bash
git clone <repository-url>
cd SnapPrint
```

Alternatively, download the repository as a ZIP file and extract it.

Then navigate to the extracted SnapPrint directory.

### 3. Create the Conda Environment

Create a new environment using Python 3.11:

```bash
conda create -n snapprint python=3.11
```

Activate it:

```bash
conda activate snapprint
```

### 4. Install Dependencies

Install the required packages:

```bash
python -m pip install --no-compile -r requirements-gpu.txt
python -m pip install --no-compile -r requirements.txt
```

The GPU packages are several gigabytes. First recognition also downloads model weights; subsequent recognition runs locally using cached models.

## Running SnapPrint

Navigate to the SnapPrint directory and make sure the environment is active:

```bash
conda activate snapprint
```

Then start the application:

```bash
python main.py
```

## Running SnapPrint Again

After the initial installation, you only need to open a Conda prompt, navigate to the SnapPrint directory, and run:

```bash
conda activate snapprint
python main.py
```

## Controls

- **Read text / R:** read the recognized document aloud.
- **Space:** stop speech.
- **Scan again / C:** explicitly start the next scan. Scanning stays paused after capture so camera movement cannot interrupt the result.
- **2:** toggle single-page/two-page mode. For two pages, align their divide with the image center; pages are read left then right.

Keep PyCharm's interpreter set to the `snapprint` Conda environment. Model loading can take over a minute on first use; later scans reuse the loaded model.

Run checks with `python -m unittest discover -s tests -v`. Run the simulated-camera UI check with `python tests/ui_smoke.py`.

See [PaddleOCR-VL setup and validation](PADDLEOCR_VL.md) for saved-image testing, memory settings, and limitations.
