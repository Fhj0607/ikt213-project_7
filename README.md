# SnapPrint

SnapPrint is a Python desktop application for image processing and optical character recognition (OCR).

## Requirements

- Windows
- Miniconda or Anaconda
- Python 3.11

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
pip install numpy opencv-python kivy paddlepaddle paddleocr
```

The installation may take a few minutes.

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