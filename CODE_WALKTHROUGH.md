# MNIST Digit Recognition: Code Walkthrough

A snippet-by-snippet explanation of how every file in this project works, how data flows through it, and why each piece is written the way it is.

*Generated 2026-10-14 from the code in this repository (branch `main`, base commit `c2c22009` plus the uncommitted web-page work). Section analysis was done by parallel subagents, one per code area, and every snippet shown is copied from the real files.*

---

## 1. The project at a glance

The project does two things with the MNIST handwritten-digit dataset:

1. **Training side (`main.py`):** reads the raw MNIST files, trains a small neural network (a multi-layer perceptron, MLP), saves it to `artifacts/mnist_model.keras`, and shows one random test digit with the model's prediction.
2. **Serving side (`app.py` + `templates/index.html`):** a local-only Flask web page where you draw a digit with the mouse; the browser converts the drawing into MNIST format and the server runs the saved model on it.

### Files

| File | Role | Covered in |
|---|---|---|
| `main.py` | Entry point: load data, train once, save, predict a sample | [main.py](#mainpy) |
| `src/data.py` | Parse MNIST IDX binary files into NumPy arrays | [src/data.py](#srcdatapy) |
| `src/logger.py` | Configure file logging at import time | [src/logger.py](#srcloggerpy) |
| `src/Model.py` | Build and train the Keras network | [src/Model.py](#srcmodelpy) |
| `src/exception_h.py` | `CustomException` that records file name and line number | [src/exception_h.py](#srcexception_hpy) |
| `src/plot.py` | Helper to plot a grid of sample digits (currently unused) | [src/plot.py](#srcplotpy) |
| `app.py` | Flask back end: serves the page and `POST /predict` | [app.py](#apppy) |
| `templates/index.html` | The drawing page: HTML, CSS and JavaScript | [templates/index.html](#templatesindexhtml) |
| `test_app.py` | pytest tests for the Flask app | [test_app.py](#test_apppy) |
| `requirement.txt` | `numpy`, `matplotlib`, `tensorflow`, `flask` (pinned versions) | section 3 below |
| `artifacts/mnist_model.keras` | The trained model (created by `main.py`, git-ignored) | |
| `mnist_data/` | The four MNIST IDX files (git-ignored) | |
| `logs/` | One timestamped log file per run (git-ignored) | |

### Quick commands

```bash
python main.py                 # train (first run only), save model, show one prediction
python app.py                  # start the drawing page at http://127.0.0.1:5000
pytest test_app.py             # run the 7 tests (needs the trained model and mnist_data/)
```

---

## 2. How everything fits together

### 2.1 Training workflow (`python main.py`)

```
mnist_data/*.idx*-ubyte  (4 binary files)
        |
        |  src/data.py : read_images_labels()
        |    parse big-endian header, check magic numbers 2049 / 2051,
        |    uint8 bytes -> reshape (N, 784) -> float32 -> / 255.0
        v
X_train (60000, 784) float32 [0,1]    y_train (60000,) uint8 digits 0-9
X_test  (10000, 784) float32 [0,1]    y_test  (10000,) uint8
        |
        |  artifacts/mnist_model.keras already exists?
        |        |-- yes --> skip training
        |        '-- no ---> src/Model.py
        |                     create_model():       784 -> Dense128 relu -> Dense64 relu -> Dense10 softmax
        |                     compile_and_train():  sparse categorical cross-entropy, Adam(0.001), 10 epochs
        |                     evaluate on test set  -> prints "Test accuracy: 0.97xx"
        |                     model.save(...)       -> artifacts/mnist_model.keras
        v
load_and_predict():  load model from disk -> random test image -> predict -> matplotlib window
```

### 2.2 Serving workflow (`python app.py`, then the browser)

```
 BROWSER (templates/index.html)                          SERVER (app.py)
 ------------------------------                          -----------------------------
 draw white strokes on 280x280 black canvas
 click Predict
   toMnist():
     1 find bounding box of the ink
     2 scale longer side to 20 px, paste into 28x28
     3 centre of mass
     4 shift so centre of mass is at (13.5, 13.5)
     5 flatten -> 784 floats in [0,1]
   fetch POST /predict  {"pixels":[784 floats]}  ----->   request.get_json(silent=True)
                                                           np.asarray(float32), shape must be (784,)
                                                           model.predict(reshape(1,784))   (model loaded once)
   show "Prediction: 7 (98.3%)"  <-----------------------  {"digit": 7, "confidence": 0.983}
```

### 2.3 Why the browser does the preprocessing

The model was trained on MNIST images: 28x28, white digit on black, the digit scaled to fit a 20x20 box and positioned by its centre of mass. A raw drawing from a 280x280 canvas would be a different size, position and stroke thickness, so a plain MLP would see something far from its training data. `toMnist()` reproduces the MNIST preparation steps natively in the browser (the canvas does the resizing), so the server stays a thin "validate, predict, reply" layer.

### 2.4 Key ideas used throughout

| Term | Meaning in this project |
|---|---|
| **IDX file** | MNIST's raw binary format: big-endian integer header (magic number, counts, rows, cols) followed by one byte per pixel or label |
| **Normalisation (`/ 255.0`)** | Scales pixel bytes 0-255 to floats 0.0-1.0; training and the web page must both do this |
| **MLP** | The network: stacked fully connected (`Dense`) layers, 109,386 trainable parameters |
| **ReLU** | Hidden-layer activation `max(0, x)` |
| **Softmax** | Output activation turning 10 scores into 10 probabilities that sum to 1; the largest is the predicted digit |
| **Sparse categorical cross-entropy** | Loss that takes integer labels (5) instead of one-hot vectors |
| **Epoch** | One full pass over the training data (batch size 32 by default, so 1,875 weight updates per epoch) |
| **Confidence** | The winning softmax probability; not a calibrated certainty |
| **`.keras` file** | Keras's native single-file format holding architecture and weights |

---

## 3. Configuration files

**`requirement.txt`** pins the four runtime dependencies:

```
numpy==2.3.2
matplotlib==3.10.5
tensorflow==2.20.0
flask==3.1.3
```

`pytest` is deliberately not listed; it is only needed to run `test_app.py`.

**`.gitignore`** is the standard GitHub Python template. The entries that matter to this project: `logs/`, `mnist_data/`, `artifacts` (so the model and dataset are never committed), and, under the comment "File under work in progress", `app.py`, `templates/` and `index.html` (so the web page files are currently ignored by git). It also still contains leftover merge-conflict markers (`<<<<<<< HEAD` ... `>>>>>>> origin/main`) around the `enveg` / `enve` lines.

---

# Part A: Source modules


## src/data.py

`src/data.py` turns the four raw MNIST IDX binary files (train images, train labels, test images, test labels) into NumPy arrays that the Keras MLP can consume directly. Images are flattened to 784-element rows and scaled to the range [0, 1]; labels stay as raw digit values 0-9. It also holds the default location of the dataset (`mnist_data/` at the repo root).

Used by: `main.py` (line 9 `from src.data import load_data_mnist_format`; called at line 43 as `load_data_mnist_format()` with no arguments, so all defaults and the full dataset are used).

### Imports (lines 1-5)

**Code**

```python
import numpy as np
import struct
from pathlib import Path
from src.logger import logging
from src.exception_h import CustomException
```

**What it does**

- `numpy` is used for `frombuffer`, `reshape`, `astype`.
- `struct` is the stdlib module used to decode the big-endian integer header of the IDX files.
- `Path` is used to build the default dataset directory.
- `from src.logger import logging` imports the `logging` module *as it is exposed by `src/logger.py`* (that module does `import logging` and calls `basicConfig`, so importing it here triggers the logger's import-time side effects; see the logger section).
- `CustomException` is imported from `src.exception_h`.

**Workflow / data flow**

Import time only: importing `src.data` first imports `src.logger` (which configures logging and creates `logs/`), then `src.exception_h`.

**Gotchas / notes**

- `CustomException` is imported but never referenced anywhere in `data.py`. Errors in this module are raised as plain `ValueError` (see below), not `CustomException`.
- The imports are `src.`-prefixed, so the module must be imported from a working directory/`sys.path` where the `src` package is resolvable (e.g. running `python main.py` from the repo root).

### `read_images_labels(images_filepath, labels_filepath)` (lines 8-32)

**Code**

```python
def read_images_labels(images_filepath, labels_filepath):
    """
    Reads MNIST images and labels from binary files.
    Returns:
        images: np.ndarray, shape (num_samples, 784)
        labels: np.ndarray, shape (num_samples,)
    """
    logging.info(f"Reading labels from {labels_filepath}")
    with open(labels_filepath, 'rb') as file:
        magic, size = struct.unpack('>II', file.read(8))
        if magic != 2049:
            raise ValueError(f"Labels file magic number mismatch: {magic}")
        labels = np.frombuffer(file.read(), dtype=np.uint8)

    logging.info(f"Reading images from {images_filepath}")
    with open(images_filepath, 'rb') as file:
        magic, size, rows, cols = struct.unpack('>IIII', file.read(16))
        if magic != 2051:
            raise ValueError(f"Images file magic number mismatch: {magic}")
        images = np.frombuffer(file.read(), dtype=np.uint8).reshape(size, rows * cols).astype(np.float32) / 255.0
    if len(images) != len(labels):
        raise ValueError(f"Image/label count mismatch: {len(images)} vs {len(labels)}")

    logging.info(f"Loaded {images.shape[0]} images of size {rows}x{cols}")
    return images, labels
```

**What it does**

Step by step:

1. Line 15: logs (INFO) which labels file is being read.
2. Line 16: opens the labels file in binary read mode (`'rb'`); the `with` block closes it automatically.
3. Line 17: `file.read(8)` reads the first 8 bytes. `struct.unpack('>II', ...)` decodes them as two unsigned 32-bit integers, `>` meaning big-endian. These are the *magic number* and the *item count* (`size`).
4. Lines 18-19: the magic number must be `2049` (hex `0x00000801`: data type byte `0x08` = unsigned byte, dimension byte `0x01` = 1-D array, i.e. a vector of labels). Otherwise a `ValueError` is raised.
5. Line 20: `file.read()` reads *everything remaining* in the file (the label bytes) and `np.frombuffer(..., dtype=np.uint8)` wraps those bytes as a 1-D `uint8` array with no copy and no parsing, since each label is exactly one byte.
6. Line 22: logs the images file path.
7. Line 24: reads 16 header bytes and unpacks four big-endian uint32 values: `magic, size, rows, cols`.
8. Lines 25-26: the magic number must be `2051` (hex `0x00000803`: type `0x08` = unsigned byte, dimension byte `0x03` = 3-D array: count x rows x cols). Otherwise `ValueError`.
9. Line 27, a single chained expression, evaluated left to right:
   - `file.read()` returns the remaining pixel bytes;
   - `np.frombuffer(..., dtype=np.uint8)` makes a flat 1-D `uint8` array of `size*rows*cols` elements;
   - `.reshape(size, rows * cols)` turns it into a 2-D array with one row per image (28*28 = 784 columns for MNIST);
   - `.astype(np.float32)` converts `uint8` (0-255) to `float32` (this allocates a new array, 4 bytes per pixel instead of 1);
   - `/ 255.0` divides every element, scaling pixel intensities into [0.0, 1.0] (result stays `float32`, because a Python float scalar does not upcast a float32 array).
10. Lines 28-29: sanity check that the number of images equals the number of labels; `ValueError` if not.
11. Line 31: logs how many images were loaded and their `rows x cols`.
12. Line 32: returns `(images, labels)`.

**Workflow / data flow**

```
labels_filepath ──> 8-byte header (>II: magic=2049, count) ──> N bytes ──> labels  (N,)        uint8, values 0-9
images_filepath ──> 16-byte header (>IIII: magic=2051, count, rows, cols) ──> N*rows*cols bytes
                    ──> flat uint8 ──reshape──> (N, 784) uint8 ──astype──> float32 ──/255.0──> images (N, 784) float32 in [0,1]
```

For the standard MNIST training set this yields `images (60000, 784) float32` and `labels (60000,) uint8`; for the test set `images (10000, 784) float32` and `labels (10000,) uint8`. (The code itself does not hard-code 60000/10000; those are the counts stored in the standard files' headers.)

IDX file layout (all multi-byte integers big-endian, followed immediately by raw data, no padding):

| File type | Bytes 0-3 | Bytes 4-7 | Bytes 8-11 | Bytes 12-15 | Data from |
|---|---|---|---|---|---|
| Labels (`idx1-ubyte`) | magic `2049` | count | - | - | byte 8, one `uint8` per label |
| Images (`idx3-ubyte`) | magic `2051` | count | rows (28) | cols (28) | byte 16, one `uint8` per pixel, row-major, image after image |

**Gotchas / notes**

- `size` read from the labels header (line 17) is never used or compared with `len(labels)`; only the image-count vs label-count check on line 28 exists. For the images file, `size` *is* used in `reshape`, so a truncated/oversized file whose byte count is not `size*rows*cols` makes `reshape` raise a NumPy `ValueError`.
- The two `ValueError`s are plain `ValueError`, not `CustomException`. In `main.py` they would be caught by the generic `except Exception` (line 75) and re-raised wrapped as `CustomException`.
- `np.frombuffer` over `file.read()` produces a *read-only* array backed by the bytes object. `labels` therefore is read-only (`labels.flags.writeable` is False); `images` is a fresh writable array because `.astype` copies.
- The `labels` array stays `uint8`; in `main.py` it is compared directly with `np.argmax` output (line 54) and passed to training as-is.
- Peak memory while loading the training images: the 47,040,000-byte `uint8` buffer plus the ~188 MB `float32` array (60000 x 784 x 4 bytes), plus the temporary result of the division step, since `astype` and `/ 255.0` each create a new array.
- Magic-number checks validate the file *kind* only; they do not check `rows == 28` and `cols == 28`, although downstream code (`main.py`) assumes 784 features.
- The function logs file paths and counts via the root logger; nothing is printed to the console.

### `DATA_DIR` constant (line 34)

**Code**

```python
DATA_DIR = Path(__file__).resolve().parent.parent / "mnist_data"
```

**What it does**

- `__file__` is the path of `src/data.py`.
- `.resolve()` makes it absolute (and resolves symlinks).
- `.parent` is the `src/` directory; the second `.parent` is the repo root.
- `/ "mnist_data"` appends the folder name using `Path`'s `/` operator, giving `<repo root>/mnist_data` (for this project, `C:\Users\05she\Mnist\mnist_data`).

**Workflow / data flow**

Evaluated once at import time; produces a `pathlib.Path`. It is used immediately afterwards as the base for the default arguments of `load_data_mnist_format`.

**Gotchas / notes**

- Anchored to the location of the source file, **not** the current working directory, so data loading works regardless of where Python is launched from (unlike `src/logger.py`, which uses `os.getcwd()`).
- It does not check that the directory exists; a missing folder only surfaces later as `FileNotFoundError` from `open(...)`.
- Defined between two functions (line 34, after `read_images_labels`), with a single blank line before it, rather than with the imports.

### `load_data_mnist_format(...)` (lines 36-57)

**Code**

```python
def load_data_mnist_format(
    train_images_path=DATA_DIR / "train-images.idx3-ubyte",
    train_labels_path=DATA_DIR / "train-labels.idx1-ubyte",
    test_images_path=DATA_DIR / "t10k-images.idx3-ubyte",
    test_labels_path=DATA_DIR / "t10k-labels.idx1-ubyte",
    num_train=None,
    num_test=None
):
    """
    Loads MNIST data from binary files, optionally limiting the number of samples.
    Returns:
        (X_train, y_train), (X_test, y_test)
    """
    X_train, y_train = read_images_labels(train_images_path, train_labels_path)
    X_test, y_test = read_images_labels(test_images_path, test_labels_path)
    if num_train:
        X_train, y_train = X_train[:num_train], y_train[:num_train]
    if num_test:
        X_test, y_test = X_test[:num_test], y_test[:num_test]
    logging.info(f"Final train shape: {X_train.shape}, {y_train.shape}")
    logging.info(f"Final test shape: {X_test.shape}, {y_test.shape}")
    return (X_train, y_train), (X_test, y_test)
```

**What it does**

1. Lines 36-43: the signature. The four path parameters default to the standard MNIST file names inside `DATA_DIR` (`train-images.idx3-ubyte`, `train-labels.idx1-ubyte`, `t10k-images.idx3-ubyte`, `t10k-labels.idx1-ubyte`). `num_train` / `num_test` default to `None` (meaning "use everything").
2. Line 49: loads the training set via `read_images_labels` (images file first, labels file second, matching that function's parameter order).
3. Line 50: loads the test set the same way.
4. Lines 51-54: if `num_train` (resp. `num_test`) is truthy, both arrays are sliced to the first N samples (`[:N]`, a NumPy view, no copy).
5. Lines 55-56: logs the final shapes.
6. Line 57: returns a nested tuple `((X_train, y_train), (X_test, y_test))`, the same layout as Keras' `mnist.load_data()`.

**Workflow / data flow**

```
DATA_DIR + 4 filenames ──> read_images_labels (train) ──> X_train (60000,784) float32, y_train (60000,) uint8
                       ──> read_images_labels (test)  ──> X_test  (10000,784) float32, y_test  (10000,) uint8
optional num_train/num_test ──> first-N slice of each pair
return ((X_train, y_train), (X_test, y_test))
```

At the call site (`main.py` line 43) the result is unpacked as `(X_train, y_train), (X_test, y_test) = load_data_mnist_format()`, printed (lines 44-45), then `X_train, y_train` go to `compile_and_train(...)` (line 50), `X_test` to `model.predict` (line 52), `y_test` to the accuracy computation (line 54), and `X_test, y_test` to `load_and_predict` (line 70).

**Gotchas / notes**

- Default argument values are evaluated once, when the function is defined (at import time), so the default paths are fixed `Path` objects built from `DATA_DIR`.
- `if num_train:` is a truthiness test: `0` is treated the same as `None` (no limiting), and negative values would slice from the end (`[:-5]`). It is not validated.
- The full train and test sets are always read and converted before any slicing, so `num_train`/`num_test` save no loading time or memory, only reduce what is returned (and a slice view keeps the whole base array alive).
- `main.py` never passes `num_train`/`num_test` and never overrides paths.
- No shuffling or train/validation split happens here.

## src/logger.py

`src/logger.py` configures Python's standard `logging` module at import time so that every module that logs through the root logger writes to a new timestamped file inside a `logs/` directory. It defines no functions or classes; its entire job is its module-level side effects. Other modules use it by doing `from src.logger import logging` and then calling `logging.info(...)`, `logging.error(...)`.

Used by: `main.py` (line 4, `from src.logger import logging`; uses `logging.info` at lines 21, 40, 59 and `logging.error` at line 76) and `src/data.py` (line 4; `logging.info` at lines 15, 22, 31, 55, 56). `main.py` imports it before `src.data`, and `src.data` imports it again, which is a no-op because the module is already cached in `sys.modules`.

### Imports and header comment (lines 1-5)

**Code**

```python
# check the documentaion and work of logger 

import logging 
import os 
from datetime import datetime
```

**What it does**

- Line 1 is a leftover to-do comment (contains the typo "documentaion"); it has no effect.
- `logging` is the stdlib logging framework; `os` provides `getcwd`, `makedirs`, `path.join`; `datetime` supplies the current time.

**Workflow / data flow**

Names `logging`, `os`, `datetime` become module attributes. Because `logging` is a module attribute of `src.logger`, `from src.logger import logging` works for consumers and hands them the real stdlib `logging` module.

**Gotchas / notes**

- Importing `logging` this way is not special: consumers receive the same global `logging` module they would get from `import logging`; the only reason they import it from `src.logger` is to guarantee this module's configuration code has run first.

### Log file name (line 10)

**Code**

```python
LOG_File=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
```

**What it does**

`datetime.now()` gets the current local time at import; `strftime('%m_%d_%Y_%H_%M_%S')` formats it as month_day_year_hour_minute_second, e.g. `10_14_2026_09_30_05`; `.log` is appended. Result is a string such as `10_14_2026_09_30_05.log`.

**Workflow / data flow**

import time -> `str` file name (not a full path). Used on line 16.

**Gotchas / notes**

- One new log file per process run (per import), named with one-second resolution; two runs started within the same second would target the same file name (the `FileHandler` opens in append mode by default, so they would share it).
- Month-first ordering means file names do not sort chronologically across years or even across months in plain alphabetical order.
- The constant is named `LOG_File` (mixed case), unlike `LOG_FILE_PATH`.

### Logs directory (lines 11 and 14)

**Code**

```python
logs_dir=os.path.join(os.getcwd(), "logs")


os.makedirs(logs_dir, exist_ok=True)
```

**What it does**

- Line 11: builds `<current working directory>/logs`. `os.getcwd()` is the directory the Python process was *started from*, not the location of this file or the repo.
- Line 14: `os.makedirs(..., exist_ok=True)` creates that directory (and any missing parents) and does nothing if it already exists. The comment on line 8 ("Create a logs directory if it doesn't exist") describes this.

**Workflow / data flow**

import time -> `str` path of the logs folder -> directory created on disk if needed.

**Gotchas / notes**

- Consequence: simply *importing* `src.logger` (directly, or indirectly via `src.data` or `main.py`) creates a `logs/` directory in the **current working directory**. Running `python main.py` from the repo root gives `<repo>/logs/`; running from another directory creates `logs/` there instead.
- It fails (raises `OSError`/`PermissionError`) at import time if the working directory is not writable, which would break every module that imports the logger.
- This is the opposite convention from `DATA_DIR` in `src/data.py`, which is anchored to the source file location and is independent of the CWD.

### Log file path (line 16)

**Code**

```python
LOG_FILE_PATH=os.path.join(logs_dir,LOG_File)
```

**What it does**

Joins the logs directory with the timestamped file name using the OS path separator, giving the full path e.g. `<cwd>\logs\10_14_2026_09_30_05.log`. The comment on line 18 ("Ensure the log file path is correct") is just a note; no check is performed.

**Workflow / data flow**

`logs_dir` + `LOG_File` -> `LOG_FILE_PATH` (`str`), passed as `filename=` to `basicConfig`.

**Gotchas / notes**

- No existence or writability check is made here; any problem surfaces in `basicConfig` when the file is opened.

### `logging.basicConfig(...)` (lines 19-24)

**Code**

```python
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,

    )
```

**What it does**

Configures the **root logger** once:

- `filename=LOG_FILE_PATH`: attaches a `FileHandler` writing to that file (so output goes to the file, not the console; the file is created when the handler is created, i.e. at import, even if nothing is logged).
- `format=...`: each record is written as `[ <timestamp> ] <line number> <logger name> - <LEVEL> - <message>`. `%(asctime)s` uses the default `YYYY-MM-DD HH:MM:SS,mmm` style; `%(lineno)d` is the source line of the logging call; `%(name)s` is the logger name (`root` when `logging.info(...)` is called on the module directly, as this project does); `%(levelname)s` is INFO/ERROR etc.; `%(message)s` is the text.
- `level=logging.INFO`: records at INFO and above (INFO, WARNING, ERROR, CRITICAL) are kept; DEBUG records are dropped.

An example line looks like: `[ 2026-10-14 09:30:05,123 ] 15 root - INFO - Reading labels from ...\mnist_data\train-labels.idx1-ubyte`.

**Workflow / data flow**

Import time: configuration -> root logger. Run time: `logging.info("...")` anywhere in the process -> formatted line appended to `logs/<timestamp>.log`.

**Gotchas / notes**

- `logging.basicConfig` does nothing if the root logger already has handlers (it has no `force=True` here). If another library configured root logging before `src.logger` is imported, this configuration is silently ignored. In `main.py` the logger is imported before TensorFlow, which keeps this module first.
- Only a file handler is configured: nothing from `logging` appears in the terminal.
- `%(lineno)d` is shown without the file/module name, so line numbers from different modules (e.g. `data.py` line 15 vs `main.py` line 21) are ambiguous in the log.
- Because it is the root logger, third-party libraries that log through `logging` at INFO or above will also be written into the same file.
- The blank line and oddly indented `)` inside the call are formatting only.

## How these two modules fit together

`src/logger.py` is a side-effect-only module: the first time anything imports it, it creates `logs/` under the current working directory and points the root logger at a fresh timestamped file. `src/data.py` imports it (line 4) and then emits INFO messages while reading each IDX file and after assembling the final train/test shapes, so a normal `main.py` run leaves a log containing the "Reading labels / Reading images / Loaded N images / Final train shape" trail for the data stage, followed by main's own messages. Note the contrast in path handling: `data.py` finds the dataset relative to its own file (`DATA_DIR`), while `logger.py` writes logs relative to the process's working directory, so datasets are found from anywhere but the `logs/` folder lands wherever the program was launched.

## src/Model.py

`src/Model.py` defines the neural network and the training step. `create_model` builds an untrained Keras `Sequential` multi-layer perceptron that maps a flattened 28x28 image (784 numbers) to 10 class probabilities, and `compile_and_train` attaches a loss/optimizer and fits the model on the training data. Both functions wrap their bodies in `try/except` and convert any failure into a `CustomException`.

**Used by:** `main.py` only (line 10 imports both functions; line 49 calls `create_model(input_dim=784)`, line 50 calls `compile_and_train(model, X_train, y_train, epochs=10)`). Both calls happen only when `artifacts/mnist_model.keras` does not already exist.

### Imports (lines 1-6)

**Code**

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from src.logger import logging
from src.exception_h import CustomException
import sys
```

**What it does**

1. `tensorflow as tf` gives access to `tf.keras.Input`, `tf.keras.losses` and `tf.keras.optimizers`, which are used with the `tf.` prefix further down.
2. `Sequential` and `Dense` are imported directly: `Sequential` is the "stack of layers, one after another" container, `Dense` is a fully connected layer.
3. `logging` comes from `src.logger`, which (on import) configures the root logger to write INFO-level lines into `logs/<timestamp>.log` under the current working directory.
4. `CustomException` is the project's error wrapper, and `sys` is imported only so the module object can be passed to `CustomException(..., sys)`.

**Workflow / data flow**

Import-time only. Importing this module triggers the `src.logger` side effect (creating `logs/` and the log file) and loads TensorFlow, which is the slow part of start-up.

**Gotchas / notes**

- `import sys` is used only as an argument to `CustomException` (it is the module itself, not a value from it).
- `from src.logger import logging` re-exports the standard-library `logging` module (the name `logging` is imported through `src.logger`), so `logging.info` here is the stdlib root logger configured by `src.logger`.

### create_model (lines 8-21)

**Code**

```python
def create_model(input_dim=784):
    try:
        logging.info("Creating Keras Sequential model for multi-class digit recognition")
        model = Sequential([
            tf.keras.Input(shape=(input_dim,)),
            Dense(128, activation='relu', name='layer1'),
            Dense(64, activation='relu', name='layer2'),
            Dense(10, activation='softmax', name='output')  # 10 outputs for digits 0-9
        ], name="mnist_model")
        logging.info("Model created")
        return model
    except Exception as e:
        logging.error(f"Error creating model: {e}")
        raise CustomException(str(e), sys)
```

**What it does**

1. Logs that model construction is starting.
2. Builds a `Sequential` model named `mnist_model` from four items:
   - `tf.keras.Input(shape=(input_dim,))` declares the input as a vector of `input_dim` numbers (784 = 28 x 28 pixels, flattened). It is not a layer with weights; it only fixes the input shape so the layers can be built immediately.
   - `Dense(128, activation='relu', name='layer1')`: 128 neurons, each computing `max(0, w . x + b)` (ReLU) over all 784 inputs.
   - `Dense(64, activation='relu', name='layer2')`: 64 ReLU neurons over the 128 outputs of layer1.
   - `Dense(10, activation='softmax', name='output')`: 10 neurons (one per digit 0-9), with softmax turning the 10 raw scores into probabilities.
3. Logs success and returns the (uncompiled, untrained) model.

Architecture and parameter counts (each Dense layer has `inputs x units` weights plus `units` biases):

| Layer | Output shape | Parameters | Arithmetic |
|---|---|---|---|
| Input | (None, 784) | 0 | none |
| layer1 Dense(128, relu) | (None, 128) | 100,480 | 784 x 128 + 128 = 100,352 + 128 |
| layer2 Dense(64, relu) | (None, 64) | 8,256 | 128 x 64 + 64 = 8,192 + 64 |
| output Dense(10, softmax) | (None, 10) | 650 | 64 x 10 + 10 = 640 + 10 |
| **Total** | | **109,386** | 100,480 + 8,256 + 650 |

All 109,386 parameters are trainable (no frozen or non-trainable layers).

What softmax means: the output vector has 10 non-negative numbers that sum to 1. Entry `k` is the model's estimated probability that the image is the digit `k`. The predicted digit is the index of the largest entry, which is exactly what `np.argmax(preds, axis=1)` does in `main.py`.

**Workflow / data flow**

- Input argument: `input_dim` (int, default 784).
- Output: a `keras.Sequential` model object. Calling it on a batch of shape `(N, 784)` float32 returns shape `(N, 10)` float32 with rows summing to 1.
- Weights are randomly initialised (Keras default Glorot uniform for Dense kernels, zeros for biases); nothing is trained yet.

**Gotchas / notes**

- The `10` in the output layer is hard-coded; only the input size is parameterised.
- Because the final layer already applies softmax, the model emits probabilities, not logits. This matters for the loss setting in `compile_and_train` (`from_logits` must stay `False`, which is the default).
- Error path: if anything inside raises, the original error is logged, then `CustomException(str(e), sys)` is raised from inside the `except` block. The file/line recorded in the resulting message is therefore this file and the line in the `try` body that was executing (line 11 for the `Sequential(...)` call), not the deeper Keras frame where the failure originated (see `src/exception_h.py` below).
- The comment "# 10 outputs for digits 0-9" is on the output layer line; the model is not specific to MNIST apart from that.

### compile_and_train (lines 23-36)

**Code**

```python
def compile_and_train(model, X, y, epochs=30):
    try:
        logging.info("Compiling and training model")
        model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),  # Changed for multi-class
            optimizer=tf.keras.optimizers.Adam(0.001),
            metrics=['accuracy']
        )
        model.fit(X, y, epochs=epochs)
        logging.info("Model training complete")
        return model
    except Exception as e:
        logging.error(f"Error during model training: {e}")
        raise CustomException(str(e), sys)
```

**What it does**

1. Logs the start.
2. `model.compile(...)` configures how learning happens:
   - **Loss: `SparseCategoricalCrossentropy()`**. Cross-entropy measures how far the predicted probability vector is from the right answer (it is `-log(p_true_class)` averaged over the batch). The "Sparse" variant takes labels as plain integers (`y` = `[5, 0, 4, ...]`, shape `(N,)`), which is how `src/data.py` returns them (`uint8` array of shape `(num_samples,)`). The non-sparse `CategoricalCrossentropy` would require one-hot labels (shape `(N, 10)`, e.g. `[0,0,0,0,0,1,0,0,0,0]`), which would need an extra conversion step. Default `from_logits=False` is correct because the output layer applies softmax.
   - **Optimizer: `Adam(0.001)`**. Adam is a gradient-descent variant that keeps running averages of each parameter's gradient and squared gradient to adapt step sizes per parameter. The positional `0.001` is the learning rate (the base step size).
   - **`metrics=['accuracy']`**. Metrics are reported during training but do not affect the weight updates. With a sparse integer-label loss, Keras resolves `'accuracy'` to a sparse categorical accuracy: the fraction of samples whose `argmax` of the output equals the integer label.
3. `model.fit(X, y, epochs=epochs)` trains the model:
   - An **epoch** is one full pass over all training samples. `epochs=30` (default of this function) means 30 passes; `main.py` overrides it to 10.
   - `batch_size` is not passed, so Keras uses its default of **32**: weights are updated after every 32 samples. With the 60,000-image MNIST training set that is ceil(60000 / 32) = 1,875 weight updates per epoch (the training set size depends on the idx files in `mnist_data/`; `num_train` in `load_data_mnist_format` can shrink it).
   - Keras shuffles the training data each epoch by default (`shuffle=True`). No validation data is given, so only training loss and accuracy are printed per epoch.
4. Logs completion and returns the same `model` object (trained in place).

**Workflow / data flow**

- Inputs: `model` (from `create_model`), `X` float32 array `(N, 784)` scaled to 0-1 (done in `src/data.py` by dividing by 255), `y` integer array `(N,)` with values 0-9, `epochs` int.
- Steps: compile -> fit (forward pass, loss, backprop, Adam update, repeated per batch, per epoch).
- Output: the trained model. `main.py` then uses it for `model.predict(X_test)` and `model.save("artifacts/mnist_model.keras")`.

**Gotchas / notes**

- The return value of `model.fit` (the `History` object with per-epoch loss/accuracy) is discarded; the only record of training progress is what Keras prints to the console.
- There is no validation split, early stopping or checkpointing; training always runs the full number of epochs.
- The inline comment `# Changed for multi-class` is a leftover from an earlier version; it does not affect behaviour.
- Default `epochs=30` in the signature differs from the value actually used by the project (10 in `main.py`).
- If `y` were one-hot encoded this loss would fail or give wrong results; if `X` were not 2-D `(N, 784)` the `fit` call would raise and be wrapped in `CustomException`.
- The `CustomException` message reports this file and the line inside the `try` body that was executing (line 26 for a `compile` failure, line 31 for a `fit` failure), not the deeper Keras line where the error originated.

## src/exception_h.py

`src/exception_h.py` provides the project's custom exception type. When code catches a generic exception it can re-raise `CustomException(str(e), sys)`, and the resulting exception's string form includes the script file name, the line number, and the original message. It also imports the project logger, but does not use it.

**Used by:** `main.py` (imports it; catches/re-raises it in `main()` lines 73-77), `src/Model.py` (lines 21 and 36), `src/plot.py` (line 28), and imported by `src/data.py` (line 5; a grep finds no `CustomException(` call there).

### Imports and helper error_mes_deta (lines 1-12)

**Code**

```python
import sys 
from src.logger import logging
# Custom exception handling module

#Custom exception handling documentation on docs.pythopn.org
def error_mes_deta(error,error_details:sys):
   _,_,exc_tb= error_details.exc_info()
   file_name=exc_tb.tb_frame.f_code.co_filename    
   #error_message is a string that contains the file name, line number, and error message
   error_message="error occured in python script name [{0}] line number [{1}] error message [{2}]".format(
       file_name,exc_tb.tb_lineno,str(error))
   return error_message
```

**What it does**

1. `error_details` is expected to be the `sys` module itself (the annotation `:sys` is just a hint; Python does not enforce it).
2. `error_details.exc_info()` is `sys.exc_info()`. Inside an `except` block (or any code running while an exception is being handled) it returns a 3-tuple `(exception_type, exception_instance, traceback_object)` describing the exception currently being handled. The first two items are discarded (`_`), and only the traceback object `exc_tb` is kept.
3. `exc_tb.tb_frame` is the frame object for the code that was executing; `.f_code.co_filename` is the path of the source file for that frame's code. `exc_tb.tb_lineno` is the line number within that frame where execution was when the exception passed through it.
4. These are formatted together with `str(error)` into one string: `error occured in python script name [<file>] line number [<line>] error message [<message>]` (the misspelling "occured" is in the code).

**Workflow / data flow**

- Inputs: `error` (anything with a meaningful `str()`), `error_details` (the `sys` module).
- Output: a single formatted `str`.
- It reads the exception that is "currently being handled" from interpreter state; it does not take the exception as a parameter, only its message.

**Gotchas / notes**

- **Outside an `except` block:** `sys.exc_info()` returns `(None, None, None)`, so `exc_tb` is `None`, and `exc_tb.tb_frame` raises `AttributeError: 'NoneType' object has no attribute 'tb_frame'`. There is no guard in the code, so constructing `CustomException` while no exception is being handled fails with that AttributeError instead of producing a `CustomException`.
- The traceback object's head is the frame in which the exception is currently being handled, so the reported file and line are where the exception passed through the `try` body of the catching function (e.g. the `model.fit(...)` line in `compile_and_train`), not necessarily the deepest frame where it was originally raised (that would require following `tb_next`).
- `from src.logger import logging` is unused in this file; importing it only triggers the logger's setup side effect.
- Comment says "docs.pythopn.org" (typo for python.org); cosmetic only.

### CustomException (lines 15-21)

**Code**

```python
class CustomException(Exception):
   def __init__(self,error_message, error_details:sys):
        super().__init__(error_message)
        self.error_message=error_mes_deta(error_message,error_details=error_details)

   def __str__(self):
       return self.error_message
```

**What it does**

1. Subclasses the built-in `Exception`.
2. `__init__` takes the original message (`error_message`) and the `sys` module (`error_details`).
3. `super().__init__(error_message)` stores the raw original message in `self.args`, so `e.args[0]` is the unformatted message.
4. It then calls `error_mes_deta(error_message, error_details=error_details)` and stores the formatted string as `self.error_message`.
5. `__str__` returns that formatted string, so `str(exc)`, `print(exc)` and the traceback's last line all show the "error occured in python script name [...] line number [...] error message [...]" text.

**Workflow / data flow**

The intended sequence, as used in `src/Model.py`:

1. Something inside the `try:` raises (for example Keras raising `ValueError` in `model.fit`).
2. `except Exception as e:` catches it; the interpreter now has this exception as the "currently handled" one.
3. `logging.error(f"... {e}")` writes the original message to the log file.
4. `raise CustomException(str(e), sys)` constructs the new exception; inside `__init__`, `sys.exc_info()` returns the original exception's traceback, so file name = path of the file containing the `try` block, line = the line in the `try` body that was executing (e.g. line 31 `model.fit(...)`).
5. The formatted message becomes the string form of the exception, e.g.
   `error occured in python script name [C:\Users\05she\Mnist\src\Model.py] line number [31] error message [<original text>]`.
6. The new exception propagates up; Python also attaches the original as `__context__` ("During handling of the above exception, another exception occurred").
7. In `main.py`, the outer `except CustomException: raise` lets it pass through unchanged; a plain `Exception` there is wrapped the same way.

**Gotchas / notes**

- The `error_details` parameter must be the `sys` module (or any object with an `exc_info()` method); every call site passes `sys`.
- `super().__init__(error_message)` receives whatever object was passed. In `src/plot.py` that is the exception object `e` itself rather than `str(e)`, so `self.args[0]` is an exception instance there; the formatted message still works because `error_mes_deta` calls `str(error)`.
- Instantiating `CustomException` outside an `except` block raises `AttributeError` (see above).
- Wrapping is not idempotent: wrapping an existing `CustomException` via `str(e)` would produce a message that contains the previous formatted message. `main.py` avoids this with its `except CustomException: raise` clause.
- The module has no `exc_info` fallback and no use of `logging`, so nothing is logged by this class itself; the callers log before raising.

## src/plot.py

`src/plot.py` contains a single helper, `plot_sample_images`, that shows a square grid of randomly chosen MNIST digits with their labels as titles, which is useful for eyeballing the dataset. It is written in the same try/except-to-`CustomException` style as the rest of the project.

**Used by:** nothing. A repository-wide grep shows `plot_sample_images` has no callers anywhere in the repo (the only matches are its own definition at `src/plot.py` line 8). `main.py` does its own `matplotlib` plotting inside `load_and_predict` rather than calling this module.

### Imports (lines 1-6)

**Code**

```python
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from src.logger import logging
from src.exception_h import CustomException
```

**What it does**

1. `matplotlib.pyplot as plt` creates figures/axes and shows them.
2. `numpy as np` supplies random index generation.
3. `logging` and `CustomException` are the project's logging and error wrapper; `sys` is passed to `CustomException`.

**Workflow / data flow**

Import-time only; importing pulls in matplotlib and triggers the `src.logger` setup.

**Gotchas / notes**

- `import os` is not used anywhere in the file.
- Nothing imports this module, so these imports never execute in the normal `main.py` run.

### plot_sample_images (lines 8-28)

**Code**

```python
def plot_sample_images(X, y, num_images=100, img_shape=(28, 28)):
    """
    Plots a grid of sample images from the dataset.
    """
    try:
        logging.info(f"Plotting {num_images} sample images")
        m = X.shape[0]
        fig, axes = plt.subplots(int(num_images ** 0.5), int(num_images ** 0.5), figsize=(8, 8))
        fig.tight_layout(pad=0.1)
        for i, ax in enumerate(axes.flat):
            if i >= num_images:
                break
            idx = np.random.randint(m)
            ax.imshow(X[idx].reshape(img_shape), cmap='gray')
            ax.set_title(str(y[idx]))
            ax.set_axis_off()
        plt.show()
        logging.info("Sample images plotted")
    except Exception as e:
        logging.error(f"Error plotting images: {e}")
        raise CustomException(e,sys)
```

**What it does**

1. Logs how many images are requested.
2. `m = X.shape[0]` is the number of samples available.
3. Creates a figure with a `side x side` grid of axes where `side = int(num_images ** 0.5)` (the integer part of the square root). Default `num_images=100` gives a 10 x 10 grid at 8 x 8 inches.
4. `fig.tight_layout(pad=0.1)` shrinks the spacing between subplots (it is applied before the images are drawn).
5. Loops over `axes.flat` (the grid flattened into a 1-D sequence of axes). For each cell:
   - picks a random row index with `np.random.randint(m)` (an integer in `[0, m)`);
   - reshapes that flattened row (784 values) to `img_shape` (28 x 28) and draws it in grayscale;
   - uses the label `y[idx]` as the subplot title;
   - hides the axis ticks and frame.
6. `plt.show()` opens the window (blocking in a normal interactive backend) and then the success message is logged.

**Workflow / data flow**

- Inputs: `X` array `(m, 784)` (float32 scaled 0-1 in this project), `y` array `(m,)` of integer labels, `num_images` int, `img_shape` tuple.
- Output: no return value (`None`); the effect is a matplotlib window and log lines.

**Gotchas / notes**

- **Grid size vs. `num_images`:** the grid has `int(sqrt(num_images))**2` cells, which is always less than or equal to `num_images`. For perfect squares (the default 100) every cell is filled. For a non-square value such as 10 the grid is 3 x 3 = 9 cells, so only 9 images are shown, not 10.
- **The `if i >= num_images: break` is effectively unreachable:** since the number of cells never exceeds `num_images`, `i` never reaches `num_images` inside the loop.
- **Random sampling with replacement:** `np.random.randint(m)` is drawn independently per cell, so the same image can appear more than once in a grid; there is also no seed, so each call shows a different set.
- **`num_images=1`:** `plt.subplots(1, 1)` returns a single `Axes` object, not an array, so `axes.flat` raises `AttributeError`, which is caught and re-raised as `CustomException`. Likewise `num_images=0` creates a 0 x 0 grid and fails in matplotlib.
- **`CustomException(e, sys)` passes the exception object, not a string** (the other modules pass `str(e)`). It still works because `error_mes_deta` formats with `str(error)`, but `self.args[0]` ends up being the exception instance.
- Titles are `str(y[idx])`, so `y` must be integer class labels (not one-hot) to give readable titles.
- Mismatched input (e.g. `X` rows not reshaping to `img_shape`) raises `ValueError` inside the try and is reported as a `CustomException`.

## How these modules fit together

`src/exception_h.py` is the shared foundation: every module wraps its work in `try/except Exception as e`, logs through `src.logger`, and re-raises as `CustomException(..., sys)` so that the error text carries the file name and line number. `src/Model.py` is the part that actually matters for the pipeline: `main.py` calls `create_model` to build the 784 -> 128 -> 64 -> 10 network (109,386 parameters) and `compile_and_train` to fit it on the float32 `(N, 784)` images and integer labels produced by `src/data.py`, then evaluates and saves the result to `artifacts/mnist_model.keras`. `src/plot.py` follows the same conventions (logger plus `CustomException`) but is currently dead code: it is not called by `main.py` or any other module.

---

# Part B: Entry point

## main.py

`main.py` is the project's command-line entry point for the Keras side of the project. It loads the MNIST IDX files, trains a small fully connected network only when no saved model exists, saves it to `artifacts/mnist_model.keras`, and then always loads that saved model back from disk to classify one randomly chosen test image and display it with matplotlib. It does not start the Flask drawing page (that is `app.py`). Run with: `python main.py`

### Imports and warnings filter (lines 1-14)

**Code**

```python
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from src.logger import logging
from src.exception_h import CustomException
import os
import sys

from src.data import load_data_mnist_format
from src.Model import create_model, compile_and_train
import numpy as np
from tensorflow.keras.models import load_model
import random
import matplotlib.pyplot as plt
```

**What it does**

1. Lines 1-2 import `warnings` and install a global filter that silences every `FutureWarning`. It is placed first so it is active before the heavier libraries (TensorFlow, NumPy, matplotlib) are imported and have a chance to emit such warnings at import time.
2. Line 4, `from src.logger import logging`, executes `src/logger.py`. As a side effect of that import, `logs/` is created under the current working directory (`os.makedirs(..., exist_ok=True)`) and `logging.basicConfig(...)` is called with `filename=logs/<mm_dd_YYYY_HH_MM_SS>.log` and `level=logging.INFO`. The name `logging` imported here is simply the standard-library `logging` module, re-exported by `logger.py`; all later `logging.info(...)` / `logging.error(...)` calls in `main.py` write to that log file (there is no console handler configured).
3. Line 5 imports `CustomException` (defined in `src/exception_h.py`), used in `main()`'s exception handling.
4. `os` and `sys` are standard library; `sys` is only used to pass the `sys` module to `CustomException(str(e), sys)` on line 77.
5. Lines 9-10 pull in the data loader and the two model helpers from `src/` (see the `src/data.py` and `src/Model.py` sections). `src/Model.py` imports TensorFlow, which is the slowest import in the script.
6. Lines 11-14 import NumPy, Keras `load_model`, the standard-library `random`, and `matplotlib.pyplot`.

**Workflow / data flow**

```
python main.py
   |
   v
warnings filter  -->  src.logger (basicConfig, creates logs/*.log)
   |
   v
src.exception_h, src.data, src.Model (imports TensorFlow)
   |
   v
numpy, keras.load_model, random, matplotlib.pyplot
   |
   v
MODEL_PATH defined, functions defined, __main__ guard calls main()
```

**Gotchas / notes**

- Import order matters: `logging.basicConfig` only configures the root logger if it has no handlers yet. Because `src.logger` is imported before TensorFlow/Keras and the other `src` modules, the log-file configuration is installed first. (Every `src/` module also imports `src.logger`, so whichever is imported first triggers it; in `main.py` that is line 4.)
- The log directory is `os.getcwd()/logs`, so it depends on the directory you launch Python from.
- The `FutureWarning` filter is process-wide, not scoped to one library.

### MODEL_PATH (line 16)

**Code**

```python
MODEL_PATH = os.path.join("artifacts", "mnist_model.keras")
```

**What it does**

Builds the relative path `artifacts\mnist_model.keras` (on Windows; `artifacts/mnist_model.keras` on POSIX) as a module-level constant. It is used in four places: the existence check, `os.path.dirname` for creating the folder, `model.save`, and `load_model` in `load_and_predict`. The `.keras` extension tells Keras to use its native single-file saving format.

**Workflow / data flow**

```
"artifacts" + "mnist_model.keras"  -->  MODEL_PATH
        |-- os.path.exists(MODEL_PATH)           (decides whether to train)
        |-- os.makedirs(dirname(MODEL_PATH))     ("artifacts")
        |-- model.save(MODEL_PATH)
        '-- load_model(MODEL_PATH)               (in load_and_predict)
```

**Gotchas / notes**

- The path is relative to the current working directory, not to the location of `main.py` (contrast with `DATA_DIR` in `src/data.py`, which is anchored to the file's own location via `Path(__file__)`). Running `python main.py` from a different directory would look for / create `artifacts/` there. (The Flask app's `app.py` is documented separately; `main.py` itself uses only the plain relative path.)
- To force retraining you must delete `artifacts/mnist_model.keras` (the check on line 48 is only "does the file exist").

### load_and_predict() (lines 18-35)

**Code**

```python
def load_and_predict(X_test, y_test):
    # Load the saved model
    model = load_model(MODEL_PATH)
    logging.info(f"loaded from the model {MODEL_PATH}")
    # Pick a random test sample
    idx = random.randint(0, len(X_test) - 1)
    sample_image = X_test[idx]
    true_label = y_test[idx]
    # Reshape and predict
    sample_pred = model.predict(sample_image.reshape(1, 784))
    predicted_label = np.argmax(sample_pred, axis=1)[0]
    # Print results
    print(f"True label: {true_label}, Predicted label: {predicted_label}")
    # Show image
    plt.imshow(sample_image.reshape(28, 28), cmap="gray")
    plt.title(f"True: {true_label}, Predicted: {predicted_label}")
    plt.axis("off")
    plt.show()
```

**What it does**

1. `load_model(MODEL_PATH)` reads the saved `.keras` file and rebuilds the `Sequential` model (784 -> Dense 128 relu -> Dense 64 relu -> Dense 10 softmax) with its trained weights. It raises if the file is missing or unreadable. The path is logged at INFO level.
2. `random.randint(0, len(X_test) - 1)` picks a random index. `random.randint(a, b)` is inclusive on both ends, so the `- 1` is required: for `len(X_test) == 10000` the valid indices are 0..9999, and without it index 10000 could be drawn and raise `IndexError`. The generator is not seeded, so a different image is usually chosen each run.
3. `sample_image = X_test[idx]` is one row of shape `(784,)`, dtype `float32`, with pixel values scaled to 0.0-1.0. `true_label = y_test[idx]` is a NumPy `uint8` scalar (0-9).
4. `sample_image.reshape(1, 784)` turns the single vector into a batch of one, shape `(1, 784)`, because Keras `predict` expects a leading batch dimension. `model.predict` returns an array of shape `(1, 10)` of softmax probabilities (and, with Keras' default verbosity, prints a progress bar to the console).
5. `np.argmax(sample_pred, axis=1)` returns, for each row, the index of the largest probability, giving shape `(1,)`; `[0]` extracts the single integer, the predicted digit.
6. `print(...)` writes both labels to stdout.
7. `plt.imshow(sample_image.reshape(28, 28), cmap="gray")` reshapes the flat vector back into a 28x28 grid and draws it in grayscale; `plt.title(...)` puts the true and predicted labels in the title; `plt.axis("off")` hides the axis ticks and frame; `plt.show()` opens the window.
8. `plt.show()` blocks the program until the figure window is closed. With a non-interactive matplotlib backend such as `Agg` it does not open a window and returns immediately (effectively a no-op). The function returns `None`.

**Workflow / data flow**

```
MODEL_PATH --load_model--> model
X_test (10000, 784) --X_test[idx]--> sample_image (784,)  float32 0..1
y_test (10000,)     --y_test[idx]--> true_label           uint8 scalar
sample_image --reshape(1, 784)--> (1, 784) --model.predict--> (1, 10) probabilities
(1, 10) --argmax(axis=1)--> (1,) --[0]--> predicted_label (int)
sample_image --reshape(28, 28)--> imshow --> title / axis off --> plt.show() (blocks)
```

**Gotchas / notes**

- It always loads the model from disk, even when `main()` has just trained and saved it in the same run.
- It does not retrain or evaluate; it classifies exactly one image per call.
- Because `plt.show()` blocks, the script does not exit until you close the image window (on an interactive backend).
- The `FutureWarning` filter and the log file do not capture the printed result; the "True label / Predicted label" line goes to stdout only (the log only records the model load).

### main() (lines 38-77)

**Code**

```python
def main():
    try:
        logging.info("Starting main workflow")

        # Load data (paths default to ./mnist_data next to the repo)
        (X_train, y_train), (X_test, y_test) = load_data_mnist_format()
        print(f"Training data shape: {X_train.shape}, {y_train.shape}")
        print(f"Test data shape: {X_test.shape}, {y_test.shape}")

        # Train only if no saved model exists; delete the file to retrain
        if not os.path.exists(MODEL_PATH):
            model = create_model(input_dim=784)
            model = compile_and_train(model, X_train, y_train, epochs=10)
            # Evaluate the model
            preds = model.predict(X_test)
            pred_classes = np.argmax(preds, axis=1)
            accuracy = np.mean(pred_classes == y_test)
            print(f"Test accuracy: {accuracy:.4f}")
            # Save the model
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            model.save(MODEL_PATH)
            logging.info(f"Model saved to {MODEL_PATH}")

         # Example prediction
        # sample_idx = 123  # Pick a random test example
        # sample_image = X_test[sample_idx]
        # sample_pred = model.predict(sample_image.reshape(1, 784))
        # predicted_digit = np.argmax(sample_pred)
        # true_digit = y_test[sample_idx]
        # print(f"Example prediction - True: {true_digit}, Predicted: {predicted_digit}")

        ## If you have created the model use the below code to load the model and predict
        load_and_predict(X_test, y_test)

        
    except CustomException:
        raise  # already logged and formatted where it was raised
    except Exception as e:
        logging.error(f"Unhandled exception in main workflow: {e}")
        raise CustomException(str(e), sys)
```

**What it does**

1. Logs "Starting main workflow".
2. `load_data_mnist_format()` is called with all defaults: it reads the four IDX files under `<repo>/mnist_data/`, divides pixels by 255 and returns `(X_train, y_train), (X_test, y_test)` with `X_*` as `float32` arrays of shape `(N, 784)` and `y_*` as `uint8` arrays of shape `(N,)`. Concretely `X_train (60000, 784)`, `y_train (60000,)`, `X_test (10000, 784)`, `y_test (10000,)`. The shapes are printed on lines 44-45.
3. `if not os.path.exists(MODEL_PATH)`: the whole training block runs only when the saved model file is absent.
   - `create_model(input_dim=784)` builds the uncompiled `Sequential` model (784 -> 128 -> 64 -> 10, softmax output).
   - `compile_and_train(model, X_train, y_train, epochs=10)` compiles it (sparse categorical cross-entropy, Adam with learning rate 0.001, accuracy metric) and calls `model.fit(X, y, epochs=10)`, returning the same model. `epochs=10` is passed explicitly, overriding that function's default of 30.
   - Evaluation is done by hand: `model.predict(X_test)` gives probabilities of shape `(10000, 10)`; `np.argmax(preds, axis=1)` gives predicted digits of shape `(10000,)`; `pred_classes == y_test` is a boolean array of shape `(10000,)`; `np.mean` of it is the fraction correct, printed to 4 decimals as "Test accuracy".
   - `os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)` creates the `artifacts` folder if needed (no error if it exists), then `model.save(MODEL_PATH)` writes the `.keras` file and the save is logged.
4. The commented-out block (lines 61-67) is not executed. See Gotchas.
5. `load_and_predict(X_test, y_test)` always runs, whether or not training happened (it is outside the `if`). It reloads the model from disk and shows one random test image with its prediction (see the previous section).
6. Exception handling wraps everything above:
   - `except CustomException: raise` comes first so an exception already created and formatted by `create_model` / `compile_and_train` (they log it and wrap it themselves) is re-raised unchanged rather than double-wrapped.
   - `except Exception as e:` catches everything else (for example `FileNotFoundError` or `ValueError` from `src/data.py`, or a failure in `load_model` / `model.save`). It logs the message with `logging.error(...)` and raises `CustomException(str(e), sys)`. `CustomException` uses `sys.exc_info()` to build a message of the form `error occured in python script name [<file>] line number [<n>] error message [<msg>]`; because the traceback it reads starts in `main()`, the file name and line number refer to the statement inside `main()`'s `try` block that was executing (for example the call on line 70), not the deepest frame.

**Workflow / data flow**

```
main()
  |
  v
load_data_mnist_format()
  X_train (60000, 784) float32   y_train (60000,) uint8
  X_test  (10000, 784) float32   y_test  (10000,) uint8
  |
  v
os.path.exists("artifacts/mnist_model.keras") ?
  |                                     |
  | NO (first run)                      | YES (later runs)
  v                                     |
create_model(input_dim=784)             |
  v                                     |
compile_and_train(..., epochs=10)       |
  v                                     |
predict(X_test) -> argmax -> mean       |
  -> print "Test accuracy"              |
  v                                     |
os.makedirs("artifacts"), model.save    |
  |                                     |
  '------------------+------------------'
                     v
        load_and_predict(X_test, y_test)
          load_model -> random test image -> predict -> print
                     v
          plt.show()  (window; blocks until closed)

errors: CustomException -> re-raised as is
        any other Exception -> logging.error -> raise CustomException(str(e), sys)
```

**Gotchas / notes**

- Lines 61-67 are dead code: an older hard-coded example prediction (`sample_idx = 123`) left as comments. Line 61 is also indented by 9 spaces rather than 8, which is harmless because it is a comment. If it were uncommented it would reference `model`, which is only defined inside the `if` block, so it would fail with `NameError` on any run where the model file already exists.
- `epochs=10` overrides the default `epochs=30` of `compile_and_train`.
- The training data (`X_train`, `y_train`) is loaded and its shape printed on every run, including runs that skip training.
- Evaluation happens before saving, so if `predict`/`mean` raised, no model file would be written (and the next run would retrain). It is only run on the first run; later runs print no accuracy.
- The evaluation is a manual `predict` + `argmax` + `mean`, not `model.evaluate`, and it uses the test set only; no validation split is used in training (`model.fit` is called with just `X, y, epochs`).
- Retraining requires deleting `artifacts/mnist_model.keras`; there is no command-line flag for it.
- The trained in-memory model is discarded in favour of the one reloaded from disk inside `load_and_predict`.
- `CustomException` is only raised by `create_model` / `compile_and_train`; errors from data loading surface through the generic branch and are wrapped there.
- Because `CustomException.__str__` returns the formatted message, the traceback shown to the user ends with that formatted text; the original exception remains attached as the chained context.

### `__main__` guard (lines 79-80)

**Code**

```python
if __name__ == "__main__":
    main()
```

**What it does**

When the file is executed directly (`python main.py`), `__name__` is `"__main__"` and `main()` is called. When the module is imported instead, the imports, `MODEL_PATH` and the function definitions still execute (so the logger configuration and the TensorFlow import still happen), but `main()` is not run.

**Workflow / data flow**

```
python main.py  -> __name__ == "__main__" -> main()
import main      -> definitions only, main() not called
```

**Gotchas / notes**

- `main()` returns `None`; it does not call `sys.exit`. An unhandled `CustomException` propagates out of `main()` and ends the script with a traceback (non-zero exit status).
- Because `from src...` imports are used, run it from the repo root (the script's own directory is placed on `sys.path` by `python main.py`).

### Execution timeline of `python main.py`

**First run (no `artifacts/mnist_model.keras`)**

1. Warnings filter set; `src.logger` import creates `logs/<timestamp>.log` and configures logging; TensorFlow, NumPy, matplotlib are imported.
2. `main()` starts, logs "Starting main workflow".
3. MNIST IDX files are read; train/test shapes printed: `(60000, 784), (60000,)` and `(10000, 784), (10000,)`.
4. The model file does not exist, so `create_model` builds the network.
5. `compile_and_train` compiles and trains for 10 epochs (Keras prints per-epoch progress).
6. `model.predict(X_test)` -> argmax -> accuracy printed as `Test accuracy: 0.xxxx`.
7. `artifacts/` is created if needed and the model is saved to `artifacts/mnist_model.keras`; the save is logged.
8. `load_and_predict` reloads the model from disk, picks a random test image, prints `True label: ..., Predicted label: ...`.
9. The matplotlib window opens; the script ends when the window is closed.

**Later runs (model file present)**

1. Same imports and logger setup; `main()` starts.
2. Data is loaded and shapes printed (training data is loaded but not used).
3. `os.path.exists(MODEL_PATH)` is true, so create / train / evaluate / save are all skipped; no accuracy is printed.
4. `load_and_predict` loads the saved model, picks a random test image, prints the labels and shows the image; the script ends when the window is closed.

---

# Part C: Web page (Flask back end, tests, browser front end)

## app.py

`app.py` is the whole web back end: a 34-line Flask program that loads the trained Keras model once, serves the drawing page at `/`, and exposes one JSON endpoint, `POST /predict`, that takes 784 pixel values and returns the predicted digit and the model's confidence. It does no image processing itself; the browser (`templates/index.html`) already scales and centres the drawing into a 28x28 MNIST-style image and flattens it to 784 numbers. The server only validates the shape, runs the model and reports the result.

Run with: `python app.py` then open <http://127.0.0.1:5000>.

(The trained model must exist first: `artifacts/mnist_model.keras`, produced by `python main.py`.)

### Imports (lines 1-5)

**Code**

```python
from pathlib import Path

import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
```

**What it does**

- `Path` (stdlib `pathlib`) is used to build a file path that works on Windows and Linux alike.
- `numpy` converts the incoming JSON list into a float array the model can consume.
- From Flask: `Flask` is the application class, `render_template` renders a file from the `templates/` folder, and `request` is the proxy object for the HTTP request currently being handled.
- `load_model` reads a saved `.keras` file (architecture + weights) back into a ready-to-use Keras model. Importing `tensorflow.keras` is the slow part of starting the app (TensorFlow is a big import).

**Workflow / data flow**

Nothing runs per request here; these names are just made available to the rest of the module. `Path` is used in lines 7-9, `np` in the `/predict` route, `render_template` in `/`, `request` in `/predict`, `load_model` on line 12.

**Gotchas / notes**

- Import order matters only in that `tensorflow` is imported at module import time, so even `from app import app` (as `test_app.py` does) pays the TensorFlow start-up cost.

### MODEL_PATH and the missing-model guard (lines 7-9)

**Code**

```python
MODEL_PATH = Path(__file__).resolve().parent / "artifacts" / "mnist_model.keras"
if not MODEL_PATH.exists():
    raise SystemExit(f"{MODEL_PATH} not found - run `python main.py` first to train it")
```

**What it does**

1. `__file__` is the path of `app.py` itself. `Path(__file__).resolve()` turns it into an absolute path (following symlinks), and `.parent` is the folder containing it, i.e. the project root. This makes the model location independent of the directory you launched Python from.
2. The `/` operator on a `Path` joins path pieces, so `MODEL_PATH` is `<project root>/artifacts/mnist_model.keras`.
3. If that file does not exist, `raise SystemExit("...")` stops the process immediately. When `SystemExit` carries a string, Python prints the string to stderr and exits with status 1, so you get a one-line instruction ("run `python main.py` first to train it") instead of a long TensorFlow/Keras traceback.

**Workflow / data flow**

At start-up: compute path -> check existence -> either exit with the message or continue to model loading. No request can ever be served in the "missing" case because the process never gets as far as `app.run`.

**Gotchas / notes**

- The guard also fires when `app.py` is merely imported (e.g. by pytest), because it is module-level code, not inside `if __name__ == "__main__":`. A pytest run without the trained model therefore aborts with `SystemExit` during test collection.
- `artifacts` is listed in `.gitignore` (line 147), so the model file is never in git; each clone must train its own.

### Creating the app and loading the model (lines 11-12)

**Code**

```python
app = Flask(__name__)
model = load_model(str(MODEL_PATH))  # once at startup, not per request
```

**What it does**

1. `Flask(__name__)` creates the application object. Passing `__name__` tells Flask where this module lives so it can find the `templates/` folder next to it.
2. `load_model(str(MODEL_PATH))` reads the `.keras` file from disk and returns a compiled-and-ready Keras model. `str()` converts the `Path` to a plain string, which is the safest input type for older Keras versions.
3. The result is stored in a module-level global named `model`, so every request handler reuses the same in-memory model. The comment states the intent: loading takes seconds, so doing it per request would make every prediction slow.

**Workflow / data flow**

Runs exactly once, when the module is first imported (either `python app.py` or `from app import app`). After that, `predict()` just reads the global `model`.

**Gotchas / notes**

- Because `model` is created at import time, importing `app` loads TensorFlow and the model, which takes a few seconds and needs `artifacts/mnist_model.keras` on disk.
- The model is shared by all requests; there is no locking in the code. The Flask development server handles requests in threads by default, but this app has only one simple read-only use of the model per request.

### GET / route (lines 15-17)

**Code**

```python
@app.get("/")
def index():
    return render_template("index.html")
```

**What it does**

- `@app.get("/")` is a decorator (Flask 2.0+ shortcut for `@app.route("/", methods=["GET"])`). It registers `index()` as the handler for HTTP GET requests to the URL `/`. Other methods (e.g. POST) to `/` get a 405 Method Not Allowed from Flask.
- `render_template("index.html")` loads `templates/index.html`, runs it through Jinja2 (the page has no template variables, so it is returned as-is) and returns the HTML string, which Flask wraps in a 200 response with `Content-Type: text/html`.

**Workflow / data flow**

Browser opens <http://127.0.0.1:5000/> -> Flask matches `GET /` -> `index()` returns the HTML -> the browser renders the 280x280 black canvas, "Predict" and "Clear" buttons, and runs the page's inline JavaScript.

**Gotchas / notes**

- `templates/` is in `.gitignore` (line 11), and so is `app.py` (line 10) under the comment "File under work in progress". `git ls-files` shows neither `app.py` nor `templates/index.html` is tracked, so a fresh clone would not contain them.

### POST /predict route (lines 20-30)

**Code**

```python
@app.post("/predict")
def predict():
    pixels = (request.get_json(silent=True) or {}).get("pixels")
    try:
        x = np.asarray(pixels, dtype="float32")
    except (TypeError, ValueError):
        x = None
    if x is None or x.shape != (784,):
        return {"error": "pixels must be a list of 784 numbers"}, 400
    probs = model.predict(x.reshape(1, 784), verbose=0)[0]
    return {"digit": int(probs.argmax()), "confidence": float(probs.max())}
```

**What it does**

Step by step:

1. **Decorator.** `@app.post("/predict")` registers `predict()` for HTTP POST on `/predict` only (shortcut for `methods=["POST"]`). A GET to `/predict` gets 405.
2. **Parse the body (line 22).** `request.get_json(silent=True)` parses the request body as JSON. With `silent=True` it does not raise on failure: it returns `None` if the body is not valid JSON, is empty, or the request's `Content-Type` is not `application/json`. `or {}` replaces any falsy result (`None`, `{}`, `[]`, `""`, `0`, `false`, `null`) with an empty dict, and `.get("pixels")` then returns the value of the `pixels` key or `None` if it is absent.
3. **Convert to an array (lines 23-26).** `np.asarray(pixels, dtype="float32")` builds a NumPy array of 32-bit floats, the dtype Keras models use by default. If the value cannot be converted (for example the string `"abc"`, or a ragged nested list), NumPy raises `ValueError` (or `TypeError` for unsupported objects such as a dict); the `except` catches those and sets `x = None`.
4. **Validate (lines 27-28).** `x.shape != (784,)` is the validation. A flat list of exactly 784 numbers becomes a 1-D array of shape `(784,)`. Anything else has a different shape: 783 or 785 items give `(783,)`/`(785,)`; `None` becomes a 0-d array `array(nan)` with shape `()`; a nested list such as 28 rows of 28 would be `(28, 28)`. On failure the function returns the tuple `({"error": ...}, 400)`.
5. **Predict (line 29).** `x.reshape(1, 784)` turns the 1-D vector into a batch of one sample, because Keras `predict` expects shape `(batch, features)`. `model.predict(..., verbose=0)` runs the network without printing a progress bar and returns an array of shape `(1, 10)`. `[0]` takes the first (only) row: a vector of 10 class probabilities.
6. **Build the response (line 30).** For a softmax output layer the 10 values are non-negative and sum to 1. `probs.argmax()` is the index of the largest one, which is the predicted digit (index = digit class 0-9). `probs.max()` is that largest probability, used as the confidence. `int(...)` and `float(...)` convert NumPy scalar types to plain Python types so Flask can serialise them. Returning a `dict` makes Flask JSON-encode it (via `jsonify`) with status 200; returning `(dict, 400)` does the same but with status 400.

**Workflow / data flow**

Request (sent by the page's `fetch`, with header `Content-Type: application/json`):

```json
{"pixels": [0.0, 0.0, 0.0, 0.0, 0.0, "... 784 floats in total, each 0.0-1.0 ...", 0.0]}
```

Response on success (HTTP 200):

```json
{"digit": 7, "confidence": 0.98}
```

Response on bad input (HTTP 400):

```json
{"error": "pixels must be a list of 784 numbers"}
```

Validation cases:

| Body sent | `pixels` after parsing | `np.asarray` result | Outcome |
|---|---|---|---|
| `{"pixels": [784 numbers]}` | list, 784 items | shape `(784,)` | 200 + digit/confidence |
| `{"pixels": [783 numbers]}` | list, 783 items | shape `(783,)` | 400 |
| `{"pixels": [785 numbers]}` | list, 785 items | shape `(785,)` | 400 |
| `{"pixels": "abc"}` | string | `ValueError` -> `x = None` | 400 |
| `{"pixels": null}` | `None` | 0-d `nan` array, shape `()` | 400 |
| `{}` (key missing) | `None` | 0-d `nan` array, shape `()` | 400 |
| not JSON / wrong `Content-Type` | `get_json` gives `None` -> `{}` -> `None` | 0-d `nan` array | 400 |

What the browser does with the reply (`templates/index.html`, not documented here): if `r.ok` it shows `Prediction: <digit> (<confidence*100, 1 decimal>%)`, otherwise it shows `j.error`.

**Gotchas / notes**

- **Non-object JSON bodies cause HTTP 500.** `(request.get_json(silent=True) or {})` only substitutes `{}` for *falsy* parsed values. If the body is valid JSON but a non-empty list (`[1,2]`), a non-empty string (`"x"`), or a truthy number/boolean (`5`, `true`), the expression is `[1,2].get("pixels")`, which raises `AttributeError` (the `try` only wraps `np.asarray`, not this line). Flask turns the unhandled exception into a 500 Internal Server Error, not a 400. Falsy bodies (`[]`, `""`, `0`, `false`, `null`) give 400.
- **Only the shape is validated, not the values.** NaN, Infinity (Python's JSON parser accepts the literals `NaN`/`Infinity`) and out-of-range numbers (e.g. `-5` or `1e30`; the page itself sends 0-1) all pass the check and go to the model. A numeric string list (`["0", "1", ...]` x 784) is also accepted because NumPy converts numeric strings to float32. `true`/`false` elements become 1.0/0.0.
- The error message always says "list of 784 numbers" regardless of which check failed.
- `confidence` is the model's softmax probability for the winning class; it is not a calibrated certainty, and a blank all-zero canvas still yields some digit with some confidence (the tests only check it is in range).
- No `app.testing`/`debug` flag is set in the code.

### The `__main__` block (lines 33-34)

**Code**

```python
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)  # local-only, debug off
```

**What it does**

- `if __name__ == "__main__":` is true only when the file is executed directly (`python app.py`), and false when it is imported (as `test_app.py` does), so importing never starts a server.
- `app.run(...)` starts Flask's built-in Werkzeug development server. `host="127.0.0.1"` binds it to the loopback interface only, so only the same machine can connect; other devices on the network cannot reach it (use `0.0.0.0` to expose it, which this code deliberately does not). `port=5000` is the TCP port, giving <http://127.0.0.1:5000>.
- `debug` is not passed, so it defaults to off: no interactive debugger and no auto-reloader (the comment says so).

**Workflow / data flow**

`python app.py` -> TensorFlow imported, model loaded (lines 1-12) -> server starts listening on 127.0.0.1:5000 and blocks, handling requests until Ctrl+C.

**Gotchas / notes**

- This is the development server, not a production WSGI server.
- With debug off, code changes require a manual restart.
- If port 5000 is already in use, `app.run` fails with an OSError at start.

---

## test_app.py

`test_app.py` is a pytest file that exercises `app.py` through Flask's built-in test client, so no real server, port or network is involved. It checks four things: the index page serves the canvas, a valid blank image gets a well-formed answer, four kinds of malformed payload are rejected with 400, and the model actually predicts correct digits for real MNIST test images.

Run with: `pytest test_app.py` (from the project root; needs `artifacts/mnist_model.keras` and the `mnist_data/` files).

### Imports (lines 1-5)

**Code**

```python
import numpy as np
import pytest

from app import app
from src.data import load_data_mnist_format
```

**What it does**

- `pytest` provides the fixture and parametrize decorators. (`numpy as np` is imported but not used anywhere in this file.)
- `from app import app` imports the Flask application object from `app.py`. Importing the module runs all of its top-level code: it imports TensorFlow, checks that the model file exists, and loads the model (see the `app.py` sections above).
- `from src.data import load_data_mnist_format` brings in the project's MNIST IDX-file loader (documented elsewhere); it returns `(X_train, y_train), (X_test, y_test)`, with images as float32 arrays of shape `(N, 784)` scaled to 0-1 and labels as uint8.

**Workflow / data flow**

pytest collects the file -> executes these imports -> `app.py` module code runs (several seconds for TensorFlow + model load) -> then tests run.

**Gotchas / notes**

- Importing `app` at collection time means the whole test file needs the trained `artifacts/mnist_model.keras`; if it is missing, `app.py` raises `SystemExit` and collection aborts.
- The test file sits in the project root so that `app` and `src` are importable as top-level packages.
- `test_app.py` is untracked in git (git status shows `?? test_app.py`).

### The `client` fixture (lines 8-10)

**Code**

```python
@pytest.fixture(scope="module")
def client():
    return app.test_client()
```

**What it does**

- `@pytest.fixture` marks a function whose return value is injected into any test that has a parameter with the same name (`client`).
- `scope="module"` means pytest creates the fixture once per test module and reuses it for every test in the file, instead of once per test (the default `"function"` scope).
- `app.test_client()` returns Flask's test client: an object with `.get()`, `.post()` etc. that calls the WSGI application in-process. No socket is opened and no server needs to be running. Responses expose `.status_code`, `.data` (bytes) and `.json` (parsed body).

**Workflow / data flow**

First test requests `client` -> pytest calls `client()` once -> every later test in the module receives the same object.

**Gotchas / notes**

- The model itself is not part of the fixture; it was already loaded when `app` was imported, so it is shared by all tests regardless of fixture scope.
- `app.testing` is not set, so an unhandled exception inside a route shows up in tests as a 500 response rather than as a raised exception.

### test_index_serves_drawing_canvas (lines 13-16)

**Code**

```python
def test_index_serves_drawing_canvas(client):
    r = client.get("/")
    assert r.status_code == 200
    assert b"<canvas" in r.data
```

**What it does**

1. Sends `GET /` through the test client.
2. Asserts the status is 200.
3. Asserts the raw response body (bytes, hence the `b"..."`) contains the text `<canvas`, i.e. the page that was rendered actually includes the drawing canvas element (`templates/index.html` has `<canvas id="pad" ...>`).

**Workflow / data flow**

test client -> `index()` -> `render_template("index.html")` -> HTML bytes -> assertions. No model is used.

**Gotchas / notes**

- It checks only that the tag text is present; it does not run any JavaScript or check that drawing works.
- It would fail if `templates/index.html` were missing (Flask would raise `TemplateNotFound`, giving a 500).

### test_predict_returns_digit_and_confidence (lines 19-23)

**Code**

```python
def test_predict_returns_digit_and_confidence(client):
    r = client.post("/predict", json={"pixels": [0.0] * 784})
    assert r.status_code == 200
    assert r.json["digit"] in range(10)
    assert 0.0 <= r.json["confidence"] <= 1.0
```

**What it does**

1. `[0.0] * 784` builds a list of 784 zeros (a completely black image).
2. `client.post(..., json=...)` serialises the dict to JSON and sets `Content-Type: application/json` automatically.
3. Asserts status 200, that `digit` is an integer 0-9 (`in range(10)`), and that `confidence` is a probability between 0 and 1.

**Workflow / data flow**

Request body `{"pixels": [0.0, 0.0, ... 784 times]}` -> shape `(784,)` passes validation -> `model.predict` -> e.g. `{"digit": 3, "confidence": 0.31}` (the actual values depend on the model) -> assertions.

| Input | Expected status | Checks |
|---|---|---|
| 784 zeros | 200 | digit in 0-9, confidence in [0, 1] |

**Gotchas / notes**

- It does not check that the digit is *correct* (a blank image has no right answer); it is a contract/smoke test of the response format. Correctness is covered by the MNIST test below.

### test_predict_rejects_bad_payload, parametrized (lines 26-28)

**Code**

```python
@pytest.mark.parametrize("pixels", [[0.0] * 783, [0.0] * 785, "abc", None])
def test_predict_rejects_bad_payload(client, pixels):
    assert client.post("/predict", json={"pixels": pixels}).status_code == 400
```

**What it does**

`pytest.mark.parametrize("pixels", [...])` runs the same test function once per value in the list, so this is four separate test cases, each reported individually. In each, `pixels` takes one value and the test posts `{"pixels": <value>}` and asserts HTTP 400:

| Case | Payload sent | Why the server rejects it |
|---|---|---|
| 1 | `[0.0] * 783` (783 zeros) | converts fine but shape is `(783,)`, not `(784,)` (one pixel short) |
| 2 | `[0.0] * 785` (785 zeros) | shape `(785,)`, one pixel too many |
| 3 | `"abc"` | `np.asarray("abc", dtype="float32")` raises `ValueError`; caught, `x = None` |
| 4 | `None` (JSON `null`) | `np.asarray(None, dtype="float32")` gives a 0-d array with shape `()`, which is not `(784,)` |

**Workflow / data flow**

Per case: test client POST -> `request.get_json` -> `.get("pixels")` -> `np.asarray` (or exception) -> shape check fails -> `({"error": "pixels must be a list of 784 numbers"}, 400)` -> assertion passes. The model is never called.

**Gotchas / notes**

- All four cases are JSON *objects* with a `pixels` key, which is why they reach the validation and return 400. The non-object bodies that cause a 500 (see the `/predict` notes above) are not covered by these tests.
- A missing `pixels` key, a wrong content type, and NaN/out-of-range values are not tested either.

### test_predict_matches_mnist_labels (lines 31-37)

**Code**

```python
def test_predict_matches_mnist_labels(client):
    _, (X_test, y_test) = load_data_mnist_format()
    hits = sum(
        client.post("/predict", json={"pixels": X_test[i].tolist()}).json["digit"] == y_test[i]
        for i in range(20)
    )
    assert hits >= 17
```

**What it does**

1. `load_data_mnist_format()` reads the MNIST IDX files from `mnist_data/` and returns `(X_train, y_train), (X_test, y_test)`. The `_, (X_test, y_test)` unpacking discards the training pair. `X_test` has shape `(10000, 784)` with values scaled to 0-1; `y_test` holds the true digits.
2. For each of the first 20 test images, `X_test[i].tolist()` converts the NumPy row into a plain Python list of 784 floats (JSON-serialisable), which is POSTed to `/predict`.
3. `.json["digit"] == y_test[i]` is `True` when the prediction equals the true label. `sum(...)` over a generator of booleans counts correct predictions (`True` counts as 1) in `hits`.
4. `assert hits >= 17` requires at least 17 of 20 correct, i.e. at least 85%, tolerating up to 3 misses.

**Workflow / data flow**

IDX files -> `X_test[i]` (784 floats in 0-1, the same scale the page sends) -> POST `{"pixels": [...]}` -> `{"digit": d, "confidence": c}` -> compare `d` with `y_test[i]` -> count -> threshold.

| Quantity | Value |
|---|---|
| Images sent | first 20 of the 10,000 test images |
| Passing threshold | `hits >= 17` (85%) |
| Allowed misses | 3 |

**Gotchas / notes**

- It requires the MNIST files under `mnist_data/` (this folder is in `.gitignore`) as well as the trained model.
- `load_data_mnist_format()` is called with no `num_train`/`num_test`, so it reads and decodes all 60,000 training and 10,000 test images just to use 20 test images; this adds start-up time to this test.
- 17/20 is a loose end-to-end sanity check (does the app's pre-processing and the model agree on real digits?), not an accuracy measurement; with only 20 samples it cannot distinguish a 90% model from a 97% one.
- The test uses MNIST's own 28x28 images, not images that went through the browser's canvas scaling/centering code, so it does not exercise that JavaScript.


## templates/index.html

`templates/index.html` is the whole front end of the project: one self-contained HTML page (markup, CSS and JavaScript in a single file, no libraries, no build step). Flask serves it from `app.py` via `render_template("index.html")` at `GET /`. The page lets the user draw a digit with mouse, pen or finger on a 280x280 canvas, converts that drawing into the same 784-number layout the Keras model was trained on, and POSTs it as JSON to `POST /predict`. The reply (`{"digit": ..., "confidence": ...}`) is shown under the canvas. All image preprocessing happens in the browser, in `toMnist()`; the server only checks that it received 784 numbers and runs the model.

### Document head, meta tags and title (lines 1-6)

**Code**

```html
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MNIST Digit Recognizer</title>
```

**What it does**

1. `<!doctype html>` puts the browser in standards mode (not quirks mode).
2. `<html lang="en">` declares the language (used by screen readers and translation tools).
3. `<meta charset="utf-8">` tells the browser the file is UTF-8 encoded.
4. The `viewport` meta tag makes phones use the device width as the layout width and an initial zoom of 1, instead of emulating a ~980px desktop screen. This matters because the page is meant to be drawn on with a finger.
5. `<title>` sets the browser-tab text.

**Workflow / data flow**

```
Flask GET /  ->  render_template("index.html")  ->  HTML response
browser parses <head> (charset, viewport, title, <style>)  ->  builds <body>  ->  runs <script>
```

**Gotchas / notes**

- `render_template` runs the file through Jinja2, but this file contains no `{{ }}` or `{% %}` expressions, so it is sent essentially verbatim. (If a `{{` or `{%` sequence were ever added to the JavaScript it would be interpreted by Jinja2.)
- There is no external CSS, JS or font reference, so the page makes exactly one request for itself plus the later `/predict` calls.

### The CSS block (lines 7-12)

**Code**

```css
<style>
  body { font-family: system-ui, sans-serif; text-align: center; margin: 2rem 1rem; }
  canvas { border: 2px solid #444; border-radius: 6px; touch-action: none; cursor: crosshair; max-width: 100%; }
  button { font-size: 1rem; padding: .5rem 1.2rem; margin: .8rem .3rem; }
  #out { font-size: 1.5rem; min-height: 2rem; }
</style>
```

**What it does**

- `body`: uses the operating system's UI font (`system-ui`, falling back to any sans-serif), centres all inline content (the canvas, the buttons and the result text are all inline/inline-block, so `text-align: center` centres them), and adds a 2rem top/bottom and 1rem left/right margin.
- `canvas`:
  - `border: 2px solid #444` draws a dark-grey frame so the user can see where the drawing area is (the pad itself is black, and the page background is white).
  - `border-radius: 6px` rounds the frame corners.
  - `touch-action: none` tells the browser not to treat touches on the canvas as scroll/pinch-zoom gestures; this is what lets a finger draw instead of scrolling the page, and stops the browser from firing `pointercancel` when it would otherwise take over the gesture.
  - `cursor: crosshair` shows a crosshair mouse pointer.
  - `max-width: 100%` lets the canvas shrink on screens narrower than 280 CSS pixels (the height scales with it because the canvas keeps its aspect ratio). Its internal bitmap stays 280x280; only the displayed size changes, which is why `pos()` rescales coordinates.
- `button`: a 1rem font and generous padding/margins so the buttons are easy to tap.
- `#out`: result text at 1.5rem with `min-height: 2rem`, so the area reserves vertical space and the layout does not jump when text appears.

**Workflow / data flow**

Purely presentational; no data flows through it. The only rule with behavioural effect is `touch-action: none`, which changes which pointer events the canvas receives (see the pointer-handlers snippet).

**Gotchas / notes**

- The 2px border is part of the canvas element's CSS box. The rendered box is therefore 284x284 CSS pixels (280 content + 2 x 2 border) at full size, and `getBoundingClientRect()` reports that 284x284 box. See `pos()` below for the consequence.
- The canvas has no `background` in CSS; the black comes from `fillRect` in `clearPad()`.

### Body markup: canvas, buttons and the result line (lines 13-21)

**Code**

```html
</head>
<body>
<h1>Draw a digit (0-9)</h1>
<canvas id="pad" width="280" height="280"></canvas>
<div>
  <button id="go">Predict</button>
  <button id="clear">Clear</button>
</div>
<p id="out" aria-live="polite"></p>
```

**What it does**

1. `<h1>` is the page heading telling the user what to do.
2. `<canvas id="pad" width="280" height="280">` is the drawing surface. The `width`/`height` attributes set the bitmap resolution: 280x280 device-independent bitmap pixels, i.e. exactly 10x the 28x28 MNIST size. Using an exact multiple keeps the later downscale simple.
3. A `<div>` wraps the two buttons: `#go` (Predict) and `#clear` (Clear). Their ids are looked up by the script at the bottom of the page.
4. `<p id="out" aria-live="polite">` is the output line. It starts empty. `aria-live="polite"` makes screen readers announce text changes in it (the prediction, "Draw a digit first", or an error) when the user is idle, without interrupting speech.

**Workflow / data flow**

```
user draws on #pad  ->  script reads pixels from #pad
click #go            ->  script POSTs /predict   ->  writes text into #out
click #clear         ->  script repaints #pad    ->  empties #out
```

**Gotchas / notes**

- The `<script>` element comes after these elements (line 23), so `document.getElementById(...)` finds them at execution time without needing a `DOMContentLoaded` handler.
- The buttons are plain `<button>` elements outside a `<form>`, so clicking them does not submit anything; all behaviour is attached with `addEventListener`.
- The canvas has no text alternative or keyboard interaction; drawing is pointer-only.

### Canvas setup and `clearPad()` (lines 23-32)

**Code**

```javascript
<script>
const c = document.getElementById('pad'), ctx = c.getContext('2d', { willReadFrequently: true });
const out = document.getElementById('out');

function clearPad() {
  ctx.fillStyle = '#000'; ctx.fillRect(0, 0, 280, 280);
  ctx.strokeStyle = '#fff'; ctx.lineWidth = 20; ctx.lineCap = 'round'; ctx.lineJoin = 'round';
  out.textContent = '';
}
clearPad();
```

**What it does**

1. `c` is the canvas element, `ctx` its 2D drawing context. The option `{ willReadFrequently: true }` is a hint that `getImageData` will be called, so the browser keeps the bitmap in CPU memory (faster reads, no GPU read-back stalls).
2. `out` is the result paragraph.
3. `clearPad()`:
   - sets the fill colour to black and fills the whole 280x280 bitmap, giving an opaque black background (every pixel `rgba(0,0,0,255)`). This matches MNIST, where the background is 0 and the digit is bright.
   - configures the pen: white stroke, 20-pixel line width, round caps and round joins. A 20 px pen on a 280 px pad is roughly the stroke-to-image ratio of a thick marker; round caps/joins make smooth strokes and let a single click leave a round dot.
   - empties the result text.
4. `clearPad()` is called once immediately, so the pad is black and the pen is configured from the first moment. The Clear button reuses the same function.

**Workflow / data flow**

```
page load  -> getContext('2d') -> clearPad() -> black 280x280 bitmap, pen = white 20px round
Clear click-> clearPad()       -> same state again, #out = ""
```

**Gotchas / notes**

- Pen settings (`strokeStyle`, `lineWidth`, caps) are set inside `clearPad()`, not once at startup, so they are re-applied on every clear. This is harmless but means the "pen configuration" and "reset" are the same code.
- The numbers 280 are hard-coded in `clearPad()` rather than read from `c.width`.
- Everything is global (script-level `const`/`function`); there is no module or IIFE wrapper.

### The `pos()` coordinate helper (lines 34-37)

**Code**

```javascript
const pos = e => {
  const r = c.getBoundingClientRect();
  return { x: (e.clientX - r.left) * c.width / r.width, y: (e.clientY - r.top) * c.height / r.height };
};
```

**What it does**

Converts a pointer event's viewport coordinates (`clientX`, `clientY`, in CSS pixels) into canvas bitmap coordinates (0..280):

1. `c.getBoundingClientRect()` returns the canvas's current on-screen box (`left`, `top`, `width`, `height` in CSS pixels).
2. `e.clientX - r.left` is the pointer's distance from the box's left edge, in CSS pixels.
3. Multiplying by `c.width / r.width` (bitmap width / displayed width) rescales to bitmap pixels. When the canvas is displayed at its natural size this ratio is close to 1; when `max-width: 100%` shrinks it, or the page is zoomed, the ratio corrects for it. Y is done the same way.

**Workflow / data flow**

```
PointerEvent (clientX, clientY)  --pos()-->  {x, y} in 0..280 bitmap coordinates
                                              |
                                              +-> ctx.moveTo / ctx.lineTo
```

**Gotchas / notes**

- `getBoundingClientRect()` returns the border box, so it includes the 2px CSS border. `r.left` is the outer edge of the border and `r.width` is 284 at natural size, not 280. A pointer on the first content pixel (2 CSS px in from the outer edge) maps to about `2 * 280/284 ~ 1.97`, not 0, so drawing lands about 2 bitmap pixels down and to the right of the cursor, and the scale factor is 280/284 instead of 1. The error is small relative to a 20px pen, and since `toMnist()` crops to the drawing's bounding box, a uniform offset has no effect on the model input.
- `pos` is an arrow function stored in a `const`, so it must be defined before use; it is, since handlers run later.

### Pointer event handlers (lines 38-47)

**Code**

```javascript
let drawing = false;
c.addEventListener('pointerdown', e => {
  drawing = true; c.setPointerCapture(e.pointerId);
  const p = pos(e); ctx.beginPath(); ctx.moveTo(p.x, p.y); ctx.lineTo(p.x + 0.01, p.y); ctx.stroke();
});
c.addEventListener('pointermove', e => {
  if (!drawing) return;
  const p = pos(e); ctx.lineTo(p.x, p.y); ctx.stroke();
});
['pointerup', 'pointercancel'].forEach(t => c.addEventListener(t, () => { drawing = false; }));
```

**What it does**

The handlers use Pointer Events, a single API that covers mouse, touch and pen.

1. `drawing` is a boolean "is a stroke in progress" flag.
2. **`pointerdown`**: sets `drawing = true`; calls `c.setPointerCapture(e.pointerId)` so that all further events for this pointer are delivered to the canvas even if the pointer moves outside it (so `pointerup` is not lost and the stroke does not get stuck "on"). Then it starts a new path at the pointer (`beginPath`, `moveTo`) and draws a 0.01-pixel-long line (`lineTo(x + 0.01, y)` then `stroke`). A zero-length line with a round cap might not render in all implementations; the tiny offset guarantees a dot (a filled circle of diameter 20, thanks to `lineCap = 'round'`) appears on a simple click or tap.
3. **`pointermove`**: ignored unless `drawing`; otherwise extends the current path to the new position with `lineTo` and calls `stroke()`.
4. **`pointerup` / `pointercancel`**: both just set `drawing = false`. `pointercancel` is what the browser fires if it takes the touch over (e.g. for scrolling, which `touch-action: none` is there to prevent). Registering both in one `forEach` avoids duplicating the handler.

**Workflow / data flow**

```
pointerdown -> drawing=true, capture pointer, new path, dot at (x,y)
pointermove -> (if drawing) path.lineTo(x,y); stroke()   [repeats]
pointerup / pointercancel -> drawing=false
                              (pixels stay on the canvas; they are read later by toMnist())
```

**Gotchas / notes**

- `pointerdown` does not check `e.button`, so any mouse button (left, middle, right) starts drawing.
- Each `pointermove` calls `ctx.stroke()` on the whole current path (everything since the last `beginPath()` at `pointerdown`), not only the newest segment. A long stroke is therefore re-drawn in full on every move event. Because the stroke is opaque white on black this does not change the shape, but the anti-aliased edge pixels are composited repeatedly and so end up slightly harder/brighter than a single pass would give.
- The path is only reset on `pointerdown` (`beginPath`), so separate strokes (pen lifts) are separate paths; they do not get joined by a line.
- There is no `pointerleave`/`lostpointercapture` handling; the capture is what keeps `pointerup` arriving.
- The `drawing` flag is a single global: it does not track which pointer is drawing, so a second simultaneous pointer would also be treated as part of the same drawing state.

### `toMnist()`: turning the 280x280 drawing into 784 floats (lines 49-74)

**Code**

```javascript
// MNIST digits are scaled to fit 20x20 and centred by centre of mass in 28x28; match that.
function toMnist() {
  const d0 = ctx.getImageData(0, 0, 280, 280).data;
  let x0 = 280, y0 = 280, x1 = -1, y1 = -1;
  for (let y = 0; y < 280; y++) for (let x = 0; x < 280; x++) {
    if (d0[(y * 280 + x) * 4] > 0) {
      if (x < x0) x0 = x; if (x > x1) x1 = x;
      if (y < y0) y0 = y; if (y > y1) y1 = y;
    }
  }
  if (x1 < 0) return null;
  const w = x1 - x0 + 1, h = y1 - y0 + 1, s = 20 / Math.max(w, h);
  const mk = () => { const k = document.createElement('canvas'); k.width = k.height = 28; return k; };
  const a = mk(), actx = a.getContext('2d', { willReadFrequently: true });
  actx.fillRect(0, 0, 28, 28);
  actx.drawImage(c, x0, y0, w, h, (28 - w * s) / 2, (28 - h * s) / 2, w * s, h * s);
  const d1 = actx.getImageData(0, 0, 28, 28).data;
  let m = 0, cx = 0, cy = 0;
  for (let i = 0; i < 784; i++) { const v = d1[i * 4]; m += v; cx += v * (i % 28); cy += v * Math.floor(i / 28); }
  cx /= m; cy /= m;
  const b = mk(), bctx = b.getContext('2d', { willReadFrequently: true });
  bctx.fillRect(0, 0, 28, 28);
  bctx.drawImage(a, Math.round(13.5 - cx), Math.round(13.5 - cy));
  const o = bctx.getImageData(0, 0, 28, 28).data;
  return Array.from({ length: 784 }, (_, i) => o[i * 4] / 255);
}
```

**What it does**

This function reproduces, in the browser, how MNIST images were prepared, so the user's drawing looks like the training data:

- The MNIST digits were size-normalised to fit inside a **20x20** box (keeping their aspect ratio) and then placed in a **28x28** image so that the digit's **centre of mass** sits at the image centre. That leaves a ~4-pixel black margin around the digit's longest side. An MLP on raw pixels is very sensitive to where the strokes sit and how big they are, so a raw 280x280 drawing shrunk to 28x28 (which could be tiny, off-centre or touching the edges) would be out of distribution.
- `toMnist()` does exactly those steps, then flattens the result to a plain array of 784 numbers in [0, 1]: row-major (index = row * 28 + col), white digit on black, which is the layout `src/data.py` produces when it reads the IDX files (`reshape(size, 784)` then `/ 255.0` to float32).
- It returns `null` if the canvas is empty, otherwise an array of 784 numbers.

The work is split into five steps, each documented below.

**Workflow / data flow**

```
280x280 canvas (RGBA, white strokes on black)
   |  1. scan red channel for pixels > 0
   v
bounding box (x0, y0) .. (x1, y1)  -> w = x1-x0+1, h = y1-y0+1
   |  2. s = 20 / max(w, h);  drawImage(crop w x h -> (w*s) x (h*s), centred in 28x28)
   v
28x28 canvas "a"  (digit fits in 20x20, centred by bounding box)
   |  3. centre of mass (cx, cy) from pixel values
   |  4. draw "a" onto new 28x28 canvas "b" shifted by round(13.5 - cx), round(13.5 - cy)
   v
28x28 canvas "b"  (centre of mass at the centre)
   |  5. getImageData -> every 4th byte (red) / 255
   v
Array of 784 floats in [0, 1], row-major
```

**Gotchas / notes**

- Returns `null` (not an empty array) for a blank pad; the Predict handler checks for it.
- Only the red channel is ever read. That is sufficient because the pad contains only black, white and greys (equal R, G, B).
- Because `s = 20 / max(w, h)` is not capped at 1, a small drawing is scaled up, not just down. The pen is 20 px wide in 280 px space, so for a full-height digit (h ~ 280) the stroke becomes about 20 * 20/280 ~ 1.4 px wide in the 28x28 image.
- There is no guard for `m === 0` before `cx /= m` (division would give `NaN`); with a 20 px round pen, the bounding-box check at step 1 means there is always visible drawing, so this branch is not normally reached.

#### Step 1: bounding box scan (lines 51-59)

**Code**

```javascript
  const d0 = ctx.getImageData(0, 0, 280, 280).data;
  let x0 = 280, y0 = 280, x1 = -1, y1 = -1;
  for (let y = 0; y < 280; y++) for (let x = 0; x < 280; x++) {
    if (d0[(y * 280 + x) * 4] > 0) {
      if (x < x0) x0 = x; if (x > x1) x1 = x;
      if (y < y0) y0 = y; if (y > y1) y1 = y;
    }
  }
  if (x1 < 0) return null;
```

**What it does**

1. `getImageData(0, 0, 280, 280).data` returns a `Uint8ClampedArray` of the whole canvas in **RGBA** order, 4 bytes per pixel, row by row. So the red byte of pixel (x, y) is at index `(y * 280 + x) * 4`, green at +1, blue at +2, alpha at +3.
2. `x0`, `y0` start at 280 (larger than any valid index) and `x1`, `y1` at -1 (smaller than any), so the first non-black pixel always updates all four.
3. The double loop visits all 78,400 pixels. Any pixel whose red value is above 0 (i.e. not pure black, including faint anti-aliased fringe) widens the min/max x and y.
4. If no pixel was non-black, `x1` is still -1, so the function exits with `null` ("nothing drawn").

**Workflow / data flow**

```
d0: 280*280*4 bytes  ->  loop over (x,y)  ->  min/max of x and y where red > 0
                                           ->  (x0,y0,x1,y1)  or  null if canvas empty
```

**Gotchas / notes**

- The threshold is `> 0`, so even a nearly black anti-aliased fringe pixel counts and slightly enlarges the box.
- The coordinates are inclusive pixel indices, hence `w = x1 - x0 + 1` in the next step.

#### Step 2: scale to fit 20x20 and draw into a 28x28 canvas (lines 60-65)

**Code**

```javascript
  const w = x1 - x0 + 1, h = y1 - y0 + 1, s = 20 / Math.max(w, h);
  const mk = () => { const k = document.createElement('canvas'); k.width = k.height = 28; return k; };
  const a = mk(), actx = a.getContext('2d', { willReadFrequently: true });
  actx.fillRect(0, 0, 28, 28);
  actx.drawImage(c, x0, y0, w, h, (28 - w * s) / 2, (28 - h * s) / 2, w * s, h * s);
  const d1 = actx.getImageData(0, 0, 28, 28).data;
```

**What it does**

1. `w` and `h` are the width and height of the bounding box in pixels (inclusive, so +1).
2. `s = 20 / Math.max(w, h)` is one uniform scale factor chosen so that the **longer** side becomes exactly 20 pixels. Using one factor for both axes preserves the aspect ratio: a thin "1" stays thin, a wide "0" stays wide; only the longer side is pinned to 20 and the shorter side ends up <= 20. For example a "1" with w = 30, h = 200 gives s = 0.1 and a 3x20 image; a "0" with w = 150, h = 180 gives s = 0.111 and about 16.7x20.
3. `mk()` creates an off-screen (not attached to the page) 28x28 canvas.
4. `actx.fillRect(0, 0, 28, 28)` paints it black: a fresh canvas is transparent black, and the default fill style is black, so this makes it opaque black, matching MNIST's 0 background.
5. `drawImage` is used in its **9-argument form**: `drawImage(source, sx, sy, sw, sh, dx, dy, dw, dh)`. The first four numbers select a source rectangle (the bounding box in the big canvas `c`), the last four a destination rectangle in the small canvas. The browser therefore crops and rescales (with its built-in image smoothing, which averages source pixels, giving grey anti-aliased edges like MNIST's) in one native call, instead of the code doing the resampling itself.
6. The destination rectangle is `w*s` by `h*s`, placed at `((28 - w*s)/2, (28 - h*s)/2)`: centred in the 28x28 canvas by bounding box. The longer side then occupies exactly pixels 4..24, leaving a 4-pixel margin on each side.
7. `d1` is the RGBA data of this 28x28 canvas (28*28*4 = 3136 bytes).

**Workflow / data flow**

```
280x280 canvas c
  crop (x0,y0,w,h) --scale s--> (w*s) x (h*s)   [longer side = 20]
  paste at ((28-w*s)/2, (28-h*s)/2) on black 28x28 canvas "a"
  d1 = RGBA bytes of "a"  (3136 bytes)
```

**Gotchas / notes**

- The destination offsets and sizes are generally fractional, and the browser resamples with smoothing, so the edges are grey, not binary.
- The temporary canvases are never added to the DOM and are garbage-collected after the function returns.
- `actx` default fill is black only because nobody has changed `fillStyle` on this new context; the main canvas's `#000`/`#fff` settings do not affect it.

#### Step 3: centre of mass (lines 66-68)

**Code**

```javascript
  let m = 0, cx = 0, cy = 0;
  for (let i = 0; i < 784; i++) { const v = d1[i * 4]; m += v; cx += v * (i % 28); cy += v * Math.floor(i / 28); }
  cx /= m; cy /= m;
```

**What it does**

The centre of mass treats pixel brightness as mass:

1. For each of the 784 pixels `i`, `v = d1[i * 4]` is that pixel's red value (0..255). `i * 4` is the byte offset because each pixel takes 4 bytes (R, G, B, A), so index `i*4` is pixel `i`'s red channel.
2. The pixel's column is `i % 28` and its row is `Math.floor(i / 28)` (row-major order).
3. `m` accumulates total mass (sum of v), `cx` the sum of `v * column`, `cy` the sum of `v * row`.
4. Dividing by `m` gives the weighted mean column and row: the centre of mass `(cx, cy)`, in **pixel-index coordinates** (pixel 0 has coordinate 0, pixel 27 has coordinate 27).

**Workflow / data flow**

```
d1 (28x28 RGBA) -> for i in 0..783: v = red(i), col = i%28, row = floor(i/28)
                -> m = sum v ; cx = sum(v*col)/m ; cy = sum(v*row)/m
```

**Gotchas / notes**

- Values stay on the 0..255 scale here; the division by m cancels the scale, so no /255 is needed.
- The result is generally fractional and can sit anywhere from the geometric centre of the bounding box (e.g. a "1" or "0") to well off it (e.g. a "7" or "4", whose ink mass is concentrated in one region).

#### Step 4: shift so the centre of mass is at the centre (lines 69-71)

**Code**

```javascript
  const b = mk(), bctx = b.getContext('2d', { willReadFrequently: true });
  bctx.fillRect(0, 0, 28, 28);
  bctx.drawImage(a, Math.round(13.5 - cx), Math.round(13.5 - cy));
```

**What it does**

1. Creates a second black 28x28 canvas `b` (same recipe as step 2).
2. Computes the shift needed to move the centre of mass to the image centre: `13.5 - cx` horizontally, `13.5 - cy` vertically.
3. **Why 13.5:** `cx`/`cy` were computed using pixel *indices* (0..27), i.e. the coordinate of pixel `i` is `i`. The middle of a 28-pixel row in index coordinates is between pixel 13 and pixel 14, at (0 + 27)/2 = 13.5. (In pixel-edge coordinates, where pixel `i` spans `i..i+1`, the same point is 14.0; the half-pixel difference is just because index coordinates measure to the pixel's centre, which is `i + 0.5` in edge coordinates.)
4. `Math.round` makes both offsets whole numbers. A 3-argument `drawImage(a, dx, dy)` at an integer offset copies pixels exactly; a fractional offset would make the browser interpolate and blur the digit.
5. The black fill of `b` shows through where the shifted image no longer covers; anything shifted off the edge is clipped.

**Workflow / data flow**

```
(cx, cy)  -->  dx = round(13.5 - cx), dy = round(13.5 - cy)
canvas a  --drawImage(a, dx, dy)-->  canvas b (black 28x28)
```

**Gotchas / notes**

- Rounding means the centre of mass lands within half a pixel of 13.5, not exactly on it.
- The shift moves the whole bounding-box-centred image; because the digit was confined to a 20x20 box in a 28x28 canvas there are 4 spare pixels per side, so typical shifts keep the digit inside the frame.

#### Step 5: flatten to 784 floats (lines 72-73)

**Code**

```javascript
  const o = bctx.getImageData(0, 0, 28, 28).data;
  return Array.from({ length: 784 }, (_, i) => o[i * 4] / 255);
```

**What it does**

1. `getImageData(0, 0, 28, 28).data` gives the final canvas's RGBA bytes (3136 bytes).
2. `Array.from({ length: 784 }, (_, i) => ...)` builds a normal JavaScript array of 784 elements by calling the mapper for `i = 0..783` (the first argument, the placeholder value, is unused, hence `_`).
3. `o[i * 4]` picks pixel `i`'s red byte (4 bytes per pixel), and `/ 255` maps 0..255 to 0.0..1.0, the same scaling `src/data.py` applies (`/ 255.0`).
4. Pixel `i` here is row `floor(i / 28)`, column `i % 28`, which is the same row-major order as the 784-vector in the training data.

**Workflow / data flow**

```
canvas b (28x28 RGBA, 3136 bytes) -> take every 4th byte (red) -> /255 -> Array(784) of floats in [0,1]
```

**Gotchas / notes**

- A plain `Array` (not a typed array) is returned so that `JSON.stringify` produces a normal JSON list; a `Uint8ClampedArray`/`Float32Array` would serialise as an object with numeric keys.
- The numbers are doubles in JS, e.g. `0.6392156862745098`; the server casts them to `float32` (`np.asarray(pixels, dtype="float32")`).

### Clear button handler (line 76)

**Code**

```javascript
document.getElementById('clear').addEventListener('click', clearPad);
```

**What it does**

Looks up the Clear button and registers `clearPad` directly as its click handler. Clicking it repaints the canvas black, re-applies the pen settings and empties the `#out` text (see `clearPad()` above).

**Workflow / data flow**

```
click #clear -> clearPad() -> canvas black, #out = ""
```

**Gotchas / notes**

- `clearPad` is passed as a function reference and so is called with the click event as an (ignored) argument.
- The result text is cleared only here (and at load). Drawing more strokes after a prediction leaves the old "Prediction: ..." text on screen until the user clicks Clear or Predict again.
- Clearing does not cancel an in-flight `fetch`; a late response can still write into `#out` after a Clear.

### Predict button handler (lines 77-89)

**Code**

```javascript
document.getElementById('go').addEventListener('click', async () => {
  const pixels = toMnist();
  if (!pixels) { out.textContent = 'Draw a digit first'; return; }
  try {
    const r = await fetch('/predict', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ pixels })
    });
    const j = await r.json();
    out.textContent = r.ok ? `Prediction: ${j.digit} (${(j.confidence * 100).toFixed(1)}%)` : j.error;
  } catch (e) {
    out.textContent = 'Could not reach the server';
  }
});
```

**What it does**

1. Registers an `async` click handler on the Predict button.
2. Calls `toMnist()`. If it returns `null` (empty pad) the handler writes "Draw a digit first" into `#out` and returns without contacting the server.
3. Otherwise it `fetch`es `/predict` (same origin, so a relative URL) with method POST, a `Content-Type: application/json` header, and a body of `{"pixels": [784 numbers]}` (`{ pixels }` is shorthand for `{ pixels: pixels }`).
4. `await r.json()` parses the response body as JSON. On success `app.py` returns `{"digit": int, "confidence": float}`; on a bad payload it returns `{"error": "pixels must be a list of 784 numbers"}` with status 400.
5. `r.ok` is true for HTTP status 200-299. If true, it displays `Prediction: <digit> (<confidence as percent with 1 decimal>%)`, e.g. `Prediction: 7 (98.3%)`. Otherwise it displays the server's `j.error` text.
6. The `try/catch` handles a rejected `fetch` (network failure, server not running) or a failed `r.json()` (body is not JSON) by showing "Could not reach the server".

**Workflow / data flow**

```
click #go
  -> toMnist()                       -> Array(784) or null
       null  -> #out = "Draw a digit first"          (no request)
  -> fetch POST /predict  {"pixels":[...784 floats...]}
        Flask: get_json -> np.asarray float32 -> shape (784,)? -> reshape(1,784) -> model.predict
  <- 200 {"digit": 7, "confidence": 0.983}   -> #out = "Prediction: 7 (98.3%)"
  <- 400 {"error": "..."}                    -> #out = that error text
  <- network error / non-JSON body           -> #out = "Could not reach the server"
```

**Gotchas / notes**

- A non-JSON error response (for example Flask's HTML page for a 404 or a 500 crash) makes `r.json()` throw, which is caught by the same `catch`, so the user sees "Could not reach the server" even though the server did respond.
- The 400 error path only works because `app.py` returns JSON for it; an error response that is JSON but has no `error` field would display `undefined`.
- The button is not disabled during the request, so repeated clicks send repeated requests; the last response to arrive wins.
- Numbers are sent as JSON text of 784 doubles (a few KB), not as an image.
- The confidence is the model's maximum softmax output (`probs.max()` in `app.py`), shown as a percentage.

### What the user sees vs what the model gets

The user sees a 280x280 black pad with thick white strokes of wherever they drew, at whatever size and position they chose, plus a one-line text result. The model never sees that picture: it receives 784 floats in [0, 1], the drawing cropped to its bounding box, scaled (aspect ratio preserved) so its longer side is 20 pixels, placed in a black 28x28 frame and nudged by a whole number of pixels so its centre of mass is at index (13.5, 13.5), flattened row by row. That matches how `src/data.py` supplies MNIST (784 floats, `/255`, white digit on black), so a small, large, left-aligned or off-centre drawing all arrive in roughly the same canvas position and size as a training digit.

---

## End-to-end request lifecycle (browser to model and back)

1. **Draw and click.** In the browser at <http://127.0.0.1:5000> the user draws white strokes on the black 280x280 canvas and clicks **Predict**.
2. **Page JavaScript builds the payload.** The click handler calls `toMnist()`, which crops the drawing, scales it to fit 20x20, centres it by centre of mass in a 28x28 image, and returns 784 numbers in 0-1 (or `null`, in which case the page shows "Draw a digit first" and sends nothing).
3. **`fetch`.** The page does `fetch('/predict', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({pixels})})`, sending `{"pixels": [784 floats]}`.
4. **Flask routing.** The dev server on 127.0.0.1:5000 matches `POST /predict` (decorated with `@app.post("/predict")`) and calls `predict()`.
5. **Parse and validate.** `request.get_json(silent=True)` parses the body; `.get("pixels")` extracts the list; `np.asarray(..., dtype="float32")` converts it; `x.shape != (784,)` rejects anything that is not a flat 784-vector with a 400 and `{"error": ...}`.
6. **Keras inference.** `x.reshape(1, 784)` makes a one-image batch; `model.predict(..., verbose=0)[0]` returns the 10 softmax probabilities from the already-loaded model.
7. **Result to JSON.** `argmax()` gives the digit and `max()` gives the confidence; the returned dict `{"digit": 7, "confidence": 0.98}` is JSON-encoded by Flask with status 200.
8. **Page shows the text.** The `fetch` promise resolves, `r.json()` parses the reply, and if `r.ok` the page writes `Prediction: 7 (98.0%)` into the `#out` paragraph; on an error status it shows the `error` string, and if the request itself fails (server down, or the reply is not parseable JSON) it shows "Could not reach the server".

---

# Appendix: Observations collected during the analysis

Everything below was noticed while reading the code. **None of it has been changed**; it is listed so you can decide what, if anything, to act on. Items are stated factually, grouped by file.

| File | Observation | Effect |
|---|---|---|
| `app.py:22` | `(request.get_json(silent=True) or {}).get(...)` only replaces *falsy* bodies. A JSON body that is a non-empty list, string or number raises `AttributeError` | HTTP 500 instead of 400 |
| `app.py:23-28` | Only the *shape* is validated. `NaN`, `Infinity`, out-of-range numbers, `true`/`false` and numeric strings pass | Model can return NaN, which Flask serialises as invalid JSON; the page then shows "Could not reach the server" |
| `app.py:34` | `app.run(host=..., port=...)` does not pass `debug=False` explicitly; the default can be overridden by the `FLASK_DEBUG` environment variable | Debugger and reloader would turn on if that variable is set |
| `app.py`, `templates/` | Listed in `.gitignore` (lines 10-11) | `git add .` would commit `test_app.py` and the README section but not the files they depend on |
| `.gitignore:138-142` | Unresolved merge-conflict markers | Harmless as patterns, but junk |
| `main.py:16` | `MODEL_PATH` is relative to the current working directory, unlike `app.py`, which is anchored to its own file | `python main.py` from another folder looks for / creates `artifacts/` there |
| `src/logger.py:11` | `logs/` is created under `os.getcwd()` at import time | Same cwd dependence; import fails if the directory is not writable |
| `main.py:61-67` | Commented-out example-prediction block | Dead code; would raise `NameError` if enabled and the model file already existed |
| `main.py:20,70` | Model is always reloaded from disk in `load_and_predict`, even right after training | Redundant, harmless |
| `src/data.py:5`, `src/plot.py:2`, `src/exception_h.py:2`, `test_app.py:1` | Unused imports (`CustomException`, `os`, `logging`, `numpy`) | None at runtime |
| `src/plot.py` | `plot_sample_images` has no callers anywhere | Dead module |
| `src/data.py:51-54` | `num_train` / `num_test` are never passed by any caller; slicing happens after the full files are read | Saves no loading time |
| `src/Model.py` | `compile_and_train` default `epochs=30` differs from the `epochs=10` that `main.py` uses | Confusing default only |
| `src/exception_h.py:7` | Constructing `CustomException` outside an `except` block raises `AttributeError` (no guard on `exc_tb is None`) | Not hit by any current call site |
| `templates/index.html` `pos()` | Uses the border box from `getBoundingClientRect()`, so strokes land up to about 2 px off the cursor | Cosmetic; the crop step makes it irrelevant to the model |
| `templates/index.html` pointer handlers | Re-strokes the whole path on every move; any mouse button starts drawing; the previous result stays on screen until Predict or Clear | Cosmetic / minor UX |
| `templates/index.html` Predict handler | A non-JSON error response is reported as "Could not reach the server" | Misleading message |
| `test_app.py` | `hits >= 17` of 20 is lenient; no test for the browser's `toMnist()`, for non-object JSON bodies or for NaN input; loads all 70,000 images to use 20 | Test gaps |
| `README.md` | Project-structure tree omits `app.py`, `templates/`, `test_app.py`; it also still lists the deleted `mnist_model.h5`; "Future Improvements" still lists a web interface | Stale documentation |

---

*End of walkthrough.*
