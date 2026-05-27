# VICA — Visual Chess Analyzer

> Automated chess position analysis from images. Give VICA a screenshot of a
> chess board, and it detects the position, reconstructs it in FEN notation,
> and uses the Stockfish engine to suggest the best move.

This is the final project for the **Introduction to Computer Vision** course
at EPICODE Institute of Technology.

---

## Overview

VICA takes a digital chess board screenshot and runs it through a six-stage
computer vision pipeline:

1. **Image acquisition & square extraction** — the board image is split into
   an 8×8 grid of 64 squares.
2. **Occupancy detection** — classical CV (Canny edge density + colour
   variance) decides which squares contain a piece.
3. **Piece classification** — a fine-tuned ResNet-18 CNN classifies each
   occupied square into one of six piece types.
4. **Rule-based post-processing** — chess rules validate and correct obvious
   classification errors.
5. **FEN construction** — the detected board, with piece colours determined
   by a brightness analysis, is converted into a FEN string.
6. **Engine analysis** — Stockfish evaluates the position and returns the
   best move.

The project combines **classical computer vision** (OpenCV) with **deep
learning** (PyTorch) and **symbolic reasoning** (chess rules).

---

## Pipeline & Architecture

| Stage | Module | Technique |
|-------|--------|-----------|
| Square extraction | `board_analyzer.py` | Image slicing (OpenCV) |
| Occupancy detection | `board_analyzer.py` | Canny edges + colour std (classical CV) |
| Piece classification | `piece_classifier.py` | ResNet-18, transfer learning (PyTorch) |
| Rule-based correction | `rule_checker.py` | Chess-rule validation |
| FEN + colour detection | `fen_builder.py` | Brightness analysis + FEN encoding |
| Engine analysis | `chess_advisor.py` | Stockfish via UCI protocol |
| Orchestration | `main.py` | End-to-end pipeline |

The classifier uses **two-stage fine-tuning**: first the ResNet-18 backbone is
frozen and only the classifier head is trained; then the final residual block
is unfrozen and trained with a reduced learning rate.

---

## Setup and Running Instructions

### 1. Requirements

- Python 3.10 or newer
- The Stockfish chess engine executable

### 2. Install dependencies

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 3. Add the Stockfish engine

Download Stockfish from <https://stockfishchess.org/download/>, and place the
executable at:

```
engine/stockfish.exe
```

### 4. Run the analyzer

Place a chess board screenshot in `data/test_photos/`, then run:

```bash
python src/main.py
```

The program will ask for the screenshot filename, whose turn it is, and the
board orientation, then print the detected position and the best move.

You can also pass the image directly:

```bash
python src/main.py data/test_photos/screenshot1.png
```

---

## Project Structure

```
Chess-vision/
├── src/                       # Source code
│   ├── piece_classifier.py    # CNN piece classifier
│   ├── board_analyzer.py      # Board splitting + occupancy detection
│   ├── fen_builder.py         # FEN construction + colour detection
│   ├── rule_checker.py        # Rule-based validation/correction
│   ├── chess_advisor.py       # Stockfish integration
│   └── main.py                # End-to-end pipeline
├── notebooks/                 # Data prep & exploration scripts
├── models/                    # Trained CNN weights (.pth)
├── engine/                    # Stockfish executable
├── data/                      # Datasets and test images
├── docs/                      # Technical Analysis Document (PDF)
├── requirements.txt
└── README.md
```

---

## Model Training

The classifier was trained on Google Colab (free T4 GPU) using transfer
learning from an ImageNet-pretrained ResNet-18. Two models were produced:

- a model trained on photographs of physical pieces (~85% test accuracy);
- a model trained on synthetically generated digital pieces.

**Training notebook (Google Colab):**
`https://colab.research.google.com/drive/1AloU4KelGsZWzucMlHSvW4gwhdM-6pz9?usp=sharing`

> To get the link: in Colab, click **Share** (top right) → set access to
> "Anyone with the link" → **Copy link**, and paste it above.

See the Technical Analysis Document for a critical discussion of these
results.

---

## Summary of Results

- The piece classifier reaches approximately **85%** test accuracy on the
  physical-piece dataset.
- On clean in-domain screenshots, the full pipeline correctly reconstructs the
  position and returns a valid engine recommendation.
- Performance degrades on out-of-domain inputs (different piece styles,
  physical boards) due to **domain shift** — discussed in detail in the
  Technical Analysis Document.

---

## Documentation

The complete **Technical Analysis Document** (problem statement, methodology,
experimental results, failure analysis, and ethical considerations) is
available in `docs/VICA_Technical_Analysis.pdf`.

---

## Ethical Note

VICA is intended for **post-game analysis, study, and training only**. Using
chess-engine assistance during live rated games violates the rules of all
major chess platforms and federations.
