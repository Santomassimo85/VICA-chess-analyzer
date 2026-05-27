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
| Orchestration | `main.py` | End-to-end pipeline (command line) |
| Web interface | `app.py` | Streamlit web app |

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
venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Download the trained models

The trained CNN model weights are hosted on the Hugging Face Hub:

**Models:** <https://huggingface.co/LucaSantomassimo/vica-chess-analyzer>

Download `chess_classifier_digital.pth` and `chess_classifier.pth` and place
them in the `models/` folder.

### 4. Add the Stockfish engine

Download Stockfish from <https://stockfishchess.org/download/>, and place the
executable at `engine/stockfish.exe`.

### 5. Run the analyzer (command line)

```bash
python src/main.py
```

### 6. Run the web app (optional)

```bash
streamlit run src/app.py
```

---

## Project Structure

Chess-vision/
├── src/
│   ├── piece_classifier.py
│   ├── board_analyzer.py
│   ├── fen_builder.py
│   ├── rule_checker.py
│   ├── chess_advisor.py
│   ├── main.py
│   └── app.py                 # Streamlit web app
├── notebooks/
├── models/                    # see step 3 (Hugging Face)
├── engine/                    # see step 4 (Stockfish)
├── data/
├── docs/                      # Technical Analysis Document (PDF)
├── requirements.txt
└── README.md

---

## Model Training

The classifier was trained on Google Colab using transfer learning from an
ImageNet-pretrained ResNet-18. Two models were produced:

- physical-piece model (~85% test accuracy)
- digital-piece model (synthetic dataset)

**Training notebook (Google Colab):**
<https://colab.research.google.com/drive/1AloU4KelGsZWzucMlHSvW4gwhdM-6pz9?usp=sharing>

---

## Summary of Results

- ~85% per-class test accuracy on the physical-piece dataset
- Full pipeline reconstructs in-domain positions and returns engine analysis
- Performance degrades on out-of-domain inputs due to **domain shift** —
  discussed in the Technical Analysis Document

---

## Documentation

Complete **Technical Analysis Document** in `docs/VICA_Technical_Analysis.pdf`.

---

## Ethical Note

VICA is intended for **post-game analysis, study, and training only**. Using
chess-engine assistance during live rated games violates the rules of all
major chess platforms and federations.
