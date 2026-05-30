# VICA — Visual Chess Analyzer

This project turns a screenshot of a chess board on lichess into a position that can be read by a chess engine. 
Give VICA a board image, it finds the pieces, builds the FEN representation of the position, and asks Stockfish for the best move.

This repository contains the final project for the "Introduction to Computer Vision" course at EPICODE Institute of Technology.
---

## Overview


How it works

VICA expects a clear digital screenshot of a chess board. It processes the image in six steps to produce a playable chess position:

1. Image acquisition & square extraction — the input board image is divided into an 8×8 grid so each square can be inspected individually.
2. Occupancy detection — simple image measures (edge density and colour
   variation) are used to decide which squares contain a piece.
3. Piece classification — occupied squares are classified into piece types using a trained convolutional network (ResNet-18 backbone).
4. Rule-based post-processing — basic chess rules are applied to catch and fix obvious mistakes (for example, impossible piece counts).
5. FEN construction — the detected board, with piece colours inferred from brightness, is converted into Forsyth–Edwards Notation (FEN).
6. Engine analysis — the FEN string is given to Stockfish to evaluate the
   position and suggest a move.

The implementation mixes traditional image-processing steps (OpenCV), a
trained classifier (PyTorch), and straightforward rule checks to improve
robustness.

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

Notes on the classifier

The piece classifier was trained by first fitting the final classification
layers and then fine-tuning the last residual block. This helps the network
adapt to the chess-piece images without overfitting.

---

## Setup and Running Instructions

Requirements

- Python 3.10 or newer
- Stockfish chess engine (executable)

Install dependencies

On Windows:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Add Stockfish

Download the Stockfish executable from https://stockfishchess.org/download/
and place it in the repository under:

engine/stockfish.exe

Run the analyzer

Put a screenshot into data/test_photos/ and run:

```bash
python src/main.py
```

The script prompts for the image filename, which side is to move, and the
board orientation. It prints the detected FEN and Stockfish's recommendation.

You may pass the image path as an argument instead:

```bash
python src/main.py data/test_photos/screenshot1.png
```

---

Project layout

```
Chess-vision/
├── src/                       # main code
│   ├── piece_classifier.py    # piece classifier (training & inference)
│   ├── board_analyzer.py      # split board into squares + occupancy checks
│   ├── fen_builder.py         # build FEN string and determine piece colours
│   ├── rule_checker.py        # simple chess-rule validation and fixes
│   ├── chess_advisor.py       # talk to Stockfish using UCI protocol
│   └── main.py                # glue code: full pipeline and user prompts
├── notebooks/                 # experiments, training and data prep
├── models/                    # saved model weights (.pth)
├── engine/                    # place Stockfish executable here
├── data/                      # images and datasets
├── docs/                      # report and analysis PDF
├── requirements.txt
└── README.md
```

---

Model training and results

The classifier was trained with transfer learning from an ImageNet-pretrained
ResNet-18. Two variants were produced: one trained on photographs of physical
pieces (about 85% test accuracy on that dataset) and another trained on
synthetic digital pieces. Details, training scripts and logs are available in
the notebooks/ folder.

Performance notes

- On clean, in-domain screenshots the pipeline usually reconstructs positions
   correctly and Stockfish returns sensible moves.
- Accuracy falls off on out-of-domain inputs (different piece styles, camera
   photos of physical boards). This is a common domain-shift issue and is
   discussed in the project report.

Documentation

See docs/VICA_Technical_Analysis.pdf for the full project report: problem
description, methods, experiments, failure cases and final thoughts.

Usage and ethics

This tool is intended for analysis, training and study. Do not use it to
cheat in live rated games or otherwise violate the rules of chess platforms.
