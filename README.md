# VICA — Visual Chess Analyzer

This project turns a screenshot of a chess board into a position that can be
read by a chess engine. Give VICA a board image, it finds the pieces, builds
the FEN representation of the position, and asks Stockfish for the best move.

Final project for the "Introduction to Computer Vision" course at EPICODE
Institute of Technology.

---

## How it works

VICA expects a clear digital screenshot of a chess board. It processes the
image in six steps:

1. **Image acquisition & square extraction** — the board image is divided
   into an 8×8 grid so each square can be inspected.
2. **Occupancy detection** — edge density and colour variation decide which
   squares contain a piece.
3. **Piece classification** — occupied squares are classified using a
   fine-tuned ResNet-18.
4. **Rule-based post-processing** — basic chess rules catch and fix obvious
   mistakes.
5. **FEN construction** — the detected board is converted into FEN, with
   piece colours inferred from brightness.
6. **Engine analysis** — Stockfish evaluates the position and returns the
   best move plus alternatives.

The implementation mixes classical image processing (OpenCV), deep learning
(PyTorch), and symbolic rule-based reasoning.

---

## Pipeline & Architecture

| Stage | Module | Technique |
|-------|--------|-----------|
| Square extraction | `board_analyzer.py` | Image slicing (OpenCV) |
| Occupancy detection | `board_analyzer.py` | Canny edges + colour std |
| Piece classification | `piece_classifier.py` | ResNet-18, transfer learning |
| Rule-based correction | `rule_checker.py` | Chess-rule validation |
| FEN + colour detection | `fen_builder.py` | Brightness analysis + FEN encoding |
| Engine analysis | `chess_advisor.py` | Stockfish via UCI |
| Orchestration | `main.py` | End-to-end pipeline (CLI) |
| Web interface | `app.py` | Streamlit web app |

---

## Setup and Running Instructions

### 1. Requirements

- Python 3.10 or newer
- Stockfish chess engine (executable)

### 2. Install dependencies

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Download the trained models

The trained CNN weights are hosted on the Hugging Face Hub:

**Models:** <https://huggingface.co/LucaSantomassimo/vica-chess-analyzer>

Download `chess_classifier_digital.pth` and `chess_classifier.pth` and place
them in the `models/` folder.

### 4. Add Stockfish

Download Stockfish from <https://stockfishchess.org/download/> and place the
executable at `engine/stockfish.exe`.

### 5. Run the analyzer (command line)

```bash
python src/main.py
```

The script prompts for the image filename, side to move, and board
orientation, then prints the detected FEN and Stockfish's recommendation.

### 6. Run the web app (optional)

```bash
streamlit run src/app.py
```

This opens a browser interface where you can upload screenshots, set the
game options, and view the analysis with alternative moves.

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
├── docs/
├── requirements.txt
└── README.md

---

## Model Training

The classifier was trained with transfer learning from an ImageNet-pretrained
ResNet-18, using a two-stage fine-tuning strategy.

**Training notebook (Google Colab):**
<https://colab.research.google.com/drive/1AloU4KelGsZWzucMlHSvW4gwhdM-6pz9?usp=sharing>

Two model variants were produced:
- Physical-piece model (~85% test accuracy)
- Digital-piece model (synthetic dataset)

---

## Performance Notes

- On clean in-domain screenshots, the pipeline reconstructs positions
  correctly and Stockfish returns sensible moves with alternatives.
- Accuracy degrades on out-of-domain inputs (different piece styles, photos
  of physical boards) due to **domain shift**, discussed in the report.

---

## Documentation

See `docs/VICA_Technical_Analysis.pdf` for the full project report: problem
description, methodology, experiments, failure analysis, and ethics.

---

## Ethical Note

VICA is intended for **post-game analysis, study, and training only**. Using
chess engine assistance during live rated games violates the rules of all
major chess platforms and federations.
