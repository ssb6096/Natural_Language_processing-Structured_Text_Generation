# NLP for Structured Text Generation: Song Lyrics in an Artist's Style

Generating new song lyrics that resemble an artist's style **without memorizing** their existing lines. Because any one artist's catalogue is a small corpus, the project uses **transfer learning** with LSTM language models.

📑 **[Project poster (PDF)](docs/NLP_Structured_Text_Generation_Poster.pdf)**

---

## Overview

Sometimes we want to hear songs in a specific style, but the band has stopped writing new ones. Generating lyrics that are similar to, but not the same as, an artist's existing songs could solve this. A single artist's songs make a small training set, so both models start from a pre-trained language model and are then fine-tuned on song lyrics.

| | Model 1 | Model 2 |
|---|---|---|
| **Pre-trained model** | 3-layer **AWD-LSTM** trained on 100 million tokens of Wikipedia (fastai) | 1-layer **LSTM** trained on a corpus of Nietzsche's writing (Keras) |
| **Fine-tuning** | Trained on the song-lyrics dataset, freezing and training only certain layers | Trained on the song-lyrics dataset for 40 epochs to retain vocabulary |
| **Advantage** | Good vocabulary | Learns grammar and sentence structure well |
| **Drawback** | Still needs a larger amount of training data | Vocabulary not as good as the first model |

**Result.** Both models produced unique text that was not memorized from the corpus. Future work: better grammar and sentence structure, and generating full, well-structured songs.

## Skills and tools

`Python` · `Deep learning` · `LSTM / AWD-LSTM` · `Transfer learning` · `fastai` · `Keras` · `NLP`

## Repository contents

All code is in `MachineLearningProject/`:

| File(s) | What it is |
|---|---|
| `MLProjectLSTM.py` | Character-level LSTM text generator (Keras), fine-tuned on lyrics |
| `MLProject1.py` – `MLProject4.py` | fastai AWD-LSTM transfer-learning experiments |
| `lstm_model.json`, `lstm_model.h5` | Trained LSTM architecture and weights |

## Context

Project from my M.S. in Electrical Engineering at Rochester Institute of Technology. Also on [Portfolium](https://portfolium.com/entry/natural-language-processing-for-structured-text-ge).
