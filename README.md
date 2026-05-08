# Unsupervised Neural Network for Multi-Genre Music Generation
**Course:** CSE425/EEE474 — Neural Networks  
**University:** BRAC University  
**Section:** 5  

---

## 📌 Project Overview
This project implements and compares four unsupervised deep learning models for symbolic music generation using the [MAESTRO Dataset](https://magenta.tensorflow.org/datasets/maestro). All models are trained on piano-roll representations of MIDI files without explicit genre labels.

---

## 👥 Group Members & Contributions

| Name | Student ID | Contribution |
|------|-----------|--------------|
| Shehraj Nayeemn Khan| 24241111 | Task 1 (LSTM Autoencoder), Task 3 (Transformer), EDA & Preprocessing |
| Mahir Jawad Chowdhury | 22301662 | Task 2 (VAE), Baseline Models, Evaluation Metrics, Report |

---

## 📁 Repository Structure

```
music-generation-unsupervised/
│
├── notebook/
│   ├── 0_EDA_and_Preprocessing.ipynb   # Dataset loading, pitch/duration histograms, piano-roll generation
│   ├── 1_RNN_Model.ipynb               # Task 1: LSTM Autoencoder — trains on piano-rolls, generates 5 MIDI samples
│   ├── 2_VAE_Model.ipynb               # Task 2: Variational Autoencoder — KL annealing, latent interpolation, 8 MIDI samples
│   ├── 3_Transformer_Model.ipynb       # Task 3: Transformer with causal mask — perplexity evaluation, 10 MIDI samples
│   └── 4_Evaluation_and_Baselines.ipynb# Baseline models (Random, Markov Chain) + full comparison table + human survey
│
├── src/
│   ├── config.py                       # All hyperparameters and path settings in one place
│   ├── preprocessing.py                # MIDI loading, piano-roll conversion, sequence creation
│   └── metrics.py                      # Pitch histogram similarity, rhythm diversity, repetition ratio, perplexity
│
├── data/
│   └── maestro-v3.0.0/                 # MAESTRO dataset (not uploaded — download from link below)
│
└── README.md
```

---

## 🎵 Generated MIDI Files
All generated MIDI files are available on Google Drive:  
🔗 **[MIDI Files — Google Drive Link](https://drive.google.com/drive/folders/1nZRD9wijyzt_5jil5UBIeknxzqdVa8Vj?usp=sharing)**

| Model | Files |
|-------|-------|
| Task 1: RNN Autoencoder | `RNN_Generated_Sample_1–5.mid` |
| Task 2: VAE | `VAE_Generated_Sample_1–8.mid` |
| Task 3: Transformer | `Transformer_Generated_Sample_1–10.mid` |

Listen online: [pianotify.com](https://pianotify.com/import-midi-file)

---

## 📊 Dataset
- **MAESTRO v3.0.0** — Classical piano MIDI recordings
- Download: https://magenta.tensorflow.org/datasets/maestro
- Place the extracted folder at `data/maestro-v3.0.0/`

---

## ⚙️ Installation

```bash
pip install torch pretty_midi pandas numpy matplotlib
```

---

## 🚀 How to Run

Run the notebooks **in order**:

```
0_EDA_and_Preprocessing.ipynb  →  1_RNN_Model.ipynb  →  2_VAE_Model.ipynb
→  3_Transformer_Model.ipynb  →  4_Evaluation_and_Baselines.ipynb
```

---

## 📈 Results Summary

| Model | Pitch Sim (↓) | Rhythm Diversity | Repetition Ratio |
|-------|-------------|-----------------|-----------------|
| Random Generator | 1.394 | 0.002 | 0.000 |
| Markov Chain | 0.324 | 0.016 | 0.036 |
| Task 1: RNN AE | 0.568 | 0.114 | 1.000 |
| Task 2: VAE | 1.521 | 0.667 | 0.979 |
| Task 3: Transformer | 1.472 | 0.818 | 0.988 |

---

📄 Report
🔗 **[Final Report PDF — Google Drive](https://drive.google.com/file/d/1Ovkjok4mDLCyguBaz1d2yr65FF8XkKkV/view?usp=sharing)**

📊 Loss Curves & Evaluation Plots
🔗 **[Plots — Google Drive](https://drive.google.com/drive/folders/1yYYqZ4pl9pb7jXcaTdp4aAZwPBd4ZVrj?usp=sharing)**

---

## 🎬 Presentation Video
🔗 **[YouTube Video Link](https://youtu.be/vt8FTzvTQrM)**
