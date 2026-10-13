# Domain Adaptation and Conservative Hypothesis Consensus for Code-Switched Indonesian–Javanese Speech Recognition

[![Paper](https://img.shields.io/badge/Paper-IEEE_Format_PDF-red)](output/pdf/Indonesian_Javanese_ASR_IEEE_Paper.pdf)
[![Competition](https://img.shields.io/badge/DrivenData-34th_%2F_500_Teams-blue)](LOST_IN_TRANSCRIPTION_CASE_STUDY.md)
[![Python](https://img.shields.io/badge/Python-3.10%2B-brightgreen)](requirements.txt)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](requirements.txt)
[![HuggingFace](https://img.shields.io/badge/🤗_Transformers-PEFT_%2F_LoRA-yellow)](https://huggingface.co/)

**Aakash Balasubramanian and Aniruth Devaraj**  
*Department of Computer Science and Engineering, Sri Sivasubramaniya Nadar College of Engineering, Chennai, India*

---

## Executive Summary

Spontaneous bilingual Indonesian–Javanese speech presents severe challenges for automatic speech recognition: within-utterance language switching, colloquial vocabulary, disfluencies, long audio clips (>30s), and variable spelling conventions. 

This repository contains the complete codebase, data audit, ablation studies, and technical report for our solution to DrivenData's **Lost in Transcription** competition. 

### Key Highlights
- **The Forensic Pivot:** Moving away from a brittle 5-model ensemble (Whisper, Qwen, XLS-R, MERaLiON, BuzzASR) that collapsed from ~0.2058 dev WER to **0.4009** on the hidden test set.
- **Anchor Adaptation:** Fine-tuning Whisper models with parameter-efficient Low-Rank Adaptation (LoRA) under conversation-disjoint cross-validation folds.
- **Conservative Hypothesis Consensus:** Rather than naive voting or unconstrained ROVER, we introduce a bounded consensus policy: keeping an adapted Whisper model as the authoritative anchor and accepting substitutions **only** when multiple alternative hypotheses unambiguously agree on the replacement.
- **Competition Result:** The conservative Whisper Large-v3-Turbo + MERaLiON consensus system achieved final leaderboard values of **0.2712 / 0.2743**, earning **34th place out of 500 entries** (executed in 30.7 minutes on a single CUDA GPU).
- **Subsequent Research:** A follow-on offline study fine-tuned Whisper Large-v3 with fold weight averaging (model soup) and deterministic word consensus, reducing aggregate conversation-disjoint out-of-fold (OOF) WER down to **0.1939** (a 3.0% relative improvement).

---

## System Architecture

```
                    ┌────────────────────────────────────────────────────────┐
                    │ Raw Audio Data (Indonesian Dev + Jember Speech Corpus) │
                    └───────────────────────────┬────────────────────────────┘
                                                │
                                                ▼
                    ┌────────────────────────────────────────────────────────┐
                    │ Data Audit & Preprocessing (16 kHz mono, lossless      │
                    │ merging into 29s chunks, diacritic normalization)     │
                    └───────────────────────────┬────────────────────────────┘
                                                │
                     Conversation-Disjoint Split (Fold A / Fold B)
                                                │
                        ┌───────────────────────┴───────────────────────┐
                        ▼                                               ▼
         ┌─────────────────────────────┐                 ┌─────────────────────────────┐
         │     Primary Anchor ASR      │                 │    Supporting Hypothesis    │
         │  Whisper Large-v3 / Turbo   │                 │      MERaLiON-3-3B ASR      │
         │  (LoRA r=32/16 on Attn+FFN) │                 │   (Decoder-only LoRA r=16)  │
         └──────────────┬──────────────┘                 └──────────────┬──────────────┘
                        │                                               │
                        │           Primary Anchor Hypothesis           │  Supporting
                        │                   (y_anchor)                  │  Hypothesis
                        └───────────────────────┬───────────────────────┘
                                                │
                                                ▼
                        ┌───────────────────────────────────────────────┐
                        │      Conservative Hypothesis Consensus        │
                        │                                               │
                        │  • Compute minimum-cost word alignment        │
                        │  • Find unambiguous 1-to-1 mappings           │
                        │  • Apply edit ONLY if supporters agree        │
                        │  • Zero ordinary insertions or deletions      │
                        │  • Surface repetition safeguard               │
                        └───────────────────────┬───────────────────────┘
                                                │
                                                ▼
                                    ┌───────────────────────┐
                                    │ Final Word Sequence ŷ │
                                    └───────────────────────┘
```

---

## Performance Progression

### 1. Local Conversation-Disjoint Development Results (372 Clips)

The target development set contains 372 clips across two conversations. Folds A and B evaluate cross-conversation generalization. Aggregate WER is computed from pooled token counts across folds:

$$\WER_{\mathrm{agg}} = \frac{\sum_{k}(S_k + D_k + I_k)}{\sum_{k}N_k}$$

| System / Progression Stage | Fold A WER | Fold B WER | Aggregate WER | Notes |
| :--- | :---: | :---: | :---: | :--- |
| **Initial 5-Model Replay (ROVER)** | — | — | 0.2058 | Brittle broad ensemble; failed on hidden test (0.4009) |
| **Whisper Turbo (Ratio-3, Half Epoch)** | 0.2236 | 0.2042 | 0.2197 | Single-anchor baseline |
| **Adapted MERaLiON-3 (Decoder LoRA)** | 0.2266 | 0.2208 | 0.2254 | Standalone supporting model |
| **Simple Turbo + MERaLiON Fusion** | 0.2185 | 0.2151 | 0.2178 | Naive fusion regressed on Fold B |
| **Three-Output Conservative Fusion (Submitted)** | **0.2098** | **0.1853** | **0.2048** | Strict substitution-only policy |
| **Whisper Large-v3 Standalone** | 0.2055 | 0.1780 | 0.1999 | Follow-on research anchor |
| **Whisper Large-v3 Unambiguous Consensus** | **0.1994** | **0.1727** | **0.1939** | **Best local research result** (3.0% relative gain) |

### 2. Competition Leaderboard Progression

| Submission Stage | Displayed Score Col 1 | Displayed Score Col 2 | Outcome |
| :--- | :---: | :---: | :--- |
| Early Multi-Model Exploration | 0.4273 | 0.4234 | Initial exploratory submission |
| Intermediate Iteration | 0.4140 | 0.4123 | Early ensemble refinements |
| Legacy 5-Model ROVER Ensemble | 0.4009 | 0.3963 | Generalization breakdown on hidden set |
| Single Adapted Whisper Turbo | 0.2862 | 0.2914 | **+0.1147 WER recovery** via single adapted anchor |
| Three-Model Protected Fusion | 0.2786 | 0.2837 | Guarded multi-hypothesis integration |
| **Final Conservative Turbo Fusion** | **0.2712** | **0.2743** | **Final Competition Rank: 34th of 500 entries** |

---

## Methodology

### 1. Data Audit & Lossless Regrouping
- **Supplementary Jember Corpus:** Audited 208 session recordings (6,679 annotation rows). Rejected 41 corrupted rows. Losslessly merged adjacent intra-session utterances into 1,471 long-form chunks (median 26s, max 29s), matching the target clip duration distribution.
- **Orthographic Normalization:** Addressed spelling divergence between Jember and the competition set (stripping non-phonemic pepet/taling diacritics `ê è ě` while preserving dialectal word identities and casing/hyphenation).

### 2. Parameter-Efficient Fine-Tuning (PEFT)
- **Whisper Large-v3-Turbo:** LoRA applied to all query, key, value, and projection modules ($r=32, \alpha=64$, dropout 0.05). Trainable parameters: 27.85M / 836.73M (3.3%). Ratio-3 competition-to-Jember upsampling with AdamW ($3 \times 10^{-4}$ learning rate, 8% warmup, cosine decay).
- **MERaLiON-3-3B-ASR:** Frozen speech encoder and audio projector; decoder-only LoRA ($r=16, \alpha=32$).
- **Whisper Large-v3 (Offline Research):** LoRA ($r=16, \alpha=32$) trained with ratio-4 sampling, followed by uniform weight averaging (model soup) across fold checkpoints.

### 3. Conservative Hypothesis Consensus Rules
Rather than allowing an alternative model to freely overwrite the primary transcript:
1. **Authoritative Anchor:** Whisper's hypothesis forms the base surface text.
2. **Deterministic Minimum-Cost Alignment:** Word-level Levenshtein alignment establishes position mappings.
3. **Unambiguous Agreement:** An edit is executed if and only if **both** alternative hypotheses map to the same target word across all optimal alignments.
4. **Zero Insertions/Deletions:** Only substitutions are permitted. Spurious insertions and deletions that cause hallucinations or word drops are strictly rejected.
5. **Surface Repetition Guard:** A deterministic safeguard catches and prunes unprompted decoding loops prior to consensus.

---

## Ablation Study

| Consensus Variant | Aggregate WER | Edit Profile | Takeaway |
| :--- | :---: | :--- | :--- |
| **Large-v3 Standalone Anchor** | 0.1999 | No edits | Baseline reference |
| **+ Short Insertion/Deletion Edits** | 0.1959 | 210 subst, 14 ins, 19 del | Additional edit operators worsened WER |
| **+ Two Whisper Supporters (No MERaLiON)** | 0.2002 | Strict support rule | Removing model diversity regressed performance |
| **+ Learned Logistic Edit Gate** | 0.1958 | Cross-fold logistic regression | ML gating was weaker than deterministic rules |
| **+ Targeted N-Best Rescoring** | 0.1934 | 305 subst, 2 ins, 8 del | Below statistical gain threshold; added complexity |
| **+ Unambiguous Boundary Consensus (Selected)** | **0.1939** | **277 subst, 0 ins, 0 del** | **Cleanest risk-controlled consensus policy** |

---

## Repository Structure

```
.
├── README.md                                  # You are here: paper summary & project hub
├── LOST_IN_TRANSCRIPTION_CASE_STUDY.md        # Comprehensive 75 KB portfolio case study
├── FINAL_9H_REPORT.md                         # Final 9-hour attack execution & evidence log
├── MODEL_ARTIFACTS.md                         # Inventory of model artifacts, weights & licenses
├── Preprocessing_EDA.ipynb                    # Consolidated, reproducible data audit notebook
├── requirements.txt                           # Python dependencies
│
├── output/pdf/                                # Research Paper & Presentations
│   ├── Indonesian_Javanese_ASR_IEEE_Paper.pdf # Full 6-page IEEE format paper
│   ├── Indonesian_Javanese_ASR_IEEE_Paper.tex # LaTeX source for the paper
│   ├── LOST_IN_TRANSCRIPTION_PRESENTATION.pdf # Defense & presentation deck
│   └── paper_assets/                          # Figures and spectrogram assets
│
├── docs/                                      # Detailed Technical Documentation
│   ├── case_study/                            # Evidence charts, logs, and score progressions
│   ├── eda_and_data/                          # Data audit, risk logs, and preprocessing contracts
│   └── legacy_ensemble/                       # Forensic analysis of the 5-model ensemble failure
│
├── final_9h_attack/                           # Final Large-v3 Research Pipeline
│   ├── models/selected_large/                 # Final Whisper Large-v3 model + LoRA delta
│   ├── runtime/                               # Consensus, alignment & repetition safeguards
│   ├── main.py                                # Inference entrypoint
│   └── evaluate_model.py                      # Out-of-fold scoring & validation
│
├── phase1_whisper_anchor/                     # Phase 1: Whisper Turbo LoRA Anchor Pipeline
├── phase2_meralion/                           # Phase 2: MERaLiON Complementarity & Decoder LoRA
├── phase3_protected_fusion/                   # Phase 3: Three-Model Protected Fusion Pipeline
├── phase4_stronger_anchor/                    # Phase 4: Extended training recipes
├── overnight_attack/                          # Large-v3 overnight training & soup recipes
├── scripts/                                   # Preprocessing, scoring, and routing utilities
├── src/                                       # Core preprocessing library
└── tests/                                     # Automated test suites
```

---

## Quickstart & Reproducibility

### 1. Environment Setup
```bash
git clone https://github.com/06-aakash-06/lost-in-transcription.git
cd lost-in-transcription

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Preprocessing & Data Validation
Execute the consolidated preprocessing pipeline:
```bash
python scripts/build_preprocessing.py
python scripts/validate_preprocessing.py
```

### 3. Evaluate the Final Large-v3 Research Candidate
Run local conversation-disjoint evaluation on the target development set:
```bash
python final_9h_attack/evaluate_model.py \
  --model final_9h_attack/models/selected_large \
  --manifest phase1_whisper_anchor/data/fold_A_valid.tsv \
  --device mps  # or cuda / cpu
```

### 4. Run Unit Tests
```bash
pytest tests/
```

---

## Citation

If you find this work, case study, or methodology useful, please cite our research paper:

```bibtex
@article{balasubramanian2026domain,
  title   = {Domain Adaptation and Conservative Hypothesis Consensus for Code-Switched Indonesian--Javanese Speech Recognition: A Competition Case Study},
  author  = {Balasubramanian, Aakash and Devaraj, Aniruth},
  journal = {Department of Computer Science and Engineering, Sri Sivasubramaniya Nadar College of Engineering},
  year    = {2026},
  url     = {https://github.com/06-aakash-06/lost-in-transcription}
}
```

---

## License

- Source code is licensed under the [Apache License 2.0](final_9h_attack/APACHE_LICENSE.txt).
- Pretrained model components are subject to their respective upstream licenses: [OpenAI Whisper License](phase1_whisper_anchor/WHISPER_LICENSE.txt) and the [MERaLiON-3 Public License](phase2_meralion/MERaLiON-3-Public-Licence.txt).
