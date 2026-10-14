# Faculty Review & Collaboration Handbook
## Indonesian–Javanese Code-Switched Automatic Speech Recognition

Welcome! This handbook is specifically prepared for faculty advisors, collaborators, and peer reviewers to understand, run, test, and evaluate the entire end-to-end research pipeline and discuss paths toward conference/journal publication.

---

## 1. Quick Orientation

- **Research Problem:** Spontaneous, conversational code-switching between Indonesian and Javanese.
- **Key Breakthrough:** Moving away from brittle, unconstrained multi-model ensembles (which catastrophically failed on hidden test data, dropping from ~0.20 to **0.4009 WER**) to **LoRA Parameter-Efficient Adaptation** with **Conservative Hypothesis Consensus** (unambiguous, risk-controlled word substitution).
- **Competition Result:** **34th place out of 500 entries** in DrivenData's *Lost in Transcription* benchmark.
- **Offline Research Result:** Fine-tuned Whisper Large-v3 with unambiguous consensus reduced conversation-disjoint Out-of-Fold (OOF) WER to **0.1939** (a **3.0% relative improvement**, verified with paired Wilcoxon $p = 1.12 \times 10^{-9}$ and bootstrap 95% CI $[+0.429\%, +0.787\%]$).
- **Full IEEE Format Paper:** `output/pdf/Indonesian_Javanese_ASR_IEEE_Paper.pdf` (6-page draft).

---

## 2. Interactive Master Notebook (`LostInTranscription_Pipeline.ipynb`)

For a visual, interactive walkthrough with in-notebook audio players, real-time transcription, Levenshtein alignment inspection, and bootstrap significance plots:
```bash
jupyter lab LostInTranscription_Pipeline.ipynb
# or open directly in VS Code / Cursor / Google Colab
```
*Note:* The notebook comes with pre-rendered outputs, figures, and audio widgets, so you can inspect everything immediately even before running.

---

## 3. Dataset Verification & Acquisition Helper

To verify or acquire the full datasets:
```bash
# Verify if datasets are present and healthy
python run_pipeline.py --stage audit
```
*(The master notebook `LostInTranscription_Pipeline.ipynb` also embeds interactive download instructions and automated helpers directly in Section 2).*  
*Zero-Download Option:* You do not need to download the 10GB dataset to test the models—lightweight sample clips are already bundled in `samples/` for immediate execution.

---

## 4. Immediate CLI Demo (Runs in ~15s)

To test speech recognition and WER calculation on bundled or custom audio:
```bash
# Transcribe bundled sample clip (~26s)
python demo.py

# Transcribe all bundled samples
python demo.py --all-samples

# Transcribe any arbitrary audio file:
python demo.py --audio path/to/your_audio.wav
```

---

## 5. The Unified Research Pipeline (`run_pipeline.py`)

Rather than navigating fragmented subdirectories, the entire workflow is controlled through [`run_pipeline.py`](file:///Users/aakash/Documents/Models%20and%20Research/LostinTranscription/run_pipeline.py):

| Stage | Command | Purpose |
| :--- | :--- | :--- |
| **0. Smoke Test** | `python run_pipeline.py --stage smoke` | 30s dry run verifying environment, models, and runtime |
| **1. Audit** | `python run_pipeline.py --stage audit` | Checks presence of raw corpora, durations, sample rates, and sessions |
| **2. Preprocess** | `python run_pipeline.py --stage preprocess` | Regroups utterances losslessly into 29s chunks and builds disjoint folds |
| **3. Evaluate** | `python run_pipeline.py --stage evaluate` | Computes clip-level and aggregate Out-of-Fold WER |
| **4. Consensus** | `python run_pipeline.py --stage consensus` | Runs conservative hypothesis consensus and substitution logs |
| **5. Significance** | `python run_pipeline.py --stage significance` | Computes paired Wilcoxon tests and 1,000-round bootstrap 95% CIs |
| **All** | `python run_pipeline.py --stage all` | Executes the complete workflow sequentially |

---

## 4. Academic Publication Strategy (Conference / Journal)

The accompanying paper (`output/pdf/Indonesian_Javanese_ASR_IEEE_Paper.pdf`) provides a solid foundation. Below are concrete avenues and additions to target acceptance at top speech and NLP venues:

### Recommended Target Venues

1. **Interspeech** (Premier Speech Science & Technology Conference)
   - *Format:* 4 pages text + 1 page references.
   - *Fit:* Dedicated tracks on *Low-Resource & Code-Switched Speech* and *ASR System Combination*.
   - *Core Narrative:* The "Forensic Pivot"—why standard ROVER collapses under spontaneous code-switching, and how bounded hypothesis consensus guarantees monotonic risk control.

2. **IEEE ICASSP** (IEEE International Conference on Acoustics, Speech and Signal Processing)
   - *Format:* 5 pages total.
   - *Fit:* Strong focus on statistical speech modeling and adaptation algorithms.

3. **Computer Speech & Language (CSL, Elsevier Journal)**
   - *Format:* Full-length journal paper (10–14 pages).
   - *Fit:* Welcomes comprehensive case studies combining competition engineering, ablation studies, and qualitative phonetic/linguistic analysis.

4. **IEEE/ACM Transactions on Audio, Speech, and Language Processing (TASLP)**
   - *Format:* Comprehensive journal article.
   - *Fit:* High-impact archival journal.

### Concrete Additions for Publication Acceptance

| Area | What Is Done | Recommended Next Step with Faculty |
| :--- | :--- | :--- |
| **Statistical Rigor** | Bootstrap 95% CI $[+0.429\%, +0.787\%]$ and Wilcoxon $p = 1.12 \times 10^{-9}$ established. | Report per-speaker / per-conversation confidence intervals. |
| **Linguistic Analysis** | General substitution vs deletion profiles logged. | Categorize errors into: *(a)* Javanese vocabulary, *(b)* code-switch transition boundaries, *(c)* orthographic variants (pepet/taling). |
| **Baseline Comparisons** | Compared against Standalone Anchor, N-best rescoring, and learned gates. | Explicitly report standard ROVER baseline numbers on the same local dev set to illustrate the contrast. |
| **Model Hosting** | Weights verified locally; adapters (~110MB) can be hosted on Hugging Face Hub. | Publish public HF Hub checkpoint (e.g. `ssn-speech/whisper-large-v3-javind-lora`) to earn the ACM/Interspeech Reproducibility Badge. |

---

## 5. Verification Checklist

To verify the codebase is 100% operational before your review meeting:
```bash
# 1. Run unit test suite (62 tests passed)
pytest tests/

# 2. Run statistical significance verification
python scripts/evaluate_significance.py

# 3. Verify local model weights
python scripts/download_weights.py --verify
```

---

*Prepared by Aakash Balasubramanian and Aniruth Devaraj, Department of Computer Science and Engineering, SSN College of Engineering.*
