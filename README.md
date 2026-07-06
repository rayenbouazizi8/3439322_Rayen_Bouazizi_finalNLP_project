# BERT vs DistilBERT for Sentiment Classification on IMDb

Final project for Python for NLP. Compares TF-IDF + Logistic Regression,
DistilBERT, and BERT-base on binary sentiment classification using the
IMDb Movie Reviews dataset.

**Research question:** Does DistilBERT offer a worthwhile trade-off between
predictive performance and computational efficiency compared to BERT-base?

## Author
Rayen Bouazizi — 3439322

## Environment
- Google Colab (GPU runtime, T4)
- Python 3.12
- Key dependencies (see first notebook cell for exact versions):
  - `transformers==4.47.0`
  - `datasets==3.1.0`
  - `accelerate>=0.26.0`
  - `evaluate>=0.4.0`
  - `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`

## Data
IMDb Movie Reviews (Maas et al., 2011), loaded via HuggingFace `datasets`
(`load_dataset("imdb")`). 25,000 train / 25,000 test, split further into
22,500 train / 2,500 validation / 25,000 test. No manual download needed —
the notebook fetches it automatically.

## Reproducing the experiments
1. Open `NLP_Project.ipynb` in Google Colab with a GPU runtime.
2. Run all cells top to bottom. Sections are:
   - Setup & reproducibility (random seed = 42)
   - TF-IDF + Logistic Regression baseline
   - Tokenization (shared config: `MAX_LENGTH=256`, `BATCH_SIZE=16`)
   - DistilBERT fine-tuning (3 epochs, lr=2e-5)
   - BERT-base fine-tuning (identical hyperparameters)
   - Model comparison (Section 7)
   - Error analysis (Section 8)
   - Conclusion (Section 9)
3. Total runtime: approximately 80-90 minutes on a T4 GPU (26.5 min
   DistilBERT + 52.8 min BERT + baseline/analysis overhead).

## Key hyperparameters
| Parameter | Value |
|---|---|
| Max sequence length | 256 |
| Batch size | 16 |
| Learning rate | 2e-5 |
| Epochs | 3 |
| Random seed | 42 |

## Results summary
| Model | Accuracy | Macro F1 | Train time | Inference (ms/sample) |
|---|---|---|---|---|
| TF-IDF + LR | 89.85% | 89.85% | ~2s | 0.001ms |
| DistilBERT | 91.36% | 91.36% | 26.5 min | 6.96ms |
| BERT-base | 92.22% | 92.22% | 52.8 min | 13.72ms |

Full results, comparison charts, and error analysis are in the notebook
and accompanying report (`[filename].pdf`).

## AI usage
See the Declaration of AI Usage section in the report PDF.
