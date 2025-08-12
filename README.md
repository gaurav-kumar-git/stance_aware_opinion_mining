

# Stance–Aware Transformers for Opinion Mining

This project explores **stance-aware sentence embeddings** for opinion mining – addressing a common limitation in standard sentence embeddings: they can detect topical similarity but often fail to distinguish between *opposing stances* (e.g., "pro" vs. "con").

We implement and compare two contrastive learning approaches – **Siamese Networks** and **Triplet Networks** – to fine-tune embeddings for stance awareness.

---

## 🚀 Key Features

- **Two Contrastive Architectures**
  - Siamese Network with `ContrastiveLoss`
  - Triplet Network with `TripletLoss`
- **Efficient Data Handling** via `data_loader.py`  
  Supports both:
  - Local directory of debate `.txt` files (training)
  - Hugging Face datasets (validation/test)
- **Smart Triplet Sampling**  
  Generates all possible triplets, then samples a subset (`MAX_TRIPLET_SAMPLES`) for efficiency.
- **Comprehensive Evaluation**  
  Produces cosine similarity KDE plots to measure stance separation.
- **Configurable & Reproducible**  
  All settings in `config.py`; environment defined in `environment.yml`.

---

## 📂 Repository Structure

````
stance\_aware\_opinion\_mining/
├── src/
│   ├── train.py              # Model training script
│   ├── evaluate.py           # Evaluation & visualization script
│   ├── data\_loader.py        # Data loading & preprocessing
│   ├── __init__.py
│   └── config.py             # Paths, hyperparameters, constants
├── environment.yml           # Conda environment specification
└── README.md                 # Project documentation

````

---

## 🛠 Getting Started

### 1️⃣ Setup Environment

```bash
conda env create -f environment.yml
conda activate stance_env
````

### 2️⃣ Train a Model

```bash
python src/train.py --model_type siamese
```

### 3️⃣ Evaluate

```bash
python src/evaluate.py --model_path results/models/siamese_model
```

---

## 📊 Results

Example cosine similarity distribution plots can be found in `results/plots/`:

* `baseline_similarity_dist.png`
* `baseline_similarity_dist_siamese_model.png`
* `baseline_similarity_dist_triplet.png`

---
