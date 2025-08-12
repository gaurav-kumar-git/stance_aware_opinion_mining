import os

LOCAL_BASE_MODEL_PATH = "/home/sslab/24m0786/stance_aware_opinion_mining/local_cache/all-mpnet-base-v2"
LOCAL_TRAIN_DATA_DIR = "/home/sslab/24m0786/stance_aware_opinion_mining/local_cache/train"
LOCAL_HF_DATASET_DIR = "/home/sslab/24m0786/stance_aware_opinion_mining/local_cache/kialo_offline_dataset"

# --- Training Hyperparameters (can be left as default) ---
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
NUM_EPOCHS = 1
LEARNING_RATE = 2e-5
WARMUP_STEPS_RATIO = 0.1

# triplet training specific
MAX_TRIPLET_SAMPLES = 500000


# --- Output Paths (can be left as default) ---
RESULTS_DIR = "results"
SIAMESE_MODEL_SAVE_PATH = os.path.join(RESULTS_DIR, "models", "siamese_model")
TRIPLET_MODEL_SAVE_PATH = os.path.join(RESULTS_DIR, "models", "triplet_model")
PLOTS_SAVE_PATH = os.path.join(RESULTS_DIR, "plots")
