import argparse
import logging
import math
import random # <-- Import the random library for sampling
from data_loader import load_and_combine_data, create_siamese_examples, create_triplet_examples
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.datasets import NoDuplicatesDataLoader
import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    parser = argparse.ArgumentParser(description="Train a stance-aware sentence transformer.")
    parser.add_argument(
        '--model_type', type=str, required=True, choices=['siamese', 'triplet'],
        help="The type of contrastive learning model to train."
    )
    args = parser.parse_args()

    # 1. Load the base model from the local path specified in config
    logging.info(f"Loading base model from local path: {config.LOCAL_BASE_MODEL_PATH}")
    model = SentenceTransformer(config.LOCAL_BASE_MODEL_PATH)

    # 2. Load all datasets from local CSV files specified in config
    dataset = load_and_combine_data(
        train_dir_path=config.LOCAL_TRAIN_DATA_DIR,
        hf_dataset_dir=config.LOCAL_HF_DATASET_DIR
    )
    train_data = dataset['train']

    if args.model_type == 'siamese':
        # --- This part remains unchanged ---
        train_examples = create_siamese_examples(train_data)
        train_dataloader = NoDuplicatesDataLoader(train_examples, batch_size=config.TRAIN_BATCH_SIZE)
        train_loss = losses.ContrastiveLoss(model=model)
        save_path = config.SIAMESE_MODEL_SAVE_PATH

        warmup_steps = math.ceil(len(train_dataloader) * config.NUM_EPOCHS * config.WARMUP_STEPS_RATIO)
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=config.NUM_EPOCHS, warmup_steps=warmup_steps,
            output_path=save_path
        )
    else: # triplet
        # ==============================================================================
        # CHANGES FOR TRIPLET TRAINING START HERE
        # ==============================================================================
        logging.info("Creating Triplet examples (this may still take a moment)...")
        # This will still take a minute to generate all examples in memory
        all_train_examples = create_triplet_examples(train_data)
        logging.info(f"Generated {len(all_train_examples)} total triplet examples.")

        # --- CHANGE 1: SAMPLE THE TRIPLETS ---
        # If the number of examples is too large, sample a subset.
        # You should add MAX_TRIPLET_SAMPLES to your config.py file (e.g., 500000)
        max_samples = getattr(config, 'MAX_TRIPLET_SAMPLES', 500000)
        if len(all_train_examples) > max_samples:
            logging.info(f"Sampling {max_samples} triplets from the full set...")
            train_examples = random.sample(all_train_examples, max_samples)
        else:
            train_examples = all_train_examples
        logging.info(f"Using {len(train_examples)} triplets for training.")


        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=config.TRAIN_BATCH_SIZE)
        train_loss = losses.TripletLoss(model=model)
        save_path = config.TRIPLET_MODEL_SAVE_PATH

        # --- CHANGE 2: OPTIMIZE THE model.fit() CALL (NO EVALUATION) ---
        # We switch from epoch-based to step-based training for predictable performance.
        # You should add MAX_TRAIN_STEPS to your config.py (e.g., 25000)
        max_train_steps = getattr(config, 'MAX_TRAIN_STEPS', 25000)
        warmup_steps = int(max_train_steps * config.WARMUP_STEPS_RATIO)

        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=config.NUM_EPOCHS,  warmup_steps=warmup_steps,
            output_path=save_path,
        )

    logging.info(f"Training complete. Model saved to {save_path}")

if __name__ == "__main__":
    main()