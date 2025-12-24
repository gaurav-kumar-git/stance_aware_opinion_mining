import argparse
import logging
import math
import random 
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

    logging.info(f"Loading base model from local path: {config.LOCAL_BASE_MODEL_PATH}")
    model = SentenceTransformer(config.LOCAL_BASE_MODEL_PATH)

    dataset = load_and_combine_data(
        train_dir_path=config.LOCAL_TRAIN_DATA_DIR,
        hf_dataset_dir=config.LOCAL_HF_DATASET_DIR
    )
    train_data = dataset['train']

    if args.model_type == 'siamese':
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
    else:
        logging.info("Creating Triplet examples (this may still take a moment)...")
        all_train_examples = create_triplet_examples(train_data)
        logging.info(f"Generated {len(all_train_examples)} total triplet examples.")

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
