import argparse
import logging
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer, util
from data_loader import load_and_combine_data

import config

def evaluate_model(model_path: str, output_plot_path: str, data_split: str = 'validation'):
    logging.info(f"--- Starting Evaluation for model: {model_path} on '{data_split}' split ---")

    # 1. Load the model
    model = SentenceTransformer(model_path)

    # 2. Load datasets from local CSVs specified in config
    dataset = load_and_combine_data(
        train_dir_path=config.LOCAL_TRAIN_DATA_DIR,
        hf_dataset_dir=config.LOCAL_HF_DATASET_DIR
    )
    # The new data loader makes the validation/test data have lists of arguments
    # We will just take the first one for evaluation simplicity.
    eval_data = dataset[data_split].map(lambda x: {
        'discussion_title': x['discussion_title'],
        'pro_argument': x['pro_arguments'][0],
        'con_argument': x['con_arguments'][0]
    })
    logging.info(f"Using {len(eval_data)} samples from the '{data_split}' set.")
    
    anchors = [row['discussion_title'] for row in eval_data]
    pros = [row['pro_argument'] for row in eval_data]
    cons = [row['con_argument'] for row in eval_data]

    anchor_embeddings = model.encode(anchors, batch_size=config.EVAL_BATCH_SIZE, show_progress_bar=True)
    pro_embeddings = model.encode(pros, batch_size=config.EVAL_BATCH_SIZE, show_progress_bar=True)
    con_embeddings = model.encode(cons, batch_size=config.EVAL_BATCH_SIZE, show_progress_bar=True)

    pro_similarities = util.cos_sim(anchor_embeddings, pro_embeddings).diag().numpy()
    con_similarities = util.cos_sim(anchor_embeddings, con_embeddings).diag().numpy()
    
    os.makedirs(os.path.dirname(output_plot_path), exist_ok=True)
    plt.figure(figsize=(10, 6))
    sns.kdeplot(pro_similarities, fill=True, label='Similarity(Anchor, Pro)')
    sns.kdeplot(con_similarities, fill=True, label='Similarity(Anchor, Con)')
    plt.title(f'Cosine Similarity Distribution\nModel: {os.path.basename(model_path)}')
    plt.xlabel('Cosine Similarity')
    plt.legend()
    plt.savefig(output_plot_path, dpi=300)
    plt.close()
    logging.info(f"Plot saved to {output_plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a stance-aware sentence transformer.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the model folder.")
    parser.add_argument('--output_filename', type=str, required=True, help="Filename for the output plot.")
    parser.add_argument('--split', type=str, default='validation', choices=['validation', 'test'], help="Data split to evaluate on.")
    args = parser.parse_args()
    full_plot_path = os.path.join(config.PLOTS_SAVE_PATH, args.output_filename)
    evaluate_model(model_path=args.model_path, output_plot_path=full_plot_path, data_split=args.split)