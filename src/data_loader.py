# In file: src/data_loader.py

import os
import re
import logging
from datasets import Dataset, DatasetDict, load_from_disk
from sentence_transformers import InputExample
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_train_from_debate_directory(directory_path: str) -> Dataset:
    # ... (This function is correct and does not need changes)
    logging.info(f"Parsing training data from debate directory: {directory_path}")
    if not os.path.isdir(directory_path):
        raise FileNotFoundError(f"The specified training directory does not exist: {directory_path}")
    parsed_debates = []
    for filename in tqdm(os.listdir(directory_path), desc="Parsing debate files"):
        if not filename.endswith(".txt"): continue
        file_path = os.path.join(directory_path, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        if not lines: continue
        discussion_title = lines[0].replace("Discussion Title:", "").strip()
        pro_statements, con_statements = [], []
        for line in lines[1:]:
            line = line.strip()
            if not line: continue
            pro_match = re.search(r'Pro:\s*', line)
            con_match = re.search(r'Con:\s*', line)
            if pro_match:
                statement = line[pro_match.end():].strip()
                if statement: pro_statements.append(statement)
            elif con_match:
                statement = line[con_match.end():].strip()
                if statement: con_statements.append(statement)
        if discussion_title and pro_statements and con_statements:
            parsed_debates.append({
                "discussion_title": discussion_title,
                "pro_arguments": pro_statements,
                "con_arguments": con_statements
            })
    if not parsed_debates:
        raise ValueError(f"No valid debates were found in '{directory_path}'.")
    return Dataset.from_list(parsed_debates)

def preprocess_hf_kialo_split(dataset_split: Dataset) -> Dataset:
    """
    Transforms a split from the Hugging Face Kialo dataset into our standard format.
    It unpacks the 'perspectives' column and renames 'question' to 'discussion_title'.
    """
    processed_rows = []
    for row in dataset_split:
        # The assignment specifies that for binary perspectives, the first is pro, the second is con.
        if len(row['perspectives']) >= 2:
            processed_rows.append({
                'discussion_title': row['question'],
                'pro_arguments': [row['perspectives'][0]],  # Wrap in a list to match train format
                'con_arguments': [row['perspectives'][1]]   # Wrap in a list
            })
    return Dataset.from_list(processed_rows)


def load_and_combine_data(train_dir_path: str, hf_dataset_dir: str) -> DatasetDict:
    """
    Loads data from all sources and unifies their schemas.
    """
    logging.info("--- Loading and combining data from all sources ---")
    try:
        # 1. Load the training data using our directory parser.
        train_split = load_train_from_debate_directory(train_dir_path)

        # 2. Load the raw validation/test data from disk.
        logging.info(f"Loading validation/test data from disk: '{hf_dataset_dir}'")
        raw_val_test_dataset = load_from_disk(hf_dataset_dir)

        # 3. Preprocess the validation and test splits to match the training schema.
        logging.info("Preprocessing validation and test splits to unify schemas...")
        processed_validation_split = preprocess_hf_kialo_split(raw_val_test_dataset['validation'])
        processed_test_split = preprocess_hf_kialo_split(raw_val_test_dataset['test'])

        # 4. Construct the final, unified DatasetDict.
        final_dataset = DatasetDict({
            'train': train_split,
            'validation': processed_validation_split,
            'test': processed_test_split
        })

        logging.info("All data splits loaded and combined successfully.")
        logging.info(f"Final dataset structure: {final_dataset}")
        return final_dataset

    except Exception as e:
        logging.error(f"Failed to load datasets. Please check paths in config.py. Error: {e}")
        raise

# --- Training Example Creation Functions (NO CHANGES NEEDED) ---
def create_siamese_examples(dataset_split):
    # ... no changes here
    examples = []
    logging.info("Creating Siamese (pairwise) examples...")
    for row in tqdm(dataset_split, desc="Creating Siamese pairs"):
        anchor = row['discussion_title']
        for pro in row['pro_arguments']: examples.append(InputExample(texts=[anchor, pro], label=1.0))
        for con in row['con_arguments']: examples.append(InputExample(texts=[anchor, con], label=0.0))
    logging.info(f"Created {len(examples)} Siamese examples.")
    return examples

def create_triplet_examples(dataset_split):
    # ... no changes here
    examples = []
    logging.info("Creating Triplet examples...")
    for row in tqdm(dataset_split, desc="Creating triplets"):
        anchor = row['discussion_title']
        for pro in row['pro_arguments']:
            for con in row['con_arguments']:
                examples.append(InputExample(texts=[anchor, pro, con]))
    logging.info(f"Created {len(examples)} Triplet examples.")
    return examples