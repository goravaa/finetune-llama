import os
import logging
from datasets import load_dataset, DatasetDict

def prepare_dataset(dataset_cfg, tokenizer):
    """
    Prepare a Hugging Face DatasetDict for language model fine-tuning.

    Args:
        dataset_cfg (dict): Configuration dict with keys:
            - path (str): Path to a CSV file, directory of CSVs, or HF dataset ID.
            - text_column (str): (Optional) Single column name for text. Defaults to None.
            - text_columns (list): (Optional) List of column names to merge as text.
            - test_size (float): Fraction for validation split if no split exists. Defaults to 0.1.
            - batch_size (int): Batch size for tokenization mapping. Defaults to 1000.
        tokenizer (PreTrainedTokenizer): Hugging Face tokenizer instance.

    Returns:
        DatasetDict: Tokenized dataset with 'train' and 'validation' splits,
                     formatted as PyTorch tensors with 'input_ids' and 'attention_mask'.
    """
    logger = logging.getLogger(__name__)
    path = dataset_cfg['path']
    logger.info(f"🚀 Loading dataset from: {path}")

    # Load dataset from CSV(s) or HF hub
    try:
        if os.path.isdir(path):
            files = sorted([os.path.join(path, f) for f in os.listdir(path) if f.endswith('.csv')])
            ds = load_dataset('csv', data_files={'train': files})
        elif path.endswith('.csv'):
            ds = load_dataset('csv', data_files={'train': path})
        else:
            ds = load_dataset(path)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise

    # Ensure validation split
    if 'validation' not in ds and 'test' not in ds:
        test_size = dataset_cfg.get('test_size', 0.1)
        logger.info(f"No validation/test split—splitting train by {test_size}")
        split = ds['train'].train_test_split(test_size=test_size)
        ds = DatasetDict({'train': split['train'], 'validation': split['test']})
    elif 'test' in ds and 'validation' not in ds:
        ds = ds.rename_column('test', 'validation')

    # Determine which columns to use as text
    train_cols = ds['train'].column_names
    user_cols = dataset_cfg.get('text_columns')
    single_col = dataset_cfg.get('text_column')

    if user_cols:
        text_cols = user_cols
    elif single_col and single_col in train_cols:
        text_cols = [single_col]
    elif 'text' in train_cols:
        text_cols = ['text']
    elif set(['instruction','input','output']).issubset(train_cols):
        text_cols = ['instruction', 'input', 'output']
    else:
        # auto-detect all string-type columns from a sample row
        sample = ds['train'][0]
        text_cols = [c for c, v in sample.items() if isinstance(v, str)]
    logger.info(f"🧠 Using text columns: {text_cols}")

    # Tokenization function: merge multiple fields if needed
    def tokenize_fn(examples):
        if len(text_cols) == 1:
            texts = examples[text_cols[0]]
        else:
            texts = []
            n = len(examples[text_cols[0]])
            for i in range(n):
                parts = []
                for col in text_cols:
                    val = examples[col][i]
                    if val is not None:
                        parts.append(f"{col.capitalize()}: {val}")
                texts.append("\n".join(parts))
        return tokenizer(texts, truncation=True, padding=False)

    batch_size = dataset_cfg.get('batch_size', 1000)
    logger.info(f"🛠️ Tokenizing ({batch_size} examples per batch)...")
    ds = ds.map(
        tokenize_fn,
        batched=True,
        batch_size=batch_size,
        remove_columns=train_cols,
        desc="Tokenizing dataset"
    )

    # Set PyTorch format
    ds.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    logger.info("✅ Dataset ready: columns=input_ids, attention_mask")

    return ds
