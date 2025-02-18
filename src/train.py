# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset
# import sentencepiece as spm
# from transformer import SimpleTransformer  # Import your model

# # ✅ Load tokenizer
# sp = spm.SentencePieceProcessor()
# sp.load("my_ai_model/models/tokenizer.model")
# vocab_size = sp.GetPieceSize()
# pad_id = sp.pad_id() if sp.pad_id() >= 0 else vocab_size  # Handle no pad_id case

# # ✅ Load and preprocess text dataset from file
# def load_text_from_file(file_path, seq_length=10):
#     with open(file_path, "r", encoding="utf-8") as f:
#         text_data = f.read().strip().split("\n")  # Read sentences from file

#     tokenized_data = []
#     for line in text_data:
#         tokens = sp.EncodeAsIds(line.strip())
#         tokens = [t if t < vocab_size else 0 for t in tokens]  # Avoid index error
#         if len(tokens) < seq_length:
#             tokens += [pad_id] * (seq_length - len(tokens))  # Padding
#         else:
#             tokens = tokens[:seq_length]  # Truncate
#         tokenized_data.append(tokens)

#     return torch.tensor(tokenized_data, dtype=torch.long)

# # ✅ Set dataset path
# data_file = "my_ai_model/data/words.txt"
# data = load_text_from_file(data_file)

# # ✅ Shift labels for next-token prediction
# targets = torch.cat(
#     [data[:, 1:], torch.full((data.shape[0], 1), pad_id, dtype=torch.long)], dim=1
# )

# # ✅ Create DataLoader
# batch_size = 32
# dataset = TensorDataset(data, targets)
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# # ✅ Model setup
# embed_dim = 128
# num_heads = 4
# hidden_dim = 256
# num_layers = 4
# num_epochs = 10
# learning_rate = 0.001
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = SimpleTransformer(vocab_size, embed_dim, num_heads, hidden_dim, num_layers).to(device)
# criterion = nn.CrossEntropyLoss(ignore_index=pad_id)  # Ignores padding tokens
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None  # Mixed precision training

# # ✅ Training loop
# for epoch in range(num_epochs):
#     total_loss = 0
#     model.train()

#     for batch in dataloader:
#         inputs, labels = batch
#         inputs, labels = inputs.to(device), labels.to(device)

#         optimizer.zero_grad()
        
#         with torch.cuda.amp.autocast() if scaler else torch.no_grad():  # Mixed precision only if CUDA is available
#             outputs = model(inputs)
#             loss = criterion(outputs.view(-1, vocab_size), labels.view(-1))

#         if scaler:
#             scaler.scale(loss).backward()
#             scaler.step(optimizer)
#             scaler.update()
#         else:
#             loss.backward()
#             optimizer.step()

#         total_loss += loss.item()

#     avg_loss = total_loss / len(dataloader)
#     print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

# # ✅ Save trained model
# torch.save(model.state_dict(), "my_ai_model/models/transformer.pth")
# print("Training complete! 🎉")



# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset
# import sentencepiece as spm
# from transformer import TransformerLanguageModel
# import logging
# from pathlib import Path
# from tqdm import tqdm
# import numpy as np
# import multiprocessing

# # Setup logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)

# def load_text_from_file(file_path, seq_length=64, sp_model_path="my_ai_model/models/tokenizer.model"):
#     """Loads and tokenizes text data"""
#     sp = spm.SentencePieceProcessor()
#     sp.load(sp_model_path)
#     vocab_size = sp.GetPieceSize()
#     pad_id = sp.pad_id()
    
#     if pad_id < 0:
#         pad_id = vocab_size

#     with open(file_path, "r", encoding="utf-8") as f:
#         text_data = [line.strip() for line in f if line.strip()]
    
#     tokenized_data = []
#     for line in tqdm(text_data, desc="Tokenizing"):
#         tokens = sp.EncodeAsIds(line)
#         tokens = [t if t < vocab_size else 0 for t in tokens]
#         tokens = tokens[:seq_length] + [pad_id] * max(0, seq_length - len(tokens))
#         tokenized_data.append(tokens)
    
#     return torch.tensor(tokenized_data, dtype=torch.long), vocab_size, pad_id

# def train_model():
#     # Model hyperparameters
#     embed_dim = 256  # Reduced from 512 for better initial training
#     hidden_dim = 1024  # Reduced from 2048
#     num_heads = 8
#     num_layers = 6
#     dropout = 0.1
    
#     # Training hyperparameters
#     batch_size = 32  # Reduced from 64 for better stability
#     grad_accumulation_steps = 4
#     num_epochs = 5
#     learning_rate = 5e-4  # Simplified learning rate
    
#     # Load and process data
#     data_file = "my_ai_model/data/clean.txt"
#     data, vocab_size, pad_id = load_text_from_file(data_file)

#     # Create targets
#     targets = torch.cat([data[:, 1:], torch.full((data.shape[0], 1), pad_id, dtype=torch.long)], dim=1)
    
#     # Clamp values
#     data = torch.clamp(data, min=0, max=vocab_size - 1)
#     targets = torch.clamp(targets, min=0, max=vocab_size - 1)

#     # Create DataLoader
#     dataset = TensorDataset(data, targets)
#     dataloader = DataLoader(
#         dataset, 
#         batch_size=batch_size, 
#         shuffle=True,
#         num_workers=0
#     )

#     # Model setup
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     logger.info(f"Using device: {device}")

#     model = TransformerLanguageModel(
#         vocab_size=vocab_size,
#         embed_dim=embed_dim,
#         num_heads=num_heads,
#         hidden_dim=hidden_dim,
#         num_layers=num_layers,
#         dropout=dropout
#     ).to(device)

#     # Loss function with label smoothing
#     criterion = nn.CrossEntropyLoss(
#         ignore_index=pad_id,
#         label_smoothing=0.1
#     )

#     # Optimizer
#     optimizer = optim.AdamW(
#         model.parameters(),
#         lr=learning_rate,
#         betas=(0.9, 0.98),
#         eps=1e-9,
#         weight_decay=0.01
#     )

#     # Training loop
#     best_loss = float('inf')
#     patience = 3
#     patience_counter = 0
    
#     model.train()
#     for epoch in range(num_epochs):
#         total_loss = 0
#         progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

#         for batch_idx, (inputs, labels) in enumerate(progress_bar):
#             inputs, labels = inputs.to(device), labels.to(device)
            
#             if batch_idx % grad_accumulation_steps == 0:
#                 optimizer.zero_grad()

#             try:
#                 outputs = model(inputs)
#                 outputs = outputs.view(-1, vocab_size)
#                 labels = labels.view(-1)
                
#                 loss = criterion(outputs, labels)
#                 loss = loss / grad_accumulation_steps
#                 loss.backward()

#                 if (batch_idx + 1) % grad_accumulation_steps == 0:
#                     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#                     optimizer.step()

#                 progress_bar.set_postfix({'loss': f'{loss.item() * grad_accumulation_steps:.4f}'})
#                 total_loss += loss.item() * grad_accumulation_steps

#             except RuntimeError as e:
#                 logger.error(f"Error in batch {batch_idx}: {str(e)}")
#                 raise

#         avg_loss = total_loss / len(dataloader)
#         logger.info(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

#         # Early stopping
#         if avg_loss < best_loss:
#             best_loss = avg_loss
#             patience_counter = 0
#             torch.save(model.state_dict(), "my_ai_model/models/transformer_best.pth")
#             logger.info(f"New best model saved with loss: {best_loss:.4f}")
#         else:
#             patience_counter += 1
#             if patience_counter >= patience:
#                 logger.info(f"Early stopping triggered after {epoch+1} epochs")
#                 break

#     logger.info("Training complete! 🎉 Model saved successfully.")

# if __name__ == "__main__":
#     multiprocessing.freeze_support()
#     train_model()















import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import sentencepiece as spm
from src.transformer import TransformerLanguageModel
import logging
from pathlib import Path
from tqdm import tqdm
import numpy as np
import multiprocessing
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Paths
MODEL_PATH = "my_ai_model/models/transformer_incremental.pth"
DATA_TRACKER_FILE = "my_ai_model/training_progress.txt"

def load_text_from_file(file_path, seq_length=64, sp_model_path="my_ai_model/models/tokenizer.model"):
    """Loads and tokenizes text data"""
    sp = spm.SentencePieceProcessor()
    sp.load(sp_model_path)
    vocab_size = sp.GetPieceSize()
    pad_id = sp.pad_id()

    if pad_id < 0:
        pad_id = vocab_size

    with open(file_path, "r", encoding="utf-8") as f:
        text_data = [line.strip() for line in f if line.strip()]

    tokenized_data = []
    for line in tqdm(text_data, desc="Tokenizing"):
        tokens = sp.EncodeAsIds(line)
        tokens = [t if t < vocab_size else 0 for t in tokens]
        tokens = tokens[:seq_length] + [pad_id] * max(0, seq_length - len(tokens))
        tokenized_data.append(tokens)

    return torch.tensor(tokenized_data, dtype=torch.long), vocab_size, pad_id

def get_untrained_data(data_folder="my_ai_model/data/"):
    """Checks which files have not been trained yet and returns them"""
    trained_files = set()

    # Read the tracker file if exists
    if os.path.exists(DATA_TRACKER_FILE):
        with open(DATA_TRACKER_FILE, "r") as f:
            trained_files = set(f.read().splitlines())

    # Get all text files in the data folder
    all_files = sorted(Path(data_folder).glob("*.txt"))
    new_files = [str(f) for f in all_files if str(f) not in trained_files]

    return new_files, trained_files

def update_trained_files(new_files):
    """Logs newly trained files"""
    with open(DATA_TRACKER_FILE, "a") as f:
        for file in new_files:
            f.write(file + "\n")

def train_model():
    # Model hyperparameters
    embed_dim = 256
    hidden_dim = 1024
    num_heads = 8
    num_layers = 6
    dropout = 0.1

    # Training hyperparameters
    batch_size = 32
    grad_accumulation_steps = 4
    num_epochs = 5
    learning_rate = 5e-4

    # Check for new data
    new_files, trained_files = get_untrained_data()
    if not new_files:
        logger.info("No new data found for training. Exiting...")
        return

    logger.info(f"Found {len(new_files)} new file(s) for training.")

    # Load and process new data
    all_data = []
    for file in new_files:
        data, vocab_size, pad_id = load_text_from_file(file)
        all_data.append(data)

    # Combine all new data
    if not all_data:
        logger.info("No valid data found. Exiting...")
        return

    data = torch.cat(all_data, dim=0)

    # Create targets
    targets = torch.cat([
        data[:, 1:], 
        torch.full((data.shape[0], 1), pad_id, dtype=torch.long)
    ], dim=1)

    # Clamp values
    data = torch.clamp(data, min=0, max=vocab_size - 1)
    targets = torch.clamp(targets, min=0, max=vocab_size - 1)

    # Create DataLoader
    dataset = TensorDataset(data, targets)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0
    )

    # Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load existing model or create new one
    model = TransformerLanguageModel(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)

    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH))
        logger.info(f"Loaded existing model from {MODEL_PATH}")

    # Loss function
    criterion = nn.CrossEntropyLoss(
        ignore_index=pad_id,
        label_smoothing=0.1
    )

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=0.01
    )

    # Training loop
    best_loss = float('inf')
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch_idx, (inputs, labels) in enumerate(progress_bar):
            inputs, labels = inputs.to(device), labels.to(device)

            if batch_idx % grad_accumulation_steps == 0:
                optimizer.zero_grad()

            try:
                outputs = model(inputs)
                outputs = outputs.view(-1, vocab_size)
                labels = labels.view(-1)

                loss = criterion(outputs, labels)
                loss = loss / grad_accumulation_steps
                loss.backward()

                if (batch_idx + 1) % grad_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                progress_bar.set_postfix({'loss': f'{loss.item() * grad_accumulation_steps:.4f}'})
                total_loss += loss.item() * grad_accumulation_steps

            except RuntimeError as e:
                logger.error(f"Error in batch {batch_idx}: {str(e)}")
                raise

        avg_loss = total_loss / len(dataloader)
        logger.info(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), MODEL_PATH)
            logger.info(f"Model saved with loss: {best_loss:.4f}")

    # Update trained files tracker
    update_trained_files(new_files)
    logger.info("Training complete! Model updated incrementally.")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    train_model()
