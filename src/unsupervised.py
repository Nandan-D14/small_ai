import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from transformer import SimpleTransformer

# Hyperparameters
vocab_size = 1000  
embed_dim = 32
num_heads = 2
hidden_dim = 64
num_layers = 2
num_epochs = 10
batch_size = 32
learning_rate = 0.001

# Dummy dataset (replace with real data)
data = torch.randint(0, vocab_size, (1000, 10))  # 1000 sequences of length 10

# Masked language modeling: Mask some tokens for the model to predict
mask_token = vocab_size - 1  # Assuming the last vocab index is the mask token
masked_data = data.clone()
mask_indices = torch.rand(masked_data.shape) < 0.15  # 15% masking
masked_data[mask_indices] = mask_token

dataset = TensorDataset(masked_data, data)  # Inputs are masked, targets are original
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model, loss function, optimizer
model = SimpleTransformer(vocab_size, embed_dim, num_heads, hidden_dim, num_layers)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    total_loss = 0
    for batch in dataloader:
        inputs, labels = batch

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs.view(-1, vocab_size), labels.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

print("Unsupervised training complete! 🎉")
