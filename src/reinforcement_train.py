import torch
import json
from reinforcement import ReinforcementAgent
from transformer import SimpleTransformer
from tokenizer import load_vocab

# Load vocabulary size
vocab = load_vocab("my_ai_model/models/tokenizer.vocab")
vocab_size = len(vocab)

# Define model
embed_dim = 32
num_heads = 2
hidden_dim = 64
model = SimpleTransformer(vocab_size, embed_dim, num_heads, hidden_dim)

# Define RL agent
agent = ReinforcementAgent(model, vocab_size)

# Load training data
with open("my_ai_model/data/rl_data.json", "r") as f:
    experiences = json.load(f)

# Convert experiences to tensors
processed_experiences = []
for exp in experiences:
    state = torch.tensor(exp["state"], dtype=torch.long)
    action = torch.tensor(exp["action"], dtype=torch.long)
    reward = torch.tensor(exp["reward"], dtype=torch.float32)
    next_state = torch.tensor(exp["next_state"], dtype=torch.long)
    processed_experiences.append((state, action, reward, next_state))

# Train the model
print("Training with Reinforcement Learning...🚀")
agent.train(processed_experiences)

# Save trained model
torch.save(model.state_dict(), "my_ai_model/models/transformer.pth")
print("Reinforcement learning training complete! 🎉")
