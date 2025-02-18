import torch
from transformer import SimpleTransformer
from tokenizer import load_vocab

# Load vocabulary
try:
    vocab = load_vocab("my_ai_model/models/tokenizer.vocab")
    vocab_size = len(vocab)
    print(f"✅ Loaded vocabulary with {vocab_size} tokens.")
except Exception as e:
    print(f"❌ Error loading vocabulary: {e}")
    exit(1)

# Define model
embed_dim = 32
num_heads = 2
hidden_dim = 64
num_layers = 2
model = SimpleTransformer(vocab_size, embed_dim, num_heads, hidden_dim, num_layers)

# Load trained model weights
try:
    model.load_state_dict(torch.load("my_ai_model/models/transformer.pth"))
    model.eval()
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)

# Sample input for evaluation
sample_input = torch.randint(0, vocab_size, (1, 10), dtype=torch.long)  # Randomized input

# Run inference
try:
    with torch.no_grad():
        output = model(sample_input)
    
    print("✅ Model output:")
    print(output)
    print("Evaluation complete! 🎯")
except Exception as e:
    print(f"❌ Error during evaluation: {e}")
