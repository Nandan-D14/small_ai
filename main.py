# import torch
# import torch.nn.functional as F
# import sentencepiece as spm
# from src.transformer import TransformerLanguageModel

# # # Load tokenizer
# sp = spm.SentencePieceProcessor()
# tokenizer_path = "my_ai_model/models/tokenizer.model"

# try:
#     sp.load(tokenizer_path)
#     print(f"✅ Tokenizer loaded from {tokenizer_path}")
# except Exception as e:
#     print(f"❌ Error loading tokenizer: {e}")
#     exit(1)

# # Model parameters
# vocab_size = sp.GetPieceSize()
# embed_dim = 128  
# num_heads = 4  
# hidden_dim = 256  
# num_layers = 4  

# model = SimpleTransformer(vocab_size, embed_dim, num_heads, hidden_dim, num_layers)


# # Load model
# # model = SimpleTransformer(vocab_size, embed_dim, num_heads, hidden_dim, num_layers)
# model_path = "my_ai_model/models/transformer.pth"

# try:
#     model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
#     model.eval()
#     print(f"✅ Model loaded from {model_path}")
# except Exception as e:
#     print(f"❌ Error loading model: {e}")
#     exit(1)

# def generate_text(prompt, max_length=50, temperature=1.0, top_p=0.9):
#     """Generates text using nucleus (top-p) sampling."""
#     input_ids = torch.tensor([sp.EncodeAsIds(prompt)], dtype=torch.long)

#     with torch.no_grad():
#         for _ in range(max_length):
#             output = model(input_ids)  # Forward pass
#             logits = output[:, -1, :] / temperature  # Apply temperature scaling
#             probabilities = F.softmax(logits, dim=-1)

#             # Apply nucleus (top-p) sampling
#             sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)
#             cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            
#             # Remove tokens with cumulative probability above top_p
#             top_p_mask = cumulative_probs > top_p
#             sorted_probs[top_p_mask] = 0
#             sorted_probs /= sorted_probs.sum()  # Normalize probabilities

#             # Sample next token
#             next_token = sorted_indices[0, torch.multinomial(sorted_probs[0], 1)].item()

#             if next_token == sp.eos_id():
#                 break
            
#             input_ids = torch.cat([input_ids, torch.tensor([[next_token]])], dim=1)

#     return sp.DecodeIds(input_ids.squeeze(0).tolist())


# # Example usage
# if __name__ == "__main__":
#     prompt =  "hello bro " #input("Enter a prompt: ")
#     generated_text = generate_text(prompt)
#     print("\n📝 Generated text:", generated_text)











import torch
import torch.nn.functional as F
import sentencepiece as spm
from src.transformer import TransformerLanguageModel
import logging
from typing import List, Optional
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TextGenerator:
    def __init__(
        self,
        model_path: str = "my_ai_model/models/transformer_best.pth", #transformer_best.pth
        tokenizer_path: str = "my_ai_model/models/tokenizer.model",
        device: Optional[str] = None
    ):
        # Setup device
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Load tokenizer
        self.sp = spm.SentencePieceProcessor()
        try:
            self.sp.load(tokenizer_path)
            logger.info(f"✅ Tokenizer loaded from {tokenizer_path}")
        except Exception as e:
            logger.error(f"❌ Error loading tokenizer: {e}")
            raise

        # Model parameters
        self.vocab_size = self.sp.GetPieceSize()
        self.embed_dim = 256  # Match training parameters
        self.num_heads = 8
        self.hidden_dim = 1024
        self.num_layers = 6
        self.dropout = 0.1

        # Initialize model
        self.model = TransformerLanguageModel(
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)

        # Load model weights
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            logger.info(f"✅ Model loaded from {model_path}")
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            raise

    def _apply_repetition_penalty(
        self,
        logits: torch.Tensor,
        generated_tokens: List[int],
        penalty: float
    ) -> torch.Tensor:
        """Apply repetition penalty to logits based on generated tokens."""
        if len(generated_tokens) == 0:
            return logits

        for token in set(generated_tokens):
            logits[0, token] /= penalty
        return logits

    def _nucleus_sampling(
        self,
        logits: torch.Tensor,
        top_p: float,
        top_k: int,
        temperature: float
    ) -> int:
        """Apply nucleus sampling (top-p) with temperature and top-k filtering."""
        # Temperature scaling
        logits = logits / max(temperature, 1e-8)

        # Top-k filtering
        top_k = min(top_k, logits.size(-1))
        top_k_logits, top_k_indices = torch.topk(logits, top_k)
        
        # Softmax for probabilities
        probs = F.softmax(top_k_logits, dim=-1)

        # Top-p (nucleus) sampling
        cumulative_probs = torch.cumsum(probs, dim=-1)
        remove_indices = cumulative_probs > top_p
        probs[remove_indices] = 0.0

        # Renormalize probabilities
        if torch.sum(probs) > 0:
            probs = probs / probs.sum()
        else:
            probs = torch.ones_like(probs) / probs.size(-1)

        # Sample token
        next_token_idx = torch.multinomial(probs[0], 1)
        return top_k_indices[0, next_token_idx].item()

    def generate(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.2,
        min_length: int = 10,
        stop_tokens: Optional[List[str]] = None
    ) -> str:
        """
        Generate text with advanced controls and better handling.
        """
        try:
            # Encode prompt
            input_ids = torch.tensor([self.sp.EncodeAsIds(prompt)], dtype=torch.long).to(self.device)
            generated = []
            
            # Convert stop tokens to ids if provided
            stop_ids = []
            if stop_tokens:
                stop_ids = [self.sp.EncodeAsIds(token)[0] for token in stop_tokens if token]

            # Generation loop
            with torch.no_grad():
                for i in range(max_length):
                    # Get model output
                    outputs = self.model(input_ids)
                    next_token_logits = outputs[:, -1, :]

                    # Apply repetition penalty
                    next_token_logits = self._apply_repetition_penalty(
                        next_token_logits,
                        generated[-min(len(generated), 10):],  # Only consider last 10 tokens
                        repetition_penalty
                    )

                    # Get next token
                    next_token = self._nucleus_sampling(
                        next_token_logits,
                        top_p,
                        top_k,
                        temperature
                    )
                    
                    # Check stop conditions
                    if i >= min_length:
                        if next_token == self.sp.eos_id():
                            break
                        if next_token in stop_ids:
                            break

                    # Add to generated sequence
                    generated.append(next_token)
                    input_ids = torch.cat([
                        input_ids,
                        torch.tensor([[next_token]], device=self.device)
                    ], dim=1)

            # Decode and clean up generated text
            generated_text = self.sp.DecodeIds(generated)
            
            # Basic cleanup
            generated_text = generated_text.strip()
            
            # Ensure proper sentence ending
            if not generated_text[-1] in '.!?':
                generated_text += '.'

            return generated_text

        except Exception as e:
            logger.error(f"Error during text generation: {str(e)}")
            return f"{prompt} [Error: {str(e)}]"

def main():
    try:
        # Initialize generator
        generator = TextGenerator()

        # Get user input with instructions
        print("\n=== Text Generation System ===")
        print("Tips for better results:")
        print("- Provide clear, specific prompts")
        print("- Use proper punctuation")
        print("- Higher temperature (0.8-1.0) for more creative text")
        print("- Lower temperature (0.3-0.7) for more focused text")
        
        prompt = input("\nEnter your prompt: ")
        
        # Get generation parameters
        # temp = float(input("Temperature (0.1-1.0, default 0.7): ") or 0.7)
        # length = int(input("Maximum length (10-200, default 100): ") or 100)

        # Generate text
        generated_text = generator.generate(
            prompt=prompt,
            max_length=70,
            temperature=0.7,
            top_p=0.92,
            top_k=50,
            repetition_penalty=1.2,
            min_length=10
        )

        print("\n=== Generated Text ===")
        print(generated_text)
        print("=====================")

    except Exception as e:
        logger.error(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main()












