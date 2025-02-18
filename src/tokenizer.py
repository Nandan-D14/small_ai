import sentencepiece as spm

def train_tokenizer(input_file, model_prefix, vocab_size=8000, character_coverage=0.98, model_type="unigram"):
    """
    Trains a SentencePiece tokenizer.
    
    :param input_file: Path to the training text file.
    :param model_prefix: Prefix for the saved model files.
    :param vocab_size: Number of vocabulary tokens.
    :param character_coverage: Percentage of characters covered.
    :param model_type: Type of model ('unigram', 'bpe', 'char', 'word').
    """
    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=character_coverage,
        model_type=model_type
    )
    print(f"Tokenizer trained and saved as {model_prefix}.model and {model_prefix}.vocab")

def load_tokenizer(model_prefix):
    """
    Loads a trained SentencePiece model.
    
    :param model_prefix: Prefix of the model file.
    :return: SentencePieceProcessor object.
    """
    sp = spm.SentencePieceProcessor()
    sp.load(f"{model_prefix}.model")
    return sp

if __name__ == "__main__":
    input_file = "my_ai_model/data/clean.txt"
    model_prefix = "my_ai_model/models/tokenizer"
    
    # Train the tokenizer
    train_tokenizer("my_ai_model/data/clean.txt", "my_ai_model/models/tokenizer", vocab_size=3000)


    # Load tokenizer for testing
    sp = load_tokenizer(model_prefix)
    sample_text = "Learning AI is fun!"
    tokenized_text = sp.encode(sample_text, out_type=str)
    print("Tokenized text:", tokenized_text)
