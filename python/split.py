import re

# text = "Hello, world. Is this-- a test?"
# result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
# result = [item.strip() for item in result if item.strip()]
# print(result)

filename = "the-verdict.txt"
with open(filename, "r", encoding="utf-8") as f:
    raw_text = f.read()
# print(raw_text[:99])

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]

print("First 30 tokens:")
print(preprocessed[:30])

print(f"Total number of characters: {len(raw_text): 10d}")
print(f"Total number of tokens:     {len(preprocessed): 10d}")

all_tokens = sorted(set(preprocessed))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab_size = len(all_tokens)
print(f"Vocabulary size:            {vocab_size: 10d}")

vocab = {token: idx for idx, token in enumerate(all_tokens)}

"""
for i, item in enumerate(vocab.items()):
    print(item)
    if i >= 50:
        break
"""

# Test the SimpleTokenizerV1
from tokeniser import SimpleTokenizerV1 
tokenizer = SimpleTokenizerV1(vocab)
text = """"It's the last he painted, you know,"
       Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
decoded_text = tokenizer.decode(ids)
print()
print("Input text:", text)
print("encoded:", ids)
print("decoded:", decoded_text)

# Test the SimpleTokenizerV2
from tokeniser import SimpleTokenizerV2 
text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))
print()
print("Input text:", text)
tokenizer = SimpleTokenizerV2(vocab)
print("encoded:", tokenizer.encode(text))
print("decoded:", tokenizer.decode(tokenizer.encode(text)))
