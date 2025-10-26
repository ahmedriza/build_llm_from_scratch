import re

class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {idx: token for token, idx in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids = [self.str_to_int[token] for token in preprocessed]
        return ids

    def decode(self, token_ids):
        text = " ".join([self.int_to_str[idx] for idx in token_ids])
        # remove spaces before punctuation except quote (')
        text = re.sub(r'\s+([,.?!"()])', r'\1', text)
        # remove spaces before and after quotes (')
        # This is to clean text such as: "It's"
        # which would be left as: "It' s" after the above substitution
        text = re.sub(r'\s+([\'])(\s+)?', r'\1', text)
        return text

class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {idx: token for token, idx in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [
                item if item in self.str_to_int else "<|unk|>"
                for item in preprocessed
                ]
        ids = [self.str_to_int[token] for token in preprocessed]
        return ids

    def decode(self, token_ids):
        text = " ".join([self.int_to_str[idx] for idx in token_ids])
        # remove spaces before punctuation except quote (')
        text = re.sub(r'\s+([,.?!"()])', r'\1', text)
        # remove spaces before and after quotes (')
        # This is to clean text such as: "It's"
        # which would be left as: "It' s" after the above substitution
        text = re.sub(r'\s+([\'])(\s+)?', r'\1', text)
        return text
