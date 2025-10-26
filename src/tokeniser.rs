use tiktoken_rs::{CoreBPE, r50k_base};

/// Byte Pair Encoding (BPE) Tokeniser
/// Use the r50k_base tokeniser from tiktoken_rs to tokenize text.
/// This is the same tokeniser used by OpenAI's GPT 2 model.
pub struct Tokeniser {
    bpe: CoreBPE,
}

impl Tokeniser {
    pub fn new() -> anyhow::Result<Self> {
        let bpe = r50k_base()?;
        Ok(Tokeniser { bpe })
    }

    pub fn encode(&self, input: &str) -> Vec<u32> {
        self.bpe.encode_with_special_tokens(input)
    }

    pub fn decode(&self, tokens: Vec<u32>) -> anyhow::Result<String> {
        self.bpe.decode(tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// See page 33 of Building Large Language Models from Scratch
    #[test]
    fn test_tokenisation() -> anyhow::Result<()> {
        let tokeniser = Tokeniser::new()?;
        let text = "Hello, do you like tea? <|endoftext|> In the sunlit terraces \
        of someunknownPlace.";
        let tokens = tokeniser.encode(text);
        let expected_tokens = [
            15496, 11, 466, 345, 588, 8887, 30, 220, 50256, 554, 262, 4252,
            18250, 8812, 2114, 286, 617, 34680, 27271, 13,
        ];
        assert_eq!(tokens, expected_tokens);

        // Decode back to verify
        let original_text = tokeniser.decode(tokens)?;
        assert_eq!(original_text, text);
        Ok(())
    }

    /// See page 34 of Building Large Language Models from Scratch
    #[test]
    fn test_unknown_tokenisation_test() -> anyhow::Result<()> {
        let tokeniser = Tokeniser::new()?;
        let text = "Akwirw ier";
        let tokens = tokeniser.encode(text);
        let expected_tokens = [33901, 86, 343, 86, 220, 959];
        assert_eq!(tokens, expected_tokens);

        // Decode back to verify
        let original_text = tokeniser.decode(tokens)?;
        assert_eq!(original_text, text);

        Ok(())
    }
}
