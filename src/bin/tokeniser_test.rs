use build_llm_from_scratch::tokeniser::Tokeniser;

fn main() -> anyhow::Result<()> {
    log4rs::init_file("log4rs.yml", Default::default()).unwrap();

    // let bpe = r50k_base()?;
    let tokeniser = Tokeniser::new()?;

    let text = std::fs::read_to_string("the-verdict.txt")?;

    let enc_text = tokeniser.encode(&text);
    log::info!("Number of tokens: {:?}", enc_text.len());

    // remove the first 50 tokens from enc_text
    let enc_sample = &enc_text[50..];

    let context_size = 4;
    // get context_size tokens from the enc_sample
    let x = &enc_sample[0..context_size];
    let y = &enc_sample[1..context_size + 1];
    log::info!("x: {:?}", x);
    log::info!("y:      {:?}", y);

    // Everything to the left of ---> refers to the input an LLM would receive
    // and the token ID on the right side of the arrow represents the target
    // token ID that the LLM is expected to predict.
    //
    // [290] ---> 4920
    // [290, 4920] ---> 2241
    // [290, 4920, 2241] ---> 287
    // [290, 4920, 2241, 287] ---> 257
    for i in 1..context_size + 1 {
        let context = &enc_sample[0..i];
        let desired = enc_sample[i];
        log::info!(
            "{} ---> {}",
            tokeniser.decode(context.to_vec())?,
            tokeniser.decode(vec![desired])?
        );
    }

    Ok(())
}
