use tch::{
    Device,
    nn::{self, Module, OptimizerConfig, VarStore},
};

fn build_nn(vs: &nn::Path, num_inputs: i64, num_outputs: i64) -> impl Module {
    nn::seq()
        // 1st hidden layer
        .add(nn::linear(vs, num_inputs, 30, Default::default()))
        .add_fn(|xs| xs.relu())
        // 2nd hidden layer
        .add(nn::linear(vs, 30, 20, Default::default()))
        .add_fn(|xs| xs.relu())
        // output layer
        .add(nn::linear(vs, 20, num_outputs, Default::default()))
}

fn print_variables(vs: &VarStore) {
    let mut sum = 0;
    let variables = vs.variables();
    variables.iter().for_each(|(_name, tensor)| {
        sum += tensor.numel();
    });
    println!("Total number of trainable model parameters: {}", sum);
}

// Example from Build a Large Language Model from Scratch, page 266
#[allow(unused)]
fn train_example_one() {
    tch::manual_seed(123);
    let vs = nn::VarStore::new(Device::Cpu);
    println!("VarStore device: {:?}", vs.device());

    let binding = vs.root();
    let model = build_nn(&binding, 50, 3);
    // println!("Network: {:#?}", model);

    print_variables(&vs);

    let x = tch::Tensor::rand(&[1, 50], (tch::Kind::Float, Device::Cpu));
    let y = model.forward(&x);
    println!("y:");
    y.print();

    let out = y.softmax(1, tch::Kind::Float);
    // let out = tch::Tensor::softmax(&y, 1, tch::Kind::Float);
    println!("output:");
    out.print();
}

// Example from Build a Large Language Model from Scratch, page 274
fn train_example_two() -> anyhow::Result<()> {
    tch::manual_seed(123);

    let v_train: Vec<f32> =
        vec![-1.2, 3.1, -0.9, 2.9, -0.5, 2.6, 2.3, -1.1, 2.7, -1.5];
    let x_train = tch::Tensor::from_slice(&v_train).view([5, 2]);
    let y_train =
        tch::Tensor::from_slice(&[0, 0, 0, 1, 1]).to_kind(tch::Kind::Int64);

    let v_test: Vec<f32> = vec![-0.8, 2.6, 2.6, -1.6];
    let _x_test = tch::Tensor::from_slice(&v_test).view([2, 2]);
    let _y_test = tch::Tensor::from_slice(&[0, 1]);

    let vs = nn::VarStore::new(Device::Cpu);
    println!("VarStore device: {:?}", vs.device());


    let binding = vs.root();
    // The dataset has 2 input features and 2 output classes
    let model = build_nn(&binding, 2, 2);
    print_variables(&vs);

    // Stochastic Gradient Descent optimiser with learning rate 0.5
    let mut optimiser = nn::Sgd::default().build(&vs, 0.5)?;

    let num_epochs = 3;
    let batch_size = 2;
    for epoch in 1..=num_epochs {
        let mut batch_id = 0;
        // Create a batch iterator over the training data
        // and call shuffle on the iterator to get batches in random order
        for (features, labels) in
            tch::data::Iter2::new(&x_train, &y_train, batch_size).shuffle()
        {
            // features.print();
            let logits = model.forward(&features);
            // logits.print();
            let loss = logits.cross_entropy_for_logits(&labels);
            // sets the gradients from the previous round to zero to prevent
            // unintented gradient accumulation
            optimiser.zero_grad();
            loss.backward();
            optimiser.step();
            println!(
                "epoch: {:>3}/{num_epochs} | batch: {:>3}/{batch_size} | loss: {:8.5}",
                epoch,
                batch_id,
                &loss.double_value(&[])
            );
            batch_id += 1;
        }
    }

    let outputs = model.forward(&x_train);
    let probabilities = outputs.softmax(-1, tch::Kind::Float);
    println!("probabilities:");
    probabilities.print();
    let predictions = outputs.argmax(-1, false);
    println!("predictions:");
    predictions.print();

    Ok(())
}

fn main() -> anyhow::Result<()> {
    train_example_two()?;
    Ok(())
}
