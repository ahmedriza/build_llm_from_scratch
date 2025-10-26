use tch::Tensor;
use tch::kind;

fn main() -> anyhow::Result<()> {
    let t = Tensor::from_slice(&[1, 2, 3]);
    let t = t * 2;
    // t.print(); // works on cpu tensors

    println!("t(Cpu): {:?}", &t);
    println!("t device: {:?}", &t.device());
    println!("");

    let t = Tensor::randn([5, 4], kind::FLOAT_CPU).to_device(tch::Device::Mps);
    t.print();
    println!("t(mps) {:?}", &t);
    println!("t device: {:?}", &t.device());

    ndarray_example();
    grad_example();

    Ok(())
}

fn ndarray_example() {
    let nd = ndarray::arr2(&[[1f64, 2.], [3., 4.]]);
    let tensor = tch::Tensor::try_from(nd.clone()).unwrap();
    println!("ndarray: \n{:?}", nd);
    println!("tensor:");
    tensor.print();
}

fn grad_example() {
    let mut x = tch::Tensor::from(2.0).set_requires_grad(true);
    let y = &x * &x + &x + 36;
    println!("y: {}", y.double_value(&[]));
    x.zero_grad();
    y.backward();
    let dy_over_dx = x.grad();
    println!("dy/dx = {}", dy_over_dx.double_value(&[]));
}
