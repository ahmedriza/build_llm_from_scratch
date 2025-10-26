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

    Ok(())
}
