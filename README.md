This is code repo to follow along the book:

Build a Large Language Model from Scratch. 

The code is in two languages, Rust and Python.  The book uses Python
code and I've written the Rust code myself based on the Python code.

Create a virtual environment and activate it and then install the
required packages.

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Note that the virtual environment must be created even when working with the
Rust code, since the Rust code depends on the verison of PyTorch installed
in the virtuan env.

# Rust

We use the [tch-rs](https://github.com/LaurentMazare/tch-rs) crate
for `torch` rust bindings.

There is a `.cargo/config.toml` file with the following contents to set `LIBTORCH` and `DYLD_LIBRARY_PATH`.
The `LIBTORCH` environment variable must point to the directory which must include `include` and `lib`
sub directories.  The PyTorch shared libs must be present in `$LIBTORCH/lib` directory.

```
[env]
LIBTORCH = { value = "venv/lib/python3.13/site-packages/torch" , relative = true }
DYLD_LIBRARY_PATH = { value = "venv/lib/python3.13/site-packages/torch/lib" , relative = true }
```

```
cargo build
```

# References

* tch-rs crate
  https://github.com/LaurentMazare/tch-rs
* Notes on running PyTorch on M1/M2 etc mac 
  https://github.com/ssoudan/tch-m1/
* Creating a Neural Network from Scratch in Rust — Part 1
  https://medium.com/@jesuskevin254/creating-a-neural-network-from-scratch-in-rust-part-1-f9f8d30ed75b
