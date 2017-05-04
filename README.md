This project is an attempt to create a simple machine learning library
containing implemenations of a commonly used algorithms implemented in
a way that is easy to understand.

### Requirements
The requirements can be installed by running:
```
pip install -r requirements.txt
```

### Useful guides:
[Peter Roelants' guide to Implementing neural networks](https://peterroelants.github.io/posts/neural_network_implementation_part01/)


## Examples

### Running the MNIST example
Begin by downloading the MNIST dataset from the [MNIST database
homepage](http://yann.lecun.com/exdb/mnist/). The location of the dataset is
hardcoded to `data` in the file `run_mnist.py`, but feel free to change it.

After having installed the requirements, you should be able to run
`python -m examples.run_mnist`, which trains the network
and prints the final accuracy.
