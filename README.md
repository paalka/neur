This project is an attempt to create a simple neural net which can be used on the
MNIST dataset.
The implementation is based on a wonderful guide which can be found
[here](https://peterroelants.github.io/posts/neural_network_implementation_part01/).

### Running the MNIST example
Begin by downloading the MNIST dataset from the [MNIST database
homepage](http://yann.lecun.com/exdb/mnist/). The location of the dataset is
hardcoded to `data` in the file `run_mnist.py`, but feel free to change it.

Then, install the required libraries specified in `requirements.txt`.
You should now be able to run `python -m examples.run_mnist`, which trains the network
and prints the final accuracy.
