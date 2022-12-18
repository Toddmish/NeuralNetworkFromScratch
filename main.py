from Experiments.experimentParams import*
from nn import *
def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.

    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


if __name__ == '__main__':
    print_hi('Building a Neural Network from scratch...')
    print(__name__)

    init_nn= NN()
    test_nn= NN_Test(init_nn)

    """
        test_nn.initialize_parameters(init_nn)
        test_nn.test2(init_nn)
        test_nn.test3(init_nn)
        """

    test_nn.run_all()
