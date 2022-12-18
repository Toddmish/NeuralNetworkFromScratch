from experimentParams import*
from nn import *

def print_hi(text):
    print(f'Hi, {text}')  


if __name__ == '__main__':
    print_hi('Building a Neural Network from scratch...')
    print(__name__)

    init_nn= NN()
    experiments= NN_Experiments(init_nn)

    """
        test_nn.initialize_parameters(init_nn)
        test_nn.test2(init_nn)
        test_nn.test3(init_nn)
        """

    experiments.run_all()
