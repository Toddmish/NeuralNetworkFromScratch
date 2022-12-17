from nn import *

class NN_Test():

    def __init__(self,nn):
        self.nn=nn

    def initialize_parameters(self):
        parameters = self.nn.initialize_parameters(3, 2, 1)
        print('\n\nTest Task 1 Initialize Parameters:\n')
        print("W1 = " + str(parameters["W1"]))
        print("b1 = " + str(parameters["b1"]))
        print("W2 = " + str(parameters["W2"]))
        print("b2 = " + str(parameters["b2"]))

    def test2(self):
        parameters = self.nn.initialize_parameters_deep([5,4,3])
        print('\n\nTest Task 2:\n')
        print("W1 = " + str(parameters["W1"]))
        print("b1 = " + str(parameters["b1"]))
        print("W2 = " + str(parameters["W2"]))
        print("b2 = " + str(parameters["b2"]))

    def test3(self):
        print('\n\nTest Task 3:\n')
        A, W, b = linear_forward_test_case()
        Z, linear_cache = self.nn.linear_forward(A, W, b)
        print("Z = " + str(Z))

    def test4(self):
        print('\n\nTest Task 4:\n')
        A_prev, W, b = linear_activation_forward_test_case()

        A, linear_activation_cache = self.nn.linear_activation_forward(A_prev, W, b, activation="sigmoid")
        print("With sigmoid: A = " + str(A))

        A, linear_activation_cache = self.nn.linear_activation_forward(A_prev, W, b, activation="relu")
        print("With ReLU: A = " + str(A))

    def test5(self):
        print('\n\nTest Task 5:\n')
        X, parameters = L_model_forward_test_case_2hidden()
        AL, caches = self.nn.L_model_forward(X, parameters)
        print("AL = " + str(AL))
        print("Length of caches list = " + str(len(caches)))

    def test6(self):
        print('\n\nTest Task 6:\n')
        Y, AL = compute_cost_test_case()

        print("cost = " + str(self.nn.compute_cost(AL, Y)))

    def test7(self):
        print('\n\nTest Task 7:\n')
        dZ, linear_cache = linear_backward_test_case()
        dA_prev, dW, db = self.nn.linear_backward(dZ, linear_cache)
        print("dA_prev = " + str(dA_prev))
        print("dW = " + str(dW))
        print("db = " + str(db))

    def linear_activation_backward(self):
        print('\n\nTest No 8: Linear Activation Backward:\n')
        dAL, linear_activation_cache = linear_activation_backward_test_case()

        dA_prev, dW, db = self.nn.linear_activation_backward(dAL, linear_activation_cache, activation="sigmoid")
        print("sigmoid:")
        print("dA_prev = " + str(dA_prev))
        print("dW = " + str(dW))
        print("db = " + str(db) + "\n")

        dA_prev, dW, db = self.nn.linear_activation_backward(dAL, linear_activation_cache, activation="relu")
        print("relu:")
        print("dA_prev = " + str(dA_prev))
        print("dW = " + str(dW))
        print("db = " + str(db))

    def L_model_backward(self):
        print('\n\nTest Step 9: L_model_backward:\n')
        AL, Y_assess, caches = L_model_backward_test_case()
        grads = self.nn.L_model_backward(AL, Y_assess, caches)
        print_grads(grads)

    def update_parameters(self):
        print('\n\nTest Step 10: update parameters:\n')
        parameters, grads = update_parameters_test_case()
        parameters = self.nn.update_parameters(parameters, grads, 0.1)
        print("W1 = " + str(parameters["W1"]))
        print("b1 = " + str(parameters["b1"]))
        print("W2 = " + str(parameters["W2"]))
        print("b2 = " + str(parameters["b2"]))


    def run_all_tests(self):
        self.initialize_parameters()
        self.test2()
        self.test3()
        self.test4()
        self.test5()
        self.test6()
        self.test7()
        self.linear_activation_backward()
        self.L_model_backward()
        self.update_parameters()






