from pyannet.neural_network import *

def dict_to_vector(dictionary):
    
    """
    Converts all the values in a dictionary to a vector
    dictionary: The original dictionary to be converted

    """
    vector = []
    for key in dictionary:
        vector = np.concatenate((vector,dictionary[f'{key}'].flatten()))
    return vector

def vector_to_dict(vector, original_dictionary):
    """
    Converts a vector to a dictionary
    vector   : Vector to be converted
    original_dictionary : Template for conversion
    """
    new_dictionary = {}
    current_index = 0
    for key,item in original_dictionary.items():
        new_index = current_index + item.size
        new_dictionary[f'{key}'] =  np.reshape(vector[current_index:new_index],item.shape)
        current_index = new_index
    return new_dictionary

def gradient_checking(network, X, y, epsilon=1e-7):

    """
       Checks whether the implementation of backpropagation is correct by comparing the gradiens calculated 
       by backpropagation with that calculated numericaly (here: Two sided epsilon gradient).
       network : A neural network object to test
       y       : true output(can be an nd array with m datasets)
       X       : input (can be an nd array with m datasets)
       epsilon : a small value for calculating gradient by two sided epsilon method

    """
    network.construct_parameters()
    network.forward_propagate()
    network.back_propagate(y)
    parameters_original = network.weights_and_biases
    original_gradients = network.parameter_gradients
    original_gradients_vector = dict_to_vector(original_gradients)
    parameters_vector = dict_to_vector(parameters_original)
    grad_two_sided_vector = np.zeros(len(parameters_vector))
    for j in range(len(parameters_vector)):
        parameters_vector = dict_to_vector(parameters_original)
        parameters_vector[j] = parameters_vector[j] + epsilon
        network.weights_and_biases = vector_to_dict(parameters_vector,parameters_original)
        y_pred = network.forward_propagate()
        C_plus = network.cost_functions(y_pred,y) 
        parameters_vector = dict_to_vector(parameters_original)
        parameters_vector[j] = parameters_vector[j] - epsilon
        network.weights_and_biases = vector_to_dict(parameters_vector,parameters_original)
        y_pred = network.forward_propagate()
        C_minus = network.cost_functions(y_pred,y)
        grad_two_sided_vector[j] = (C_plus - C_minus)/(2*epsilon)
    error_difference = (np.linalg.norm(original_gradients_vector-grad_two_sided_vector))/(np.linalg.norm(grad_two_sided_vector) + np.linalg.norm(original_gradients_vector))
    return error_difference

def test_gradient_checking_multiple_output_network():
    X = np.asarray([2,3,5,6])
    y = np.asarray([1,6,3])
    network = NeuralNetwork((4,3,3), X , activations=('swish','swish'))
    assert(gradient_checking(network,X,y)<1e-7) == True

def test_gradient_checking_single_output_network():
    X = np.asarray([2,3,5,6])
    y = np.asarray([3])
    network = NeuralNetwork((4,3,1), X , activations=('swish','linear'))
    assert(gradient_checking(network,X,y)<1e-7) == True

def test_gradient_checking_single_output_multiple_hidden_layered_network():
    X = np.asarray([2,3,5,6])
    y = np.asarray([3])
    network = NeuralNetwork((4,5,3,1), X , activations=('swish','swish','linear'))
    assert(gradient_checking(network,X,y)<1e-7) == True

def test_gradient_checking_multiple_output_multiple_hidden_layered_network():
    X = np.asarray([0.7866,7.99,1.66,6])
    y = np.asarray([3,5,7,9,3.8])
    network = NeuralNetwork((4,5,3,5), X , activations=('swish','swish','linear'))
    assert(gradient_checking(network,X,y)<1e-7) == True