from pyannet.neural_network import *

def test_nn_correctly_takes_architecture_data():
    assert(NeuralNetwork((4, 2, 1)).architecture.all() == np.asarray((4, 2, 1)).all()) == True

def test_nn_activation_swish():
    assert(NeuralNetwork((4, 2, 1)).activation(x=[(-20, -1.0, 0.0, 1.0, 20)],type="swish")).all() == (np.asarray((-4.12230724e-08,-2.68941421e-01,0.00000000e+00,7.31058579e-01, 2.00000000e+01))).all()

def test_nn_activation_sigmoid():
    assert(NeuralNetwork((4, 2, 1)).activation(x=[(-20, -1.0, 0.0, 1.0, 20)],type="sigmoid")).all() == (np.asarray((2.06115362e-09, 2.68941421e-01, 5.00000000e-01, 7.31058579e-01, 9.99999998e-01))).all()

def test_nn_activation_relu():
    assert(NeuralNetwork((4, 2, 1)).activation(x=[(-10, -5, 0.0, 5, 10)],type="relu")).all() == (np.asarray(( 0.,  0., 0. , 5., 10.))).all()

def test_nn_activation_linear():
    assert(NeuralNetwork((4, 2, 1)).activation(x=[(-10, -5, 0.0, 5, 10)],type="relu")).all() == (np.asarray((-10, -5, 0.0, 5, 10))).all()

def test_nn_activation_leakyrelu():
    assert(NeuralNetwork((4, 2, 1)).activation(x=[(-10, -5, 0.0, 5, 10)],type="relu")).all() == (np.asarray((-0.1, -0.05,  0.,    5.,   10.  ))).all()

def test_nn_forward_propagate_1_hidden_layer_with_1_node_each_with_linear_activations_unit_parameters_return_input():
    X = np.asarray(1)
    d = NeuralNetwork((1,1,1), X , activations=('linear','linear'))
    W1 = 1
    W2 = 1
    b1 = 0
    b2 = 0
    W = np.asarray((W1, W2),dtype=object)
    b = np.asarray((b1, b2),dtype=object)
    d.construct_parameters(method='manual', W=W, b=b)
    assert(d.forward_propagate()) == X

def test_nn_forward_propagate_2_hidden_layer_with_1_node_each_with_linear_activations_unit_parameters_return_input():
    X = np.asarray(1)
    d = NeuralNetwork((1,1,1,1), X , activations=('linear', 'linear', 'linear'))
    W1 = 0
    W2 = 0
    W3 = 0

    b1 = 1
    b2 = 1
    b3 = 1

    W = np.asarray((W1, W2, W3),dtype=object)
    b = np.asarray((b1, b2, b3),dtype=object)
    d.construct_parameters(method='manual', W=W, b=b)
    assert(d.forward_propagate()) == X

def test_nn_forward_propagate_1_hidden_layer_with_1_node_each_with_linear_activations_unit__parameters_return_2_plus_input():
    """
    Forward propagation of neural network with 1 hidden layer with linear activation with unit parameters return 2 plus input
    """
    X = np.asarray(2)
    d = NeuralNetwork((1,1,1), X , activations=('linear','linear'))
    W1 = 1
    W2 = 1
    b1 = 1
    b2 = 1
    W = np.asarray((W1, W2),dtype=object)
    b = np.asarray((b1, b2),dtype=object)
    d.construct_parameters(method='manual', W=W, b=b)
    assert(d.forward_propagate()) == X+2

def test_nn_forward_propagate_1_hidden_layer_compared_with_analytical_results():
    X = np.asarray((1,4,3,0))
    a = NeuralNetwork((4,2,3), X , activations=('swish','linear'))
    W1 = np.array(([1,1],[0.5,0.25],[0,0.75],[0.25,0.25]))
    W2 = np.array(([1,1,1],[1,1,0.5]))
    b1 = np.zeros(2)
    b2 = np.zeros(3)
    W = np.asarray((W1, W2),dtype=object)
    b = np.asarray((b1, b2),dtype=object)
    a.construct_parameters(method='manual', W=W, b=b)

    #analytical calculation
    input_to_layer_1 = X
    input_to_hidden_layer = input_to_layer_1.dot(W1)+ b1
    after_activation_of_hidden_layer = a.activation(input_to_hidden_layer, type='swish')
    input_to_final_layer = after_activation_of_hidden_layer.dot(W2) + b2
    output = a.activation(input_to_final_layer, type='linear')
    assert(a.forward_propagate()).all() == output.all()

def test_cost_function_mse_same_predicted_and_actual_outputs_return_zero_loss():
    y = np.ones(10)
    y_predicted = y
    assert((NeuralNetwork((4, 2, 1)).cost_functions(y_predicted,y)) == 0)

def test_loss_function_mse():   ##for one training example with 4 output elements
    y = np.array([1,2,6,4])
    y_predicted = ([0,0.5,6,3])
    mse = (1/4)*((y_predicted - y)**2).sum()
    assert((NeuralNetwork((4, 2, 4)).cost_functions(y_predicted,y)) == mse)

def test_cost_function_mse():
    y = np.array(([1,2], [2,3], [0.5,4]))
    y_predicted = ([0,0.5],[3,0.6],[1,2])
    mse = (1/2)*(1/3)*(((y_predicted - y)**2).sum()).sum()
    assert((NeuralNetwork((4, 2, 2)).cost_functions(y_predicted,y)) == mse)

def test_derivative_sigmoid():
    a = NeuralNetwork((4,2,1))
    x = np.ones(5)
    assert((a.derivatives(x, "sigmoid")).all()) == np.asarray([0.19661193, 0.19661193, 0.19661193, 0.19661193, 0.19661193]).all()

def test_derivative_swish():
    a = NeuralNetwork((4,2,1))
    x = np.ones(5)
    assert((a.derivatives(x, "swish")).all()) == np.asarray([0.92767051, 0.92767051, 0.92767051, 0.92767051, 0.92767051]).all()

def test_derivative_linear():
    a = NeuralNetwork((4,2,1))
    x = np.asarray([0,3,67,-45,2])
    assert((a.derivatives(x, "linear")).all()) == np.asarray([1., 1., 1., 1., 1.]).all()

def test_derivative_relu():
    a = NeuralNetwork((4,2,1))
    x = np.asarray([0,1,6,-3,6])
    assert((a.derivatives(x, "relu")).all()) == np.asarray([0, 1, 1, 0, 1]).all()

def test_derivative_leakyrelu():
    a = NeuralNetwork((4,2,1))
    x = np.asarray([0,1,6,-3,6])
    assert((a.derivatives(x, "leakyrelu")).all()) == np.asarray([0, 1, 1, 0.01, 1]).all()

def test_performance_metrics_mae():
    a = NeuralNetwork((4,2,1))
    y_true = [[0.5, 1], [-1, 1], [7, -6]]
    y_pred = [[0, 2], [-1, 2], [8, -5]]
    assert(a.performance_metrics(np.asarray(y_true),np.asarray( y_pred), 'mae')) == 0.75

def test_performance_metrics_mse():
    a = NeuralNetwork((4,2,1))
    y_true = [[0.5, 1], [-1, 1], [7, -6]]
    y_pred = [[0, 2], [-1, 2], [8, -5]]
    assert(a.performance_metrics(np.asarray(y_true),np.asarray( y_pred), 'mse')) == 0.7083333333333334

def test_performance_metrics_mape():
    a = NeuralNetwork((4,2,1))
    y_true = [[0.5, 1], [-1, 1], [7, -6]]
    y_pred = [[0, 2], [-1, 2], [8, -5]]
    assert(a.performance_metrics(np.asarray(y_true),np.asarray( y_pred), 'mape')) == 0.5515873015873015

def test_performance_metrics_msle():
    a = NeuralNetwork((4,2,1))
    y_true = [[0.5, 1], [1, 1], [7, 6]]
    y_pred = [[0, 2], [1, 2], [8, 5]]
    assert(a.performance_metrics(np.asarray(y_true),np.asarray( y_pred), 'msle')) == 0.08847352287652256

def test_performance_metrics_r2():
    a = NeuralNetwork((4,2,1))
    y_true = [[0.5, 1], [1, 1], [7, 6]]
    y_pred = [[0, 2], [1, 2], [8, 5]]
    assert(a.performance_metrics(np.asarray(y_true),np.asarray( y_pred), 'r2')) == 0.9416004707266843


def test_performance_metrics_rmse():
    a = NeuralNetwork((4,2,1))
    y_true = [[0.5, 1], [1, -1], [7, -6]]
    y_pred = [[0, -2], [1, 2], [8, 5]]
    assert(a.performance_metrics(np.asarray(y_true),np.asarray( y_pred), 'rmse')) == 4.834769901453429