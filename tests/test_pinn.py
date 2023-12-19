import sys
sys.path.insert(0, '.')
import numpy as np
import tensorflow as tf
import scipy.io
from scipy.interpolate import griddata

from navier_stokes_fdm.environment import Environment
from navier_stokes_pinn.PINN import PhysicsInformedNN
from navier_stokes_pinn.input_output import NavierStokesPINN_IO

np.random.seed(1234)
tf.random.set_seed(1234)

def test_PINN_IO_initialization():
    IO_manager = NavierStokesPINN_IO(input_path="navier_stokes_pinn/data", output_path="navier_stokes_pinn/output")
    
    assert IO_manager.input_path == "navier_stokes_pinn/data", "Input path should be ../navier_stokes_pinn/data"
    assert IO_manager.output_path == "navier_stokes_pinn/output", "Output path should be ../navier_stokes_pinn/output"

    assert IO_manager.training_data == dict(), "Training data should be an empty dictionary"
    assert IO_manager.test_data == dict(), "Test data should be an empty dictionary"
    assert IO_manager.predict_data == dict(), "Predict data should be an empty dictionary"
    assert IO_manager.multi_predict_data == dict(), "Multi predict data should be an empty dictionary"

    assert IO_manager.parsed == False, "Data should not be parsed yet"
    assert IO_manager.test_data_prepared == False, "Test data should not be prepared yet"

def test_PINN_IO_parse_data_file():
    IO_manager = NavierStokesPINN_IO(input_path="navier_stokes_pinn/data", output_path="navier_stokes_pinn/output")
    IO_manager.parse_data_file(filename="cylinder_nektar_wake.mat")

    # Checking the data input is correct
    assert IO_manager.U_star.shape == (5000, 2, 200), "U_star should be a 3D array with shape (5000, 2, 200)"
    assert IO_manager.P_star.shape == (5000, 200), "p_star should be a 2D array with shape (5000, 200)"
    assert IO_manager.t_star.shape == (200, 1), "t_star should be a 2D array with shape (200, 1)"
    assert IO_manager.X_star.shape == (5000, 2), "X_star should be a 2D array with shape (5000, 2)"

    # Checking the parse result is correct
    assert IO_manager.x.shape == (100000, 1), "x should be (100000, 1)"
    assert IO_manager.y.shape == (100000, 1), "y should be (100000, 1)"
    assert IO_manager.t.shape == (100000, 1), "t should be (100000, 1)"
    assert IO_manager.u.shape == (100000, 1), "u should be (100000, 1)"
    assert IO_manager.v.shape == (100000, 1), "v should be (100000, 1)"
    assert IO_manager.p.shape == (100000, 1), "p should be (100000, 1)"

    assert IO_manager.parsed == True, "Data should be parsed now"

def test_pinn_initialization():
    IO_manager = NavierStokesPINN_IO(input_path="navier_stokes_pinn/data", output_path="navier_stokes_pinn/output")
    IO_manager.parse_data_file(filename="cylinder_nektar_wake.mat")
    IO_manager.select_training_data(N_train=50)

    training_data = IO_manager.training_data
    # Casting the training data into tensorflow
    x_train = tf.cast(training_data['x_train'], dtype=tf.float32)
    y_train = tf.cast(training_data['y_train'], dtype=tf.float32)
    t_train = tf.cast(training_data['t_train'], dtype=tf.float32)
    u_train = tf.cast(training_data['u_train'], dtype=tf.float32)
    v_train = tf.cast(training_data['v_train'], dtype=tf.float32)

    layers = [3, 20, 20, 20, 20, 20, 20, 20, 20, 2]
    Re = 100
    model = PhysicsInformedNN(x_train, y_train, t_train, u_train, v_train, Re, layers)
    model_loss = model.loss()

    assert model.x.shape == (50, 1), "x should be (50, 1)"
    assert model.y.shape == (50, 1), "y should be (50, 1)"
    assert model.t.shape == (50, 1), "t should be (50, 1)"
    assert model.u.shape == (50, 1), "u should be (50, 1)"
    assert model.v.shape == (50, 1), "v should be (50, 1)"

    assert model.lb.shape == (3,), "lb should be (3,)"
    assert model.ub.shape == (3,), "ub should be (3,)"

    assert len(model.weights) == 9, "Model should have 9 lists of weights"
    assert all(isinstance(weight, tf.Variable) for weight in model.weights), "Model weights should be tf.Variable"

    assert len(model.biases) == 9, "Model should have 9 lists of biases"
    assert all(isinstance(bias, tf.Variable) for bias in model.biases), "Model biases should be tf.Variable"

    assert isinstance(model_loss, tf.Tensor), "Model loss should be a tensor"

    















