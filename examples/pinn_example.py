import sys
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR) #Gets rid of intrusive warnings
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
import time

from navier_stokes_pinn.PINN import PhysicsInformedNN
from navier_stokes_pinn.input_output import NavierStokesPINN_IO
from navier_stokes_pinn.plotting import NavierStokesPINN_Plotter


IO_manager = NavierStokesPINN_IO("navier_stokes_pinn/data", "navier_stokes_pinn/output")
IO_manager.parse_data_file('cylinder_nektar_wake.mat')

# Selecting training data
IO_manager.select_training_data(N_train=5000)

# Extract training data from IO_manager
training_data = IO_manager.training_data

# Casting the training data into tensorflow
x_train = tf.cast(training_data['x_train'], dtype=tf.float32)
y_train = tf.cast(training_data['y_train'], dtype=tf.float32)
t_train = tf.cast(training_data['t_train'], dtype=tf.float32)
u_train = tf.cast(training_data['u_train'], dtype=tf.float32)
v_train = tf.cast(training_data['v_train'], dtype=tf.float32)

# Setting model architechture
layers = [3, 20, 20, 20, 20, 20, 20, 20, 20, 2]

# Setting Reynold's Number
Re = 100

# Initializing the PINN model
# Model training support TensorFlow 2 and GPU acceleration
model = PhysicsInformedNN(x_train, y_train, t_train, u_train, v_train, Re, layers)

# Train PINN model
model.train(2000, learning_rate=1e-3)

# Select test data at a time snapshot to test run inference and test the trained model
time_snapshot = 100
IO_manager.select_test_data(time_snapshot)
test_data = IO_manager.test_data

# Run inference and save predicted data
IO_manager.save_predict_data(model, 'example_prediction_100.npz')

# Plot the predicted data and save the plot with plotting class
Plot_manager = NavierStokesPINN_Plotter("navier_stokes_pinn/data", "navier_stokes_pinn/plots", IO_manager)
# Saves the u, v. and p values at the timestep 100
Plot_manager.plot_compare_predictions('example_prediction_100.npz', time_snapshot)