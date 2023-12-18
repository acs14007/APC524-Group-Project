import scipy.io
import os
from scipy.interpolate import griddata
from PINN import PhysicsInformedNN
import numpy as np
import tensorflow as tf

class NavierStokesPINN_IO():

    def __init__(self, input_path: str, output_path: str):

        self.input_path = input_path
        self.output_path = output_path

        self.training_data = dict()
        self.test_data = dict()
        self.predict_data = dict()

    def load_data_file(self, filename: str):
        self.data = scipy.io.loadmat(os.path.join(self.input_path, filename))

    def prepare_data(self):
        self.U_star = self.data['U_star'] # N x 2 x T
        self.P_star = self.data['p_star'] # N x T
        self.t_star = self.data['t'] # T x 1
        self.X_star = self.data['X_star'] # N x 2

        self.N = self.X_star.shape[0]
        self.T = self.t_star.shape[0]

        # Rearrange Data
        self.XX = np.tile(self.X_star[:,0:1], (1,self.T)) # N x T
        self.YY = np.tile(self.X_star[:,1:2], (1,self.T)) # N x T
        self.TT = np.tile(self.t_star, (1,self.N)).T # N x T

        self.UU = self.U_star[:,0,:] # N x T
        self.VV = self.U_star[:,1,:] # N x T
        self.PP = self.P_star # N x T

        self.x = self.XX.flatten()[:,None] # NT x 1
        self.y = self.YY.flatten()[:,None] # NT x 1
        self.t = self.TT.flatten()[:,None] # NT x 1

        self.u = self.UU.flatten()[:,None] # NT x 1
        self.v = self.VV.flatten()[:,None] # NT x 1
        self.p =self.PP.flatten()[:,None] # NT x 1

    def select_training_data(self, N_train: int):
        idx = np.random.choice(self.X_star.shape[0], N_train, replace=False)
        self.training_data['x_train'] = self.x[idx,:]
        self.training_data['y_train'] = self.y[idx,:]
        self.training_data['t_train'] = self.t[idx,:]
        self.training_data['u_train'] = self.u[idx,:]
        self.training_data['v_train'] = self.v[idx,:]

    def select_test_data(self, time_snap: float):
        snap = np.array([time_snap])
        self.x_star = self.X_star[:,0:1]
        self.y_star = self.X_star[:,1:2]
        self.t_star = self.TT[:,snap]

        self.u_star = self.U_star[:,0,snap]
        self.v_star = self.U_star[:,1,snap]
        self.p_star = self.P_star[:,snap]

        x_test = tf.cast(self.x_star, dtype=tf.float32)
        y_test = tf.cast(self.y_star, dtype=tf.float32)
        t_test = tf.cast(self.t_star, dtype=tf.float32)

        self.test_data['x_test'] = x_test
        self.test_data['y_test'] = y_test
        self.test_data['t_test'] = t_test

    def save_predict_data(self, trained_model):
        u_pred, v_pred, p_pred = trained_model.predict(self.test_data['x_test'], self.test_data['y_test'], self.test_data['t_test'])

    



        