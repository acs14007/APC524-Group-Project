import scipy.io
import os
from scipy.interpolate import griddata
import numpy as np
import tensorflow as tf

class NavierStokesPINN_IO():

    def __init__(self, input_path: str, output_path: str):

        self.input_path = input_path
        self.output_path = output_path

        self.training_data = dict()
        self.test_data = dict()
        self.predict_data = dict()
        self.multi_predict_data = dict()

        self.parsed = False
        self.test_data_prepared = False

    def parse_data_file(self, filename: str):
        self.data = scipy.io.loadmat(os.path.join(self.input_path, filename))

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

        self.parsed = True

    def select_training_data(self, N_train: int):
        if not self.parsed:
            raise Exception("Data file not parsed yet! Run parse_data_file() first.")
        
        idx = np.random.choice(self.N*self.T, N_train, replace=False)
        self.training_data['x_train'] = self.x[idx,:]
        self.training_data['y_train'] = self.y[idx,:]
        self.training_data['t_train'] = self.t[idx,:]
        self.training_data['u_train'] = self.u[idx,:]
        self.training_data['v_train'] = self.v[idx,:]

    def select_test_data(self, time_snap_idx: int):
        if not self.parsed:
            raise Exception("Data file not parsed yet! Run parse_data_file() first.")
        
        snap = np.array([time_snap_idx])
        self.x_star = self.X_star[:,0:1]
        self.y_star = self.X_star[:,1:2]
        self.t_star = self.TT[:,snap]

        self.u_star = self.U_star[:,0,snap]
        self.v_star = self.U_star[:,1,snap]
        self.p_star = self.P_star[:,snap]

        self.test_data['x_test'] = self.x_star
        self.test_data['y_test'] = self.y_star
        self.test_data['t_test'] = self.t_star

        self.test_data_prepared = True

    def save_predict_data(self, trained_model, predict_filename: str = None):
        if not self.test_data_prepared:
            raise Exception("Test data has not been prepared. Run select_data_file() first.")
               
        x_test = tf.cast(self.test_data['x_test'], dtype=tf.float32)
        y_test = tf.cast(self.test_data['y_test'], dtype=tf.float32)
        t_test = tf.cast(self.test_data['t_test'], dtype=tf.float32)
        u_pred, v_pred, p_pred = trained_model.predict(x_test, y_test, t_test)
        self.predict_data['u_pred'] = u_pred.numpy()
        self.predict_data['v_pred'] = v_pred.numpy()
        self.predict_data['p_pred'] = p_pred.numpy()
        self.predict_data['u_exact'] = self.u_star
        self.predict_data['v_exact'] = self.v_star
        self.predict_data['p_exact'] = self.p_star
    

        if predict_filename is not None:
            np.savez(os.path.join(self.output_path, predict_filename), **self.predict_data)

    def save_multi_predict_data(self, trained_model, time_snap_arr, predict_filename: str):
        if not self.parsed:
            raise Exception("Data file not parsed yet! Run parse_data_file() first.")
        
        for i in time_snap_arr:
            snap_num = i
            snap = np.array([snap_num])
            x_star = self.X_star[:,0:1]
            y_star = self.X_star[:,1:2]
            t_star = self.TT[:,snap]

            u_star = self.U_star[:,0,snap]
            v_star = self.U_star[:,1,snap]
            p_star = self.P_star[:,snap]

            x_test = tf.cast(x_star, dtype=tf.float32)
            y_test = tf.cast(y_star, dtype=tf.float32)
            t_test = tf.cast(t_star, dtype=tf.float32)

            # Prediction
            u_pred, v_pred, p_pred = trained_model.predict(x_test, y_test, t_test)
            self.multi_predict_data['u_pred_%d' %i] = u_pred.numpy()
            self.multi_predict_data['v_pred_%d' %i] = v_pred.numpy()
            self.multi_predict_data['p_pred_%d' %i] = p_pred.numpy()
            self.multi_predict_data['u_exact_%d' %i] = u_star
            self.multi_predict_data['v_exact_%d' %i] = v_star
            self.multi_predict_data['p_exact_%d' %i] = p_star
            
        np.savez(os.path.join(self.output_path, predict_filename), **self.multi_predict_data)






        