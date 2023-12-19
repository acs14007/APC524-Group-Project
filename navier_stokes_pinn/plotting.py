import scipy.io
import os
from scipy.interpolate import griddata
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class NavierStokesPINN_Plotter():
    def __init__(self, data_path: str, plot_path: str, IO_class):
        self.data_path = data_path
        self.plot_path = plot_path
        
        # Having the IO class will help us with the peripheral data for the plots
        self.IO_class = IO_class 
        if not self.IO_class.test_data_prepared:
            raise Exception("Test data has not been prepared. Run the method select_data_file() from NavierStokesPINN_IO first.")
        
    def plot_compare(self, predicted_grid, exact_grid, time_snap, extent: list, quantity_name: str):
        fig, ax = plt.subplots(2, figsize=(12,12))

        ax[0].set_title('Predicted ' + quantity_name + ' (timestep = %d)' %time_snap, fontsize = 20)
        ax[0].set_xlabel(xlabel='x' ,size=15)
        ax[0].set_ylabel(ylabel='y', size=15)
        pred = ax[0].imshow(predicted_grid, interpolation='nearest', cmap='rainbow', 
                    extent=extent, 
                    origin='lower', aspect='auto')
        fig.colorbar(pred)

        ax[1].set_title('Exact ' + quantity_name + ' (timestep = %d)' %time_snap, fontsize = 20)
        ax[1].set_xlabel(xlabel='x' ,size=15)
        ax[1].set_ylabel(ylabel='y', size=15)
        real = ax[1].imshow(exact_grid, interpolation='nearest', cmap='rainbow', 
                    extent=extent, 
                    origin='lower', aspect='auto')
        fig.colorbar(real)
    
    def plot_compare_predictions(self, data_filename: str, time_snap_idx: int):
        plot_data = np.load(os.path.join(self.data_path, data_filename))

        X_star = self.IO_class.X_star
        x_star = X_star[:,0:1]
        y_star = X_star[:,1:2]

        lb = X_star.min(0)
        ub = X_star.max(0)
        nn = 200
        x = np.linspace(lb[0], ub[0], nn)
        y = np.linspace(lb[1], ub[1], nn)
        X, Y = np.meshgrid(x,y)    
        plot_limit =  [x_star.min(), x_star.max(), y_star.min(), y_star.max()]
        
        if len(plot_data) < 7: #This is when we only have predictions for one timestep
            u_pred = plot_data['u_pred']
            v_pred = plot_data['v_pred']
            p_pred = plot_data['p_pred']

            u_exact = plot_data['u_exact']
            v_exact = plot_data['v_exact']
            p_exact = plot_data['p_exact']

            UU_star = griddata(X_star, u_pred.flatten(), (X, Y), method='cubic')    
            VV_star = griddata(X_star, v_pred.flatten(), (X, Y), method='cubic')
            PP_star = griddata(X_star, p_pred.flatten(), (X, Y), method='cubic')
            U_exact = griddata(X_star, u_exact.flatten(), (X, Y), method='cubic')
            V_exact = griddata(X_star, v_exact.flatten(), (X, Y), method='cubic')
            P_exact = griddata(X_star, p_exact.flatten(), (X, Y), method='cubic')

            self.plot_compare(UU_star, U_exact, time_snap_idx, plot_limit, 'Velocity Component u')
            plt.savefig(os.path.join(self.plot_path, 'u_%d.png' %time_snap_idx))

            self.plot_compare(VV_star, V_exact, time_snap_idx, plot_limit, 'Velocity Component v')
            plt.savefig(os.path.join(self.plot_path, 'v_%d.png' %time_snap_idx))

            self.plot_compare(PP_star, P_exact, time_snap_idx, plot_limit, 'Pressure')
            plt.savefig(os.path.join(self.plot_path, 'p_%d.png' %time_snap_idx))

        else: #This is when we have predictions for multiple timesteps
            u_pred = plot_data['u_pred_%d' %time_snap_idx]
            v_pred = plot_data['v_pred_%d' %time_snap_idx]
            p_pred = plot_data['p_pred_%d' %time_snap_idx]

            u_exact = plot_data['u_exact_%d' %time_snap_idx]
            v_exact = plot_data['v_exact_%d' %time_snap_idx]
            p_exact = plot_data['p_exact_%d' %time_snap_idx]

            # u_exact = self.IO_class.u_star
            # v_exact = self.IO_class.v_star
            # p_exact = self.IO_class.p_star

            UU_star = griddata(X_star, u_pred.flatten(), (X, Y), method='cubic')    
            VV_star = griddata(X_star, v_pred.flatten(), (X, Y), method='cubic')
            PP_star = griddata(X_star, p_pred.flatten(), (X, Y), method='cubic')
            U_exact = griddata(X_star, u_exact.flatten(), (X, Y), method='cubic')
            V_exact = griddata(X_star, v_exact.flatten(), (X, Y), method='cubic')
            P_exact = griddata(X_star, p_exact.flatten(), (X, Y), method='cubic')

            self.plot_compare(UU_star, U_exact, time_snap_idx, plot_limit, 'Velocity Component u')
            plt.savefig(os.path.join(self.plot_path, 'u_%d.png' %time_snap_idx))

            self.plot_compare(VV_star, V_exact, time_snap_idx, plot_limit, 'Velocity Component v')
            plt.savefig(os.path.join(self.plot_path, 'v_%d.png' %time_snap_idx))

            self.plot_compare(PP_star, P_exact, time_snap_idx, plot_limit, 'Pressure')
            plt.savefig(os.path.join(self.plot_path, 'p_%d.png' %time_snap_idx))






