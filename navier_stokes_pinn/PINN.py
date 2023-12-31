try:
    from abc import ABC, abstractmethod
    import sys
    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.io
    from scipy.interpolate import griddata
    import time
except:
    print("One or more packages are missing...")

np.random.seed(1234)
tf.random.set_seed(1234)

print(tf.__version__)

class NeuralNetwork(ABC):
    @abstractmethod
    def initialize_NN(self, layers):
        pass

    @abstractmethod
    def neural_net(self, X, weights, biases):
        pass

    @abstractmethod
    def loss(self):
        pass

    @abstractmethod
    def train(self, nIter: int, learning_rate: float):
        pass

    @abstractmethod
    def predict(self, X_star):
        pass

class PhysicsInformedNN:

    def __init__(self, x, y, t, u, v, Re, layers):

        X = np.concatenate([x, y, t], 1)

        self.lb = X.min(0)
        self.ub = X.max(0)

        self.X = X

        self.x = x
        self.y = y
        self.t = t

        self.u = u
        self.v = v

        self.Re = Re

        self.layers = layers

        # Initialize the NN
        self.weights, self.biases = self.initialize_NN(layers)

        # Create a list including all training variables
        self.train_variables = self.weights + self.biases
        # Key point: anything updates in train_variables will be
        #            automatically updated in the original tf.Variable

    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.random.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1

        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_psi_p(self, x, y, t):
        psi_p = self.neural_net(tf.concat([x,y,t], 1), self.weights, self.biases)
        return psi_p

    def net_NS(self, x, y, t):

        psi_and_p = self.net_psi_p(x, y, t)
        psi = psi_and_p[:,0:1]
        p = psi_and_p[:,1:2]

        u = tf.gradients(psi, y)[0]
        v = -tf.gradients(psi, x)[0]

        u_t = tf.gradients(u, t)[0]
        u_x = tf.gradients(u, x)[0]
        u_y = tf.gradients(u, y)[0]
        u_xx = tf.gradients(u_x, x)[0]
        u_yy = tf.gradients(u_y, y)[0]

        v_t = tf.gradients(v, t)[0]
        v_x = tf.gradients(v, x)[0]
        v_y = tf.gradients(v, y)[0]
        v_xx = tf.gradients(v_x, x)[0]
        v_yy = tf.gradients(v_y, y)[0]

        p_x = tf.gradients(p, x)[0]
        p_y = tf.gradients(p, y)[0]

        Re = self.Re

        # Functions describing the NS equation

        f_u = u_t + (u*u_x + v*u_y) + p_x - (1/Re)*(u_xx + u_yy)
        f_v = v_t + (u*v_x + v*v_y) + p_y - (1/Re)*(v_xx + v_yy)

        return u, v, p, f_u, f_v

    @tf.function
    # Loss function for the entire PINN
    def loss(self):
        self.u_pred, self.v_pred, self.p_pred, self.f_u_pred, self.f_v_pred = \
            self.net_NS(self.x, self.y, self.t)

        loss = tf.reduce_sum(tf.square(self.u - self.u_pred)) +\
               tf.reduce_sum(tf.square(self.v - self.v_pred)) +\
               tf.reduce_sum(tf.square(self.f_u_pred)) +\
               tf.reduce_sum(tf.square(self.f_v_pred))

        return loss

    def train(self, nIter: int, learning_rate: float):
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        varlist = self.weights + self.biases
        start_time = time.time()

        for it in range(nIter):
            optimizer.minimize(self.loss, varlist)

            # Print training updates
            if it % 100 == 0:
                elapsed = time.time() - start_time
                loss = self.loss().numpy()
                print('It: %d, Train Loss: %.3e, Time: %.2f' % (it, loss, elapsed), flush=True)
                start_time = time.time()

    @tf.function
    def predict(self, x_star, y_star, t_star):
        u_star, v_star, p_star, _, _ = self.net_NS(x_star, y_star, t_star)
        return u_star, v_star, p_star
        
#     def set_weights(self, new_weights, new_biases):
#         self.weights = [tf.Variable(weight) for weight in new_weights]
#         self.biases = [tf.Variable(bias) for bias in new_biases]