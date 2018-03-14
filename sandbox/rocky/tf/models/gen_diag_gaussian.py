import numpy as np
import tensorflow as tf
from sandbox.rocky.tf.core.network import MLP
import sandbox.rocky.tf.core.layers as L
from sandbox.rocky.tf.distributions.diagonal_gaussian import DiagonalGaussian

class GenDiagGaussian:
    def __init__(
        self,
        name,
        abstract_dim,
        hidden_sizes=(32),
        min_std=1e-6,
        hidden_nonlinearity=tf.nn.tanh,
        output_nonlinearity=None,
        optim=tf.train.AdamOptimizer(learning_rate=0.001)
    ):
        self.obs_dim = abstract_dim
        self.min_std = min_std
        self.distribution = DiagonalGaussian(self.obs_dim)
        with tf.variable_scope(name):
            self.net = MLP(
                name="mu_log_sigma",
                input_shape=self.obs_dim,
                output_dim=2*self.obs_dim,
                hidden_sizes=hidden_sizes,
                hidden_nonlinearity=hidden_nonlinearity,
                output_nonlinearity=output_nonlinearity,
            )
        self.obs_var = self.net.input_layer.input_var
        self.output = L.get_output(self.net.output_layer, self.obs_var)
        self.mu, unstable_log_sigma = tf.split(self.output, [self.obs_dim, self.obs_dim], 1)
        self.log_sigma = tf.maximum(unstable_log_sigma, self.min_std)

        self.nexts = tf.placeholder(tf.float32, shape=(None, self.obs_dim))
        self.loss = -self.distribution.log_likelihood(self.nexts, dist_info=dict(mean=self.mu, log_stds=self.log_sigma))
        self.optimizer = optim
        self.train_op = optim.minimize(self.loss)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

    def forward(self, obs):
        return self.sess.run([self.mu, self.log_sigma], feed_dict={self.obs_var: obs})

    def get_loglikelihood(self, obs, next):
        mu, sigma = self.forward(obs)
        return self.distribution.log_likelihood(next, dist_info=dict(mean=mu, log_stds=sigma))

    def fit(self, obs, nexts, n_steps=25):
        for _ in range(n_steps):
            self.sess.run(self.train_op, feed_dict={self.obs_var: obs, self.nexts: nexts})
        return True
