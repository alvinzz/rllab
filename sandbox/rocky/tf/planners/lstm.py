import tensorflow as tf
import sandbox.rocky.tf.core.layers as L
from sandbox.rocky.tf.core.network import LSTMNetwork
from rllab.envs.mujoco.half_cheetah_env import HalfCheetahTargEnv

class LSTMplanner:
    def __init__(
        self,
        name,
        model,
        abstract_dim,
        reward_fn=None,
        hidden_dim=32,
        hidden_nonlinearity=tf.tanh,
        output_nonlinearity=None,
        lstm_layer_cls=L.LSTMLayer,
    ):
        # possible to pass in reward_fn?
        with tf.variable_scope(name):
            self.obs_dim = abstract_dim
            self.net = LSTMNetwork(
                input_shape=self.obs_dim,
                input_layer=l_feature,
                output_dim=self.obs_dim,
                hidden_dim=hidden_dim,
                hidden_nonlinearity=hidden_nonlinearity,
                output_nonlinearity=output_nonlinearity,
                lstm_layer_cls=lstm_layer_cls,
                name="planner",
            )
            self.obs_var = self.net.input_layer.input_var
            self.output = L.get_output(self.net.output_layer, self.obs_var)

            env = HalfCheetahTargEnv()
            target_init = tf.constant(env.TARGET)
            target = tf.get_variable('target', initializer=init, trainable=False)
            self.loss = -self.model.get_loglikelihood(self.obs_var, self.output) * tf.norm(target - self.output)

            self.optimizer = optim
            self.train_op = optim.minimize(self.loss)

            config = tf.ConfigProto()
            config.gpu_options.allow_growth=True
            self.sess = tf.Session(config=config)
            self.sess.run(tf.global_variables_initializer())

    def forward(self, obs):
        return self.sess.run(self.output, feed_dict={self.obs_var: obs})

    def train(self, obs, n_steps=25):
        for _ in range(n_steps):
            self.sess.run(self.train_op, feed_dict={self.obs_var: obs})
        return True
