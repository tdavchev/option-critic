
import tensorflow as tf
import tensorflow.contrib.slim as slim

class OptionsNetwork(object):
    def __init__(self, sess, h_size, temp, state_dim, action_dim,
                 option_dim, learning_rate, tau, gamma, entropy_reg=0.01, clip_delta=0):
        self.sess = sess
        self.h_size = h_size
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.o_dim = option_dim
        # self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.tau = tau
        self.temp = temp

        # State Network
        self.inputs = tf.placeholder(
            shape=[None, 84, 84, 4], dtype=tf.uint8, name="inputs")
        scaled_image = tf.to_float(self.inputs) / 255.0
        self.next_inputs = tf.placeholder(
            shape=[None, 84, 84, 4],
            dtype=tf.uint8,
            name="next_inputs")
        next_scaled_image = tf.to_float(self.next_inputs) / 255.0
        with tf.variable_scope("state_out") as scope:
            self.state_out = self.apply_state_model(scaled_image)
            scope.reuse_variables()
            self.next_state_out = self.apply_state_model(next_scaled_image)

        with tf.variable_scope("q_out") as q_scope:
            # self.state_out = self.create_state_network(scaledImage)
            self.Q_out = self.apply_q_model(self.state_out)
            q_scope.reuse_variables()
            self.next_Q_out = self.apply_q_model(self.next_state_out)

        self.network_params = tf.trainable_variables()[:-2]
        self.Q_params = tf.trainable_variables()[-2:]

        # Prime Network
        self.target_inputs = tf.placeholder(
            shape=[None, 84, 84, 4],
            dtype=tf.uint8,
            name="target_inputs")
        target_scaled_image = tf.to_float(self.target_inputs) / 255.0
        self.target_state_out_holder = self.create_state_network(
            target_scaled_image)
        with tf.variable_scope("target_q_out") as target_q_scope:
            self.target_Q_out = self.apply_target_q_model(
                self.target_state_out_holder)
            target_q_scope.reuse_variables()
        self.target_network_params = tf.trainable_variables()[
            len(self.network_params) + len(self.Q_params):-2]
        self.target_Q_params = tf.trainable_variables()[-2:]

        # Op for periodically updating target network with online network
        # weights
        # self.update_target_network_params = \
        #     [self.target_network_params[i].assign(self.network_params[i])
        #      for i in range(len(self.target_network_params))]
        self.update_target_network_params = \
            [self.target_network_params[i].assign(
                tf.multiply(self.network_params[i], self.tau) +
                tf.multiply(self.target_network_params[i], 1. - self.tau))
             for i in range(len(self.target_network_params))]

        # self.update_target_q_params = \
        #     [self.target_Q_params[i].assign(self.Q_params[i])
        #      for i in range(len(self.target_Q_params))]
        self.update_target_q_params = \
            [self.target_Q_params[i].assign(
                tf.multiply(self.Q_params[i], self.tau) +
                tf.multiply(self.target_Q_params[i], 1. - self.tau))
             for i in range(len(self.target_Q_params))]

        # gather_nd should also do, though this is sexier
        self.option = tf.placeholder(tf.int32, [None, 1], name="option")
        self.action = tf.placeholder(
            shape=[None, 1], dtype=tf.int32, name="action")
        self.actions_onehot = tf.squeeze(tf.one_hot(
            self.action, self.a_dim, dtype=tf.float32), [1])
        self.options_onehot = tf.squeeze(tf.one_hot(
            self.option, self.o_dim, dtype=tf.float32), [1])

        # Action Network
        self.action_input = tf.concat(
            [self.state_out, self.state_out, self.state_out, self.state_out,
             self.state_out, self.state_out, self.state_out, self.state_out], 1)
        self.action_input = tf.reshape(
            self.action_input, shape=[-1, self.o_dim, 1, self.h_size])
        oh = tf.reshape(self.options_onehot, shape=[-1, self.o_dim, 1])
        self.action_input = tf.reshape(
            tf.reduce_sum(
                tf.squeeze(
                    self.action_input, [2]) * oh, [1]),
            shape=[-1, 1, self.h_size])

        self.action_probs = tf.contrib.layers.fully_connected(
            inputs=self.action_input,
            num_outputs=self.a_dim,
            activation_fn=tf.nn.softmax)
        self.action_probs = tf.squeeze(self.action_probs, [1])
        self.action_params = tf.trainable_variables()[
            len(self.network_params) + len(self.target_network_params) +
            len(self.Q_params) + len(self.target_Q_params):]
        # always draws 0 ...
        # self.sampled_action = tf.argmax(tf.multinomial(self.action_probs, 1), axis=1)

        # Termination Network
        with tf.variable_scope("termination_probs") as term_scope:
            self.termination_probs = self.apply_termination_model(
                tf.stop_gradient(self.state_out))
            term_scope.reuse_variables()
            self.next_termination_probs = self.apply_termination_model(
                tf.stop_gradient(self.next_state_out))

        self.termination_params = tf.trainable_variables()[-2:]

        self.option_term_prob = tf.reduce_sum(
            self.termination_probs * self.options_onehot, [1])
        self.next_option_term_prob = tf.reduce_sum(
            self.next_termination_probs * self.options_onehot, [1])

        self.reward = tf.placeholder(tf.float32, [None, 1], name="reward")
        self.done = tf.placeholder(tf.float32, [None, 1], name="done")
        # self.disc_option_term_prob = tf.placeholder(tf.float32, [None, 1])

        disc_option_term_prob = tf.stop_gradient(self.next_option_term_prob)

        y = tf.squeeze(self.reward, [1]) + \
            tf.squeeze((1 - self.done), [1]) * \
            gamma * (
                (1 - disc_option_term_prob) *
                tf.reduce_sum(self.target_Q_out * self.options_onehot, [1]) +
                disc_option_term_prob *
                tf.reduce_max(self.target_Q_out, reduction_indices=[1]))

        y = tf.stop_gradient(y)

        option_Q_out = tf.reduce_sum(self.Q_out * self.options_onehot, [1])
        td_errors = y - option_Q_out
        # self.td_errors = tf.squared_difference(self.y, self.option_Q_out)

        if clip_delta > 0:
            quadratic_part = tf.minimum(abs(td_errors), clip_delta)
            linear_part = abs(td_errors) - quadratic_part
            td_cost = 0.5 * quadratic_part ** 2 + clip_delta * linear_part
        else:
            td_cost = 0.5 * td_errors ** 2

        # critic updates
        self.critic_cost = tf.reduce_sum(td_cost)
        critic_params = self.network_params + self.Q_params
        grads = tf.gradients(self.critic_cost, critic_params)
        self.critic_updates = tf.train.AdamOptimizer().apply_gradients(zip(grads, critic_params))
        # self.critic_updates = tf.train.RMSPropOptimizer(
        #     self.learning_rate, decay=0.95, epsilon=0.01).apply_gradients(zip(grads, critic_params))

        # actor updates
        self.value = tf.stop_gradient(
            tf.reduce_max(self.Q_out, reduction_indices=[1]))
        self.disc_q = tf.stop_gradient(tf.reduce_sum(
            self.Q_out * self.options_onehot, [1]))
        self.picked_action_prob = tf.reduce_sum(
            self.action_probs * self.actions_onehot, [1])
        actor_params = self.termination_params + self.action_params
        entropy = - tf.reduce_sum(self.action_probs *
                                  tf.log(self.action_probs))
        policy_gradient = - tf.reduce_sum(tf.log(self.picked_action_prob) * y) - \
            entropy_reg * entropy
        self.term_gradient = tf.reduce_sum(
            self.option_term_prob * (self.disc_q - self.value))
        self.loss = self.term_gradient + policy_gradient
        grads = tf.gradients(self.loss, actor_params)
        self.actor_updates = tf.train.AdamOptimizer().apply_gradients(zip(grads, actor_params))
        # self.actor_updates = tf.train.RMSPropOptimizer(
        #     self.learning_rate, decay=0.95, epsilon=0.01).apply_gradients(zip(grads, actor_params))

    def apply_state_model(self, input_image):
        with tf.variable_scope("input"):
            output = self.state_model(
                input_image,
                [[8, 8, 4, 32], [4, 4, 32, 64], [3, 3, 64, 64]],
                [[3136, 512]])
        return output

    def apply_q_model(self, input):
        with tf.variable_scope("q_input"):
            output = self.q_model(input, [self.h_size, self.o_dim])
        return output

    def apply_target_q_model(self, input):
        with tf.variable_scope("target_q_input"):
            output = self.target_q_model(input, [self.h_size, self.o_dim])
        return output

    def apply_termination_model(self, input):
        with tf.variable_scope("term_input"):
            output = self.termination_model(input, [self.h_size, self.o_dim])
        return output

    def state_model(self, input, kernel_shapes, weight_shapes):
        weights1 = tf.get_variable(
            "weights1", kernel_shapes[0],
            initializer=tf.contrib.layers.xavier_initializer())
        weights2 = tf.get_variable(
            "weights2", kernel_shapes[1],
            initializer=tf.contrib.layers.xavier_initializer())
        weights3 = tf.get_variable(
            "weights3", kernel_shapes[2],
            initializer=tf.contrib.layers.xavier_initializer())
        weights4 = tf.get_variable(
            "weights5", weight_shapes[0],
            initializer=tf.contrib.layers.xavier_initializer())
        bias1 = tf.get_variable(
            "q_bias1", weight_shapes[0][1],
            initializer=tf.constant_initializer())
        # Convolve
        conv1 = tf.nn.relu(tf.nn.conv2d(
            input, weights1, strides=[1, 4, 4, 1], padding='VALID'))
        conv2 = tf.nn.relu(tf.nn.conv2d(
            conv1, weights2, strides=[1, 2, 2, 1], padding='VALID'))
        conv3 = tf.nn.relu(tf.nn.conv2d(
            conv2, weights3, strides=[1, 1, 1, 1], padding='VALID'))
        # Flatten and Feedforward
        flattened = tf.contrib.layers.flatten(conv3)
        net = tf.nn.relu(tf.nn.xw_plus_b(flattened, weights4, bias1))

        return net

    def target_q_model(self, input, weight_shape):
        weights1 = tf.get_variable(
            "target_q_weights1", weight_shape,
            initializer=tf.contrib.layers.xavier_initializer())
        bias1 = tf.get_variable(
            "target_q_bias1", weight_shape[1],
            initializer=tf.constant_initializer())
        return tf.nn.xw_plus_b(input, weights1, bias1)

    def q_model(self, input, weight_shape):
        weights1 = tf.get_variable(
            "q_weights1", weight_shape,
            initializer=tf.contrib.layers.xavier_initializer())
        bias1 = tf.get_variable(
            "q_bias1", weight_shape[1],
            initializer=tf.constant_initializer())
        return tf.nn.xw_plus_b(input, weights1, bias1)

    def termination_model(self, input, weight_shape):
        weights1 = tf.get_variable(
            "term_weights1", weight_shape,
            initializer=tf.contrib.layers.xavier_initializer())
        bias1 = tf.get_variable(
            "term_bias1", weight_shape[1],
            initializer=tf.constant_initializer())
        return tf.nn.sigmoid(tf.nn.xw_plus_b(input, weights1, bias1))

    def create_state_network(self, scaledImage):
        # Convolve
        conv1 = slim.conv2d(
            inputs=scaledImage, num_outputs=32, kernel_size=[8, 8], stride=[4, 4],
            padding='VALID', biases_initializer=None)
        conv2 = slim.conv2d(
            inputs=conv1, num_outputs=64, kernel_size=[4, 4], stride=[2, 2],
            padding='VALID', biases_initializer=None)
        conv3 = slim.conv2d(
            inputs=conv2, num_outputs=64, kernel_size=[3, 3], stride=[1, 1],
            padding='VALID', biases_initializer=None)
        # Flatten and Feedforward
        flattened = tf.contrib.layers.flatten(conv3)
        net = tf.contrib.layers.fully_connected(
            inputs=flattened,
            num_outputs=self.h_size,
            activation_fn=tf.nn.relu)

        return net

    def predict(self, inputs):
        return self.sess.run(self.Q_out, feed_dict={
            self.inputs: [inputs]
        })

    def predict_action(self, inputs, option):
        return self.sess.run(self.action_probs, feed_dict={
            self.inputs: inputs,
            self.option: option
        })

    def predict_termination(self, inputs, option):
        return self.sess.run(
            [self.option_term_prob, self.Q_out],
            feed_dict={
                self.inputs: inputs,
                self.option: option
            })

    def train_actor(self, inputs, target_inputs, options, actions, r, done):
        return self.sess.run(self.actor_updates, feed_dict={
            self.inputs: inputs,
            self.next_inputs: target_inputs,
            self.target_inputs: target_inputs,
            self.option: options,
            self.action: actions,
            self.reward: r,
            self.done: done
        })

    def train_critic(self, inputs, target_inputs, options, r, done):
        return self.sess.run([self.critic_cost, self.critic_updates], feed_dict={
            self.inputs: inputs,
            self.next_inputs: target_inputs,
            self.target_inputs: target_inputs,
            self.reward: r,
            self.option: options,
            self.done: done
        })

    def update_target_network(self):
        self.sess.run([self.update_target_network_params,
                       self.update_target_q_params])