import numpy as np
import tensorflow as tf
import random
import tensorflow.layers as layer
from collections import deque
import random
import datetime
import time
from multiagent.environment import MultiAgentEnv
from multiagent.policy import InteractivePolicy
import multiagent.scenarios as scenarios


########################################
action_size = 5

load_model = False
train_mode = True

batch_size = 256
mem_maxlen = 50000
discount_factor = 0.99
learning_rate = 0.00025

run_episode = 10000
start_train_episode = 500
target_update_step = 5000

print_interval = 100
save_interval = 1000

epsilon_min = 0.1
softlambda = 0.9

date_time = str(datetime.date.today()) + '_' + \
            str(datetime.datetime.now().hour) + '_' + \
            str(datetime.datetime.now().minute) + '_' + \
            str(datetime.datetime.now().second)

env_name = "simple_adverary.py"
save_path = "./saved_models/"+date_time+"_maddpg"
load_path = ""

numGoals = 3
###########################################

class Critic(object):
    def __init__(self, state_size, action_size, input, action_input, other_action, model_name="Qmodel", agent_num=3, reuse=False):

        self.state_size = state_size
        self.action_size = action_size
        self.agent_num = agent_num

        # =================================
        self.input = input
        self.action_input = action_input
        self.other_actions = other_action
        # =================================

        with tf.variable_scope(name_or_scope=model_name, reuse=reuse):
            self.mlp1 = layer.dense(inputs=self.input, units=256, activation = tf.nn.leaky_relu)
            self.concat_action = tf.concat([self.action_input, self.other_actions], axis=1)
            self.concat = tf.concat([self.mlp1, self.concat_action], axis=1)
            self.mlp2 = layer.dense(inputs=self.concat, units=256, activation = tf.nn.leaky_relu)
            self.mlp3 = layer.dense(inputs=self.mlp2, units=512, activation = tf.nn.leaky_relu)
            self.mlp4 = layer.dense(inputs=self.mlp3, units=512, activation = tf.nn.leaky_relu)
            self.Q_Out = layer.dense(self.mlp4, units=1, activation=None)
        self.q_predict = self.Q_Out
        self.critic_optimizer = tf.train.AdamOptimizer(learning_rate)

class Actor(object):
    def __init__(self, state_size, action_size, input, model_name="Pimodel"):

        self.agent_num = 3
        self.state_size = state_size
        self.action_size = action_size

        # =================================
        self.input = input
        # =================================

        with tf.variable_scope(name_or_scope=model_name):
            self.mlp1 = layer.dense(inputs=self.input, units=512, activation = tf.nn.leaky_relu)
            self.mlp2 = layer.dense(inputs=self.mlp1, units=512, activation = tf.nn.leaky_relu)
            self.mlp3 = layer.dense(inputs=self.mlp2, units=512, activation = tf.nn.leaky_relu)
            self.mlp4 = layer.dense(inputs=self.mlp3, units=512, activation = tf.nn.leaky_relu)
            self.Pi_Out = layer.dense(self.mlp4,  units=self.action_size, activation=tf.nn.tanh)
        self.pi_predict = self.Pi_Out
        self.actor_optimizer = tf.train.AdamOptimizer(learning_rate)

class MADDPGAgent(object):
    def __init__(self, agent_num, state_size, action_size, idx):
        # (1) "actor" : agent in reinforcement learning
        # (2) "critic" : helps the actor decide what actions to reinforce during training.
        # Traditionally, the critic tries to predict the value (i.e. the reward we expect to get in the future) of an action in a particular state s(t)
        # predicted value from critic is used to update the actor policy
        # Using critic value as an baseline for update s is more stable than directly using the reward, which can vary considerably
        # variation of reward makes the update pertuative
        # In maddpg, we enhance our critics so they can access the observations and actions of all the agents,

        # Default Environment Information =====
        self.state_size = state_size
        self.action_size = action_size
        self.agent_num = agent_num
        # =====================================

        # Experience Buffer ===================
        self.memory = deque(maxlen=mem_maxlen)
        self.batch_size = batch_size
        # =====================================

        # Placeholer =============================================================================
        self.input = tf.placeholder(shape=[None, self.state_size], dtype=tf.float32)
        self.action_input = tf.placeholder(shape=[None, self.action_size], dtype=tf.float32)
        self.other_actions = tf.placeholder(shape=[None, self.action_size * (self.agent_num-1)], dtype=tf.float32)
        self.target_Q = tf.placeholder(shape=[None,1],dtype=tf.float32)
        self.reward = tf.placeholder(shape=[None,1], dtype=tf.float32)
        # ========================================================================================

        self.actor = Actor(self.state_size, self.action_size, self.input, "Pimodel_" + idx)
        self.critic = Critic(self.state_size, self.action_size, self.input, self.action_input, self.other_actions, "Qmodel_" + idx, self.agent_num, reuse=False)


        '''
        critic_value = Critic(self.state_size, self.action_size, self.input, self.actor.pi_predict, self.other_actions, "Qmodel_" + idx, self.agent_num, reuse=True).q_predict
        self.action_gradients = tf.gradients(critic_predict.q_predict, self.actor.pi_predict)[0]
        self.actor_gradients = tf.gradients(self.actor.pi_predict, actor_var, -self.action_gradients)
        self.grads_and_vars = list(zip(self.actor_gradients, actor_var))
        self.actor_train = self.actor.actor_optimizer.apply_gradients(self.grads_and_vars)
        '''

        actor_var = [i for i in tf.trainable_variables() if ("Pimodel_" + idx) in i.name]
        action_Grad = tf.gradients(self.critic.q_predict, self.action_input)
        self.policy_Grads = tf.gradients(ys=self.actor.pi_predict, xs=actor_var, grad_ys=action_Grad)
        for idx, grads in enumerate(self.policy_Grads):
            self.policy_Grads[idx] = -grads / batch_size
        self.actor_train = self.actor.actor_optimizer.apply_gradients(zip(self.policy_Grads, actor_var))

        self.critic_loss = tf.reduce_mean(tf.square(self.target_Q - self.critic.q_predict))
        self.critic_train = self.critic.critic_optimizer.minimize(self.critic_loss)

    def train_actor(self, state, action, other_action, sess):
        sess.run(self.actor_train,
                 {self.input: state, self.action_input : action, self.other_actions: other_action})

    def train_critic(self, state, action, other_action, target, sess):
        sess.run(self.critic_train,
                 {self.input: state, self.action_input: action, self.other_actions: other_action, self.target_Q: target})

    def action(self, state, sess):
        return sess.run(self.actor.pi_predict, {self.input: state})

    def Q(self, state, action, other_action, sess):
        return sess.run(self.critic.q_predict,
                        {self.input: state, self.action_input: action, self.other_actions: other_action})
