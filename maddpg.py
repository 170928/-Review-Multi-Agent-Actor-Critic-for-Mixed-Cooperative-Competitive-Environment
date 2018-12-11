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
load_model = False
train_mode = True

batch_size = 32
mem_maxlen = 50000
discount_factor = 0.99
learning_rate = 0.00025

run_episode = 10000
start_train_episode = 500
target_update_step = 5000

print_interval = 100
save_interval = 1000

epsilon_min = 0.1

date_time = str(datetime.date.today()) + '_' + \
            str(datetime.datetime.now().hour) + '_' + \
            str(datetime.datetime.now().minute) + '_' + \
            str(datetime.datetime.now().second)

env_name = "../envs/Socoban"
save_path = "saved_models/"+date_time+"_dqn"
load_path = ""

numGoals = 3
###########################################

class Critic(object):
    def __init__(self, state_size, action_size,  model_name="Qmodel"):

        # state_size is [1 x state_dim * agent_num]
        self.state_size = state_size
        # action_size is [1 x action_dim * agent_num]
        self.action_size = action_size

        self.input = tf.placeholder(shape=[None, self.state_size],dtype=tf.float32)

        with tf.variable_scope(name_or_scope='QNet'+model_name):
            self.mlp1 = layer.dense(inputs=self.input, activation = tf.nn.relu)
            self.mlp2 = layer.dense(inputs=self.mlp1, activation = tf.nn.relu)
            self.mlp3 = layer.dense(inputs=self.mlp2, activation = tf.nn.relu)
            self.mlp4 = layer.dense(inputs=self.mlp3, activation = tf.nn.relu)
            self.Q_Out = layer.dense(self.mlp4, 1, activation=None)

        self.q_predict = self.Q_Out

class Actor(object):
    def __init__(self, state_size, action_size, model_name="Pimodel"):

        # state_size = state_dim
        self.state_size = state_size
        # action_size is action_dim
        self.action_size = action_size

        self.input = tf.placeholder(shape=[None, self.state_size],dtype=tf.float32)

        with tf.variable_scope(name_or_scope='PiNet'+model_name):
            self.mlp1 = layer.dense(inputs=self.input, activation = tf.nn.relu)
            self.mlp2 = layer.dense(inputs=self.mlp1, activation = tf.nn.relu)
            self.mlp3 = layer.dense(inputs=self.mlp2, activation = tf.nn.relu)
            self.mlp4 = layer.dense(inputs=self.mlp3, activation = tf.nn.relu)
            self.Pi_Out = layer.dense(self.mlp4, self.action_size, activation=None)

        self.pi_predict = self.Pi_Out

class MADDPGAgent(object):
    def __init__(self, agent_num, state_dim, action_dim, learning_rate=0.00025):
        # (1) "actor" : agent in reinforcement learning
        # (2) "critic" : helps the actor decide what actions to reinforce during training.
        # Traditionally, the critic tries to predict the value (i.e. the reward we expect to get in the future) of an action in a particular state s(t)
        # predicted value from critic is used to update the actor policy
        # Using critic value as an baseline for update s is more stable than directly using the reward, which can vary considerably
        # variation of reward makes the update pertuative
        # In maddpg, we enhance our critics so they can access the observations and actions of all the agents,

        self.state_size = state_dim
        self.action_size = action_dim
        self.agent_num = agent_num

        








