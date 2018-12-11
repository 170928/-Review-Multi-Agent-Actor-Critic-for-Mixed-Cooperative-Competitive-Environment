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

env_name = "simple_adverary.py"
save_path = "./saved_models/"+date_time+"_maddpg"
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

        self.target_Q = tf.placeholder(shape=[None],dtype=tf.float32)
        self.loss = tf.losses.mean_squared_error(self.target_Q, self.q_predict)
        self.UpdateModel = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

class Actor(object):
    def __init__(self, state_size, action_size, model_name="Pimodel"):

        # state_size = state_dim
        self.state_size = state_size
        # action_size is action_dim
        self.action_size = action_size

        self.input = tf.placeholder(shape=[None, self.state_size],dtype=tf.float32)

        with tf.variable_scope(name_or_scope='PiNet'+model_name):
            self.mlp1 = layer.dense(inputs=self.input,activation = tf.nn.relu)
            self.mlp2 = layer.dense(inputs=self.mlp1, activation = tf.nn.relu)
            self.mlp3 = layer.dense(inputs=self.mlp2, activation = tf.nn.relu)
            self.mlp4 = layer.dense(inputs=self.mlp3, activation = tf.nn.relu)
            self.Pi_Out = layer.dense(self.mlp4, self.action_size, activation=None)

        self.pi_predict = self.Pi_Out

class MADDPGAgent(object):
    def __init__(self, agent_num, state_size_n, action_size, learning_rate=0.00025, batch_size = 32):
        # (1) "actor" : agent in reinforcement learning
        # (2) "critic" : helps the actor decide what actions to reinforce during training.
        # Traditionally, the critic tries to predict the value (i.e. the reward we expect to get in the future) of an action in a particular state s(t)
        # predicted value from critic is used to update the actor policy
        # Using critic value as an baseline for update s is more stable than directly using the reward, which can vary considerably
        # variation of reward makes the update pertuative
        # In maddpg, we enhance our critics so they can access the observations and actions of all the agents,

        # various state size can exist.
        self.state_size_n = state_size_n
        self.action_size = action_size
        self.agent_num = agent_num

        self.actors = [Actor(self.state_size_n[i], self.action_size, str(i)+"Pimodel") for i in range(agent_num)]
        self.critics = [Critic(self.state_size_n[i], self.action_size, str(i)+"Qmodel") for i in range(agent_num)]
        self.target_actors = [Actor(self.state_size_n[i], self.action_size, str(i)+"targetPimodel") for i in range(agent_num)]
        self.target_critics = [Critic(self.state_size_n[i], self.action_size, str(i)+"targetQmodel") for i in range(agent_num)]

        self.memory = deque(maxlen=mem_maxlen)
        self.batch_size = batch_size




        # Save & Load ###########################################
        self.Saver = tf.train.Saver(max_to_keep=5)
        self.save_path = save_path
        self.load_path = load_path
        self.Summary,self.Merge = self.make_Summary()
        #########################################################

        # Session Initialize ####################################
        self.sess = tf.Session()
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)
        #########################################################


    def append_sample(self,state, action_n, reward, next_state, done):
        self.memory.append((state[0], action_n, reward, next_state[0], done))

    def save_model(self):
        self.Saver.save(self.sess,self.save_path + "\model.ckpt")

    def make_Summary(self):
        self.summary_loss = tf.placeholder(dtype=tf.float32)
        self.summary_reward = tf.placeholder(dtype=tf.float32)
        tf.summary.scalar("loss", self.summary_loss)
        tf.summary.scalar("reward",self.summary_reward)
        return tf.summary.FileWriter(logdir=save_path,graph=self.sess.graph),tf.summary.merge_all()

    def Write_Summray(self,reward,loss,episode):
        self.Summary.add_summary(self.sess.run(self.Merge,feed_dict={self.summary_loss:loss,self.summary_reward:reward}),episode)




if __name__=="__main__":
    # Particle-environment
    # https://github.com/openai/multiagent-particle-envs
    # MADDPG
    # https://github.com/xuehy/pytorch-maddpg/blob/master/MADDPG.py

    print(tf.__version__)

    # load scenario from script
    scenario = scenarios.load('simple_adversary.py').Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, info_callback=None,
                        shared_viewer=True)
    env.render()
    obs_n = env.reset()
    print("# of agent {}".format(env.n))
    obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
    print("observation dim : {}".format(obs_shape_n))
    print("action dim : {}".format(action_size))


    #while True:
        # query for action from each agent's policy
    #act_n = []
    #for i, policy in enumerate(policies):
    #    act_n.append(policy.action(obs_n[i]))
        # step environment
    #obs_n, reward_n, done_n, _ = env.step(act_n)
        # render all agent views
    #env.render()
        # display rewards
        # for agent in env.world.agents:
        #    print(agent.name + " reward: %0.3f" % env._get_reward(agent))



