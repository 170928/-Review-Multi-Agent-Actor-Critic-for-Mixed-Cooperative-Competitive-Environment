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
    def __init__(self, state_size, action_size,  model_name="Qmodel", agent_num = 3):

        # state_size is [1 x state_dim * agent_num]
        self.state_size = state_size
        # action_size is [1 x action_dim]
        self.action_size = action_size

        self.input = tf.placeholder(shape=[None, self.state_size], dtype=tf.float32)
        self.action_input = [tf.placeholder(shape=[None, self.action_size], dtype=tf.float32) for _ in range(agent_num)]

        with tf.variable_scope(name_or_scope=model_name):
            self.mlp1 = layer.dense(inputs=self.input, units=64, activation = tf.nn.relu)
            self.concat_action = tf.concat(self.action_input, axis = 1)
            self.concat = tf.concat([self.mlp1, self.concat_action], axis = 1)
            self.mlp2 = layer.dense(inputs=self.concat, units=64, activation = tf.nn.relu)
            self.mlp3 = layer.dense(inputs=self.mlp2, units=64, activation = tf.nn.relu)
            self.mlp4 = layer.dense(inputs=self.mlp3, units=64, activation = tf.nn.relu)
            self.Q_Out = layer.dense(self.mlp4, units=1, activation=None)

        self.q_predict = self.Q_Out

        self.target_Q = tf.placeholder(shape=[None,1],dtype=tf.float32)
        self.loss = tf.losses.mean_squared_error(self.target_Q, self.q_predict)
        self.UpdateModel = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

        self.action_grads = tf.gradients(self.q_predict, self.action_input)

class Actor(object):
    def __init__(self, state_size, action_size, model_name="Pimodel", agent_num = 3):

        self.agent_num = agent_num
        # state_size = state_dim
        self.state_size = state_size
        # action_size is action_dim
        self.action_size = action_size

        self.input = tf.placeholder(shape=[None, self.state_size],dtype=tf.float32)

        with tf.variable_scope(name_or_scope=model_name):
            self.mlp1 = layer.dense(inputs=self.input, units=64, activation = tf.nn.relu)
            self.mlp2 = layer.dense(inputs=self.mlp1, units=64, activation = tf.nn.relu)
            self.mlp3 = layer.dense(inputs=self.mlp2, units=64, activation = tf.nn.relu)
            self.mlp4 = layer.dense(inputs=self.mlp3, units=64, activation = tf.nn.relu)
            self.Pi_Out = layer.dense(self.mlp4,  units=self.action_size, activation=None)


        self.pi_predict = self.Pi_Out



        trainable_variables = tf.trainable_variables()
        trainable_variables_Actor = [var for var in trainable_variables if var.name.startswith('Pi')]
        self.action_gradients = tf.placeholder(tf.float32, [None, self.action_size])
        self.actor_gradients = tf.gradients(self.pi_predict, trainable_variables_Actor, -self.action_gradients)
        self.grads_and_vars = list(zip(self.actor_gradients, trainable_variables_Actor))
        self.updateGradients = tf.train.AdamOptimizer(learning_rate).apply_gradients(self.grads_and_vars)


class MADDPGAgent(object):
    def __init__(self, agent_num, state_size_n, action_size, learning_rate=0.00025, batch_size = 32, run_episode=10000, epsilon=1.0, epsilon_min=0.1, discount_factor=0.99):
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

        self.actors = [Actor(self.state_size_n[i], self.action_size, "Pimodel" + str(i)) for i in range(agent_num)]
        self.critics = [Critic(self.state_size_n[i], self.action_size, "Qmodel" + str(i), self.agent_num) for i in range(agent_num)]
        self.target_actors = [Actor(self.state_size_n[i], self.action_size, "targetPimodel" + str(i)) for i in range(agent_num)]
        self.target_critics = [Critic(self.state_size_n[i], self.action_size, "targetQmodel" + str(i)) for i in range(agent_num)]

        self.memory = deque(maxlen=mem_maxlen)
        self.batch_size = batch_size

        # Session Initialize ####################################
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.3
        self.sess = tf.Session(config=config)
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)
        #########################################################

        # Update Parameters ####################################
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.discount_factor = discount_factor
        self.run_episode = run_episode
        #########################################################


        # Save & Load ###########################################
        self.Saver = tf.train.Saver(max_to_keep=5)
        self.save_path = save_path
        self.load_path = load_path
        self.Summary,self.Merge = self.make_Summary()
        #########################################################

    # 수정 안함 ========================================================================================================================
    def train_model(self, done):

        if done:
            if self.epsilon > self.epsilon_min:
                self.epsilon -= 1/self.run_episode

        self.batch_size = 2

        mini_batch = random.sample(self.memory, self.batch_size)

        states = []
        actions = []
        rewards = []
        next_states = []
        next_actions = []
        dones = []

        # ======================================= Buffer에 데이터 저장 & 읽기 구조에 대해서 고민해봐야 할듯 ===================================
        for i in range(self.batch_size):
            states.append(mini_batch[i][0])
            actions.append(mini_batch[i][1])
            rewards.append(mini_batch[i][2])
            next_states.append(mini_batch[i][3])
            dones.append(mini_batch[i][4])
        # ============================================================================================================================

        # [batch_size x agent_num]
        states = np.array(states)
        next_states = np.array(next_states)
        actions = np.array(actions)
        rewards = np.array(rewards)

        start = 0
        end = 0
        for i in range(self.agent_num):
            end += self.state_size_n[i]
            next_actions.append(self.sess.run(self.actors[i].pi_predict, feed_dict={self.actors[i].input: states[:,start:end]}))
            start += self.state_size_n[i]

        # actions reshape =====================
        next_actions = np.hstack(next_actions)
        # =====================================
        targets = []
        target_vals = []
        start = 0
        end = 0
        for i in range(self.agent_num):
            end+=self.state_size_n[i]
            targets.append(self.sess.run(self.critics[i].q_predict,feed_dict={self.critics[i].input: states[:,start:end], self.critics[i].action_input: actions}))
            target_vals.append(self.sess.run(self.target_critics[i].q_predict,feed_dict={self.target_critics[i].input: next_states[:,start:end], self.target_critics[i].action_input: next_actions}))
            start+=self.state_size_n[i]

        # [agent_num x batch_size]
        targets = np.array(targets)
        target_vals = np.array(target_vals)

        # calculate y^
        for i in range(self.batch_size):
            for j in range(self.agent_num):
                if dones[i][j]:
                    targets[j][i] = rewards[i][j]
                else:
                    targets[j][i] = rewards[i][j] + self.discount_factor*target_vals[j][i]

        # update Critic
        start = 0
        end = 0
        loss_n = []
        for i in range(self.agent_num):
            end+=self.state_size_n[i]
            _, loss = self.sess.run([self.critics[i].UpdateModel,self.critics[i].loss],feed_dict={self.critics[i].input: states[:,start:end], self.critics[i].action_input: actions, self.critics[i].target_Q: targets[i]})
            start += self.state_size_n[i]
            loss_n.append(loss)

        # update Actor
        pi_n = self.get_actions(states)
        print(pi_n)
        grads = self.action_gradients(states, pi_n)
        self.updateActor(states, grads)

        self.update_target()
        return loss_n

    def update_target(self):
        trainable_variables = tf.trainable_variables()
        trainable_variables_Critic = [var for var in trainable_variables if var.name.startswith('Q')]
        trainable_variables_Actor = [var for var in trainable_variables if var.name.startswith('Pi')]
        trainable_variables_targetCritic = [var for var in trainable_variables if var.name.startswith('targetQ')]
        trainable_variables_targetActor= [var for var in trainable_variables if var.name.startswith('targetPi')]

        for i in range(len(trainable_variables_Critic)):
            self.sess.run(tf.assign(trainable_variables_targetCritic[i], (1-softlambda)*trainable_variables_targetCritic[i] + softlambda * trainable_variables_Critic[i]))
        for i in range(len(trainable_variables_Actor)):
            self.sess.run(tf.assign(trainable_variables_targetActor[i], (1-softlambda)*trainable_variables_targetActor[i] + softlambda * trainable_variables_Actor[i]))

    def updateActor(self, states, grads):
        start = 0
        end = 0
        for i in range(self.agent_num):
            end+=self.state_size_n[i]
            self.sess.run(self.actors[i].updateGradients, feed_dict={self.actors[i].input : states[:,start:end], self.actors[i].action_gradients : grads[i]})
            start += self.state_size_n[i]

    def action_gradients(self, states, actions):
        start = 0
        end = 0
        grads = []
        for i in range(self.agent_num):
            end+=self.state_size_n[i]
            grads.append(self.sess.run(self.critics[i].action_grads, feed_dict={self.critics[i].input: states[:,start:end], self.critics[i].action_input: actions}))
            start += self.state_size_n[i]
        return grads

    def get_actions(self,state, train_mode=True):
        start = 0
        end = 0
        predict = []
        for i in range(self.agent_num):
            end+=self.state_size_n[i]
            predict.append(self.sess.run(self.actors[i].pi_predict, feed_dict={self.actors[i].input:state}))
            start += self.state_size_n[i]
        return predict

    # 모든 정보는 [ batch_size x # of agents ] 형태로 저장 된다
    def append_sample(self, state_n, action_n, reward_n, next_state_n, done_n):
        self.memory.append((state_n, action_n, reward_n, next_state_n, done_n))

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
    obs_shape_n = [np.array(env.observation_space[i].shape)[0] for i in range(env.n)]
    print("observation dim : {}".format(obs_shape_n))
    print("action dim : {}".format(action_size))

    maddpg = MADDPGAgent(env.n, obs_shape_n, action_size)

    # Temp ==============================================================
    acs_n = [[1,1,1,1,1], [2,1,1,1,1], [3,1,1,1,1]]
    next_obs_n, reward_n, done_n, _ = env.step(acs_n)

    # good_agent # = 2,  adversary_agent # = 1
    maddpg.append_sample(np.hstack(obs_n), np.hstack(acs_n), np.hstack(reward_n), np.hstack(next_obs_n), done_n)
    maddpg.append_sample(np.hstack(obs_n), np.hstack(acs_n), np.hstack(reward_n), np.hstack(next_obs_n), done_n)

    maddpg.train_model(False)