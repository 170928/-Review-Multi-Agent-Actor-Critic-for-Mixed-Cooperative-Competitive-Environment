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
            self.mlp1 = layer.dense(inputs=self.input, units=64, activation = tf.nn.relu)
            self.concat_action = tf.concat([self.action_input, self.other_actions], axis=1)
            self.concat = tf.concat([self.mlp1, self.concat_action], axis=1)
            self.mlp2 = layer.dense(inputs=self.concat, units=64, activation = tf.nn.relu)
            self.mlp3 = layer.dense(inputs=self.mlp2, units=64, activation = tf.nn.relu)
            self.mlp4 = layer.dense(inputs=self.mlp3, units=64, activation = tf.nn.relu)
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
            self.mlp1 = layer.dense(inputs=self.input, units=64, activation = tf.nn.relu)
            self.mlp2 = layer.dense(inputs=self.mlp1, units=64, activation = tf.nn.relu)
            self.mlp3 = layer.dense(inputs=self.mlp2, units=64, activation = tf.nn.relu)
            self.mlp4 = layer.dense(inputs=self.mlp3, units=64, activation = tf.nn.relu)
            self.Pi_Out = layer.dense(self.mlp4,  units=self.action_size, activation=None)
        self.pi_predict = self.Pi_Out
        self.actor_optimizer = tf.train.AdamOptimizer(learning_rate)

class MADDPGAgent(object):
    def __init__(self, agent_num, state_size, action_size, idx, learning_rate=0.00025, batch_size = 32, run_episode=10000, epsilon=1.0, epsilon_min=0.1, discount_factor=0.99):
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

        # Session Initialize =====================================
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.3
        self.sess = tf.Session(config=config)
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)
        # ========================================================

        # Update Parameters ======================================
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.discount_factor = discount_factor
        self.run_episode = run_episode
        # ========================================================

        # Save & Load ============================================
        self.Saver = tf.train.Saver(max_to_keep=5)
        self.save_path = save_path
        self.load_path = load_path
        self.Summary,self.Merge = self.make_Summary()
        # ========================================================

        # Placeholer =============================================================================
        self.input = tf.placeholder(shape=[None, self.state_size], dtype=tf.float32)
        self.action_input = tf.placeholder(shape=[None, self.action_size], dtype=tf.float32)
        self.other_actions = tf.placeholder(shape=[None, self.action_size * (self.agent_num-1)])
        self.target_Q = tf.placeholder(shape=[None,1],dtype=tf.float32)
        self.reward = tf.placeholder(shape=[None, 1], dtype=tf.float32)
        # ========================================================================================

        self.actor = Actor(self.state_size, self.action_size, input, "Pimodel" + str(idx))
        self.critic = Critic(self.state_size, self.action_size, self.input, self.action_input, self.other_actions, "Qmodel" + str(idx), self.agent_num, reuse=False)

        self.actor_loss = -tf.reduce_mean(Critic(self.state_size, self.action_size, self.input, self.actor.pi_predict, self.other_actions, "Qmodel" + str(idx), self.agent_num, reuse=True))
        self.actor_train = self.actor.actor_optimizer.minimize(self.actor_loss)

        self.critic_loss = tf.reduce_mean(tf.square(self.target_Q - self.critic.q_predict))
        self.critic_train = self.critic.critic_optimizer.minimize(self.critic_loss)

    def train_actor(self, state, other_action, sess):
        sess.run(self.actor_train, {self.input: state, self.other_actions: other_action})

    def train_critic(self, state, action, other_action, target, sess):
        sess.run(self.critic_train,
                 {self.input: state, self.action_input: action, self.other_actions: other_action, self.target_Q: target})

    def action(self, state, sess):
        return sess.run(self.actor.pi_predict, {self.input: state})

    def Q(self, state, action, other_action, sess):
        return sess.run(self.critic.q_predict,
                        {self.input: state, self.action_input: action, self.other_actions: other_action})

# ====================================================================================================================================================================================================

    def train_model(self, done):

        if done:
            if self.epsilon > self.epsilon_min:
                self.epsilon -= 1/self.run_episode

        self.batch_size = 2

        # batch from individual memories

        mini_batch = []
        for i in range(self.agent_num):
            mini_batch.append(random.sample(self.memory, self.batch_size))

        states = []
        actions = []
        rewards = []
        next_states = []
        next_actions = []
        dones = []

        # =================================================================================================================================
        for j in range(self.agent_num):
            for i in range(self.batch_size):
                states.append(mini_batch[j][i][0])
                actions.append(mini_batch[j][i][1])
                rewards.append(mini_batch[j][i][2])
                next_states.append(mini_batch[j][i][3])
                dones.append(mini_batch[j][i][4])



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
            targets.append(self.sess.run(self.critics[i].q_predict,feed_dict={self.critics[i].input: states[:, start:end], self.critics[i].action_input: actions}))
            target_vals.append(self.sess.run(self.target_critics[i].q_predict,feed_dict={self.target_critics[i].input: next_states[:, start:end],self.target_critics[i].action_input: next_actions}))
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
        grads = self.action_gradients(states, pi_n)
        self.updateActor(states, grads)
        # =================================================================================================================================

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

    def get_actions(self, state, train_mode=True):
        start = 0
        end = 0
        predict = []
        for i in range(self.agent_num):
            end+=self.state_size_n[i]
            predict.append(self.sess.run(self.actors[i].pi_predict, feed_dict={self.actors[i].input:state[:, start:end] }))
            start += self.state_size_n[i]
        return predict

    # 모든 정보는 [ batch_size x # of agents ] 형태로 저장 된다
    def append_sample(self,idx, state_n, action, other_action_n, reward_n, next_state_n, done_n):
        self.memory[idx].append((state_n, action, other_action_n, reward_n, next_state_n, done_n))

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

    # Test용 ==============================================================
    acs_n = [[1,1,1,1,1], [2,2,2,2,2], [3,3,3,3,3]]
    next_obs_n, reward_n, done_n, _ = env.step(acs_n)

    # good_agent # = 2,  adversary_agent # = 1
    maddpg.append_sample(0,np.hstack(obs_n), [1,1,1,1,1], [[2,2,2,2,2], [3,3,3,3,3]], np.hstack(reward_n), np.hstack(next_obs_n), done_n)
    maddpg.append_sample(0,np.hstack(obs_n), [1,1,1,1,1], [[2,2,2,2,2], [3,3,3,3,3]], np.hstack(reward_n), np.hstack(next_obs_n), done_n)
    maddpg.append_sample(1,np.hstack(obs_n), [2,2,2,2,2], [[1,1,1,1,1], [3,3,3,3,3]], np.hstack(reward_n), np.hstack(next_obs_n), done_n)
    maddpg.append_sample(1,np.hstack(obs_n), [2,2,2,2,2], [[1,1,1,1,1], [3,3,3,3,3]], np.hstack(reward_n), np.hstack(next_obs_n), done_n)
    maddpg.append_sample(2,np.hstack(obs_n), [3,3,3,3,3], [[1,1,1,1,1], [2,2,2,2,2]], np.hstack(reward_n), np.hstack(next_obs_n), done_n)
    maddpg.append_sample(2,np.hstack(obs_n), [3,3,3,3,3], [[1,1,1,1,1], [2,2,2,2,2]], np.hstack(reward_n), np.hstack(next_obs_n), done_n)

    maddpg.train_model(False)