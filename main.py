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
from maddpg import MADDPGAgent
from ReplayBuffer import ReplayBuffer

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
###########################################


def get_agents_action(obs_n, obs_shape_n, sess, noise_rate=0.0):
    agent1_action = agent1_ddpg.action(state=np.reshape(obs_n[0], newshape=[-1,obs_shape_n[0]]), sess=sess) + np.random.randn(5) * noise_rate
    agent2_action = agent2_ddpg.action(state=np.reshape(obs_n[1], newshape=[-1,obs_shape_n[1]]), sess=sess) + np.random.randn(5) * noise_rate
    agent3_action = agent3_ddpg.action(state=np.reshape(obs_n[2], newshape=[-1,obs_shape_n[2]]), sess=sess) + np.random.randn(5) * noise_rate

    return agent1_action, agent2_action, agent3_action

def train_agent(agent, agent_target, agent_memory, agent_actor_target_update, agent_critic_target_update, sess):
    obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = agent_memory.sample(batch_size)
    new_action_batch = agent.action(next_obs_batch, sess)
    target = rew_batch.reshape(-1, 1) + 0.9999 * agent_target.Q(state=next_obs_batch, action=new_action_batch, sess=sess)
    agent.train_actor(state=obs_batch, sess=sess)
    agent.train_critic(state=obs_batch, action=act_batch, target=target, sess=sess)

    sess.run([agent_actor_target_update, agent_critic_target_update])

def train_agent(agent_ddpg, agent_ddpg_target, agent_memory, agent_actor_target_update, agent_critic_target_update, sess, other_actors):
    total_obs_batch, total_act_batch, rew_batch, total_next_obs_batch, done_mask = agent_memory.sample(32)

    act_batch = total_act_batch[:, 0, :]
    other_act_batch = np.hstack([total_act_batch[:, 1, :], total_act_batch[:, 2, :]])

    obs_batch = total_obs_batch[:, 0, :]

    next_obs_batch = total_next_obs_batch[:, 0, :]
    next_other_actor1_o = total_next_obs_batch[:, 1, :]
    next_other_actor2_o = total_next_obs_batch[:, 2, :]

    next_other_action = np.hstack([other_actors[0].action(next_other_actor1_o, sess), other_actors[1].action(next_other_actor2_o, sess)])
    target = rew_batch.reshape(-1, 1) + 0.9999 * agent_ddpg_target.Q(state=next_obs_batch, action=agent_ddpg.action(next_obs_batch, sess),
                                                                     other_action=next_other_action, sess=sess)
    agent_ddpg.train_actor(state=obs_batch, other_action=other_act_batch, sess=sess)
    agent_ddpg.train_critic(state=obs_batch, action=act_batch, other_action=other_act_batch, target=target, sess=sess)

    sess.run([agent_actor_target_update, agent_critic_target_update])

def create_init_update(oneline_name, target_name, tau=0.99):
    online_var = [i for i in tf.trainable_variables() if oneline_name in i.name]
    target_var = [i for i in tf.trainable_variables() if target_name in i.name]

    target_init = [tf.assign(target, online) for online, target in zip(online_var, target_var)]
    target_update = [tf.assign(target, (1 - tau) * online + tau * target) for online, target in
                                   zip(online_var, target_var)]

    return target_init, target_update


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
    obs_n = env.reset()
    print("# of agent {}".format(env.n))
    obs_shape_n = [np.array(env.observation_space[i].shape)[0] for i in range(env.n)]
    print("observation dim : {}".format(obs_shape_n))
    print("action dim : {}".format(action_size))

    # Agent Generation =======================================
    agent1_ddpg = MADDPGAgent(env.n, obs_shape_n[0], action_size, '1')
    agent1_ddpg_target = MADDPGAgent(env.n, obs_shape_n[0], action_size, 'target1')

    agent2_ddpg = MADDPGAgent(env.n, obs_shape_n[1], action_size, '2')
    agent2_ddpg_target = MADDPGAgent(env.n, obs_shape_n[1], action_size, 'target2')

    agent3_ddpg = MADDPGAgent(env.n, obs_shape_n[2], action_size, '3')
    agent3_ddpg_target = MADDPGAgent(env.n, obs_shape_n[2], action_size, 'target3')

    # Save & Load ============================================
    Saver = tf.train.Saver(max_to_keep=5)
    save_path = save_path
    load_path = load_path
    # self.Summary,self.Merge = self.make_Summary()
    # ========================================================

    # Agent initialization ===================================
    agent1_actor_target_init, agent1_actor_target_update = create_init_update('Pimodel_1', 'Pimodel_target1')
    agent1_critic_target_init, agent1_critic_target_update = create_init_update('Qmodel_1', 'Qmodel_target1')

    agent2_actor_target_init, agent2_actor_target_update = create_init_update('Pimodel_2', 'Pimodel_target2')
    agent2_critic_target_init, agent2_critic_target_update = create_init_update('Qmodel_2', 'Qmodel_target2')

    agent3_actor_target_init, agent3_actor_target_update = create_init_update('Pimodel_3', 'Pimodel_target3')
    agent3_critic_target_init, agent3_critic_target_update = create_init_update('Qmodel_3', 'Qmodel_target3')
    # ========================================================

    # Session Initialize =====================================
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.4
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    sess.run(init)
    sess.run([agent1_actor_target_init, agent1_critic_target_init,
              agent2_actor_target_init, agent2_critic_target_init,
              agent3_actor_target_init, agent3_critic_target_init])
    # ========================================================

    # Tensorboard ============================================
    reward_history = [tf.Variable(0, dtype=tf.float32) for i in range(env.n)]
    reward_op = [tf.summary.scalar('agent' + str(i) + '_reward', reward_history[i]) for i in range(env.n)]
    summary_writer = tf.summary.FileWriter('./three_summary', graph=tf.get_default_graph())
    # ========================================================

    # Replay Buffer ======================================
    agent1_memory = ReplayBuffer(mem_maxlen)
    agent2_memory = ReplayBuffer(mem_maxlen)
    agent3_memory = ReplayBuffer(mem_maxlen)
    # ========================================================

    # Test용 ==============================================================
    e = 1
    train_mode = True
    for roll_out in range(1000000):
        env.render()
        if roll_out % 1000 == 0:
            obs_n = env.reset()

        agent1_action, agent2_action, agent3_action = get_agents_action(obs_n, obs_shape_n, sess, noise_rate=0.2)

        # Discrete action space ================
        acs_agent1 = np.zeros((action_size,))
        acs_agent2 = np.zeros((action_size,))
        acs_agent3 = np.zeros((action_size,))

        acs = []
        if train_mode == True and e > np.random.rand():
            for agent_index in range(env.n):
                acs.append(np.random.randint(0,action_size))
        print(acs)
        acs_agent1[acs[0]] = 1
        acs_agent2[acs[1]] = 1
        acs_agent3[acs[2]] = 1
        acs_n = [acs_agent1, acs_agent2, acs_agent3]
        # ======================================

        next_obs_n, reward_n, done_n, _ = env.step(acs_n)

        # good_agent # = 2,  adversary_agent # = 1
        o_n_next, r_n, d_n, i_n = env.step(acs_n)

        agent1_memory.add(obs_n[0], agent1_action[0], r_n[0], o_n_next[0], d_n[0])
        agent2_memory.add(obs_n[1], agent2_action[0], r_n[1], o_n_next[1], d_n[1])
        agent3_memory.add(obs_n[2], agent3_action[0], r_n[2], o_n_next[2], d_n[2])

        if roll_out > 50000:
            e *= 0.9999
            # agent1 train
            train_agent(agent1_ddpg, agent1_ddpg_target, agent1_memory, agent1_actor_target_update,
                        agent1_critic_target_update, sess)

            train_agent(agent2_ddpg, agent2_ddpg_target, agent2_memory, agent2_actor_target_update,
                        agent2_critic_target_update, sess)

            train_agent(agent3_ddpg, agent3_ddpg_target, agent3_memory, agent3_actor_target_update,
                        agent3_critic_target_update, sess)

        # Agent Reward Tensorboard ===================================================================
        for agent_index in range(3):
            summary_writer.add_summary(
                sess.run(reward_op[agent_index], {reward_history[agent_index]: r_n[agent_index]}), roll_out)
        # ============================================================================================
        if roll_out % 1000 == 0:
            Saver.save(sess, './three_weight/' + str(roll_out) + '.cptk')

        o_n = o_n_next



