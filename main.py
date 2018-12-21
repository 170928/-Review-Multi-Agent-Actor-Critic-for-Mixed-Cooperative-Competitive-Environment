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

load_path = './three_weight/'
###########################################


def get_agents_action(obs_n, obs_shape_n, sess):
    agent1_action = agent1_ddpg.action(state=np.reshape(obs_n[0], newshape=[-1,obs_shape_n[0]]), sess=sess)
    agent2_action = agent2_ddpg.action(state=np.reshape(obs_n[1], newshape=[-1,obs_shape_n[1]]), sess=sess)
    agent3_action = agent3_ddpg.action(state=np.reshape(obs_n[2], newshape=[-1,obs_shape_n[2]]), sess=sess)

    return agent1_action, agent2_action, agent3_action


def train_agent(agent, agent_target, agent_memory, agent_actor_target_update, agent_critic_target_update, sess, other_actors):
    total_obs_batch, total_act_batch, rew_batch, total_next_obs_batch, done_mask = agent_memory.sample(batch_size)

    act_batch = total_act_batch[:, 0, :]
    other_act_batch = np.hstack([total_act_batch[:, 1, :], total_act_batch[:, 2, :]])

    obs_batch = total_obs_batch[:, 0, :]

    next_obs_batch = total_next_obs_batch[:, 0, :]
    next_other_actor1_o = total_next_obs_batch[:, 1, :]
    next_other_actor2_o = total_next_obs_batch[:, 2, :]

    next_other_action = np.hstack([other_actors[0].action(next_other_actor1_o, sess), other_actors[1].action(next_other_actor2_o, sess)])
    target = rew_batch.reshape(-1, 1) + 0.99 * agent_target.Q(state=next_obs_batch, action=agent.action(next_obs_batch, sess), other_action=next_other_action, sess=sess)
    agent.train_actor(state=obs_batch, action=act_batch, other_action=other_act_batch, sess=sess)
    agent.train_critic(state=obs_batch, action=act_batch, other_action=other_act_batch, target=target, sess=sess)

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
    # scenario = scenarios.load('simple_adversary.py').Scenario()
    scenario = scenarios.load('simple_spread.py').Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, info_callback=None,
                        shared_viewer=True)
    obs_n = env.reset()
    print("# of agent {}".format(env.n))
    obs_shape_n = [18,18,18]
    print(env.action_space)
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

    if load_model == True:
        ckpt = tf.train.get_checkpoint_state(load_path)
        Saver.restore(sess, ckpt.model_checkpoint_path)
        print("Restore Model")
    else:
        init = tf.global_variables_initializer()
        sess.run(init)
        print("Initialize Model")

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

    e = 1
    train_mode = True
    for roll_out in range(1000000):
        obs_n = env.reset()
        for episode in range(500):
            env.render()
            agent1_action, agent2_action, agent3_action = get_agents_action(obs_n, obs_shape_n, sess)
            # Discrete action space ================
            acs_agent1 = np.zeros((action_size,))
            acs_agent2 = np.zeros((action_size,))
            acs_agent3 = np.zeros((action_size,))
            acs = []
            if train_mode == True and e > np.random.rand():
                for agent_index in range(env.n):
                    acs.append(np.random.randint(0,action_size))
            else:
                acs.append(np.argmax(agent1_action))
                acs.append(np.argmax(agent2_action))
                acs.append(np.argmax(agent3_action))
                print(agent1_action)
                print(np.argmax(agent1_action))

            acs_agent1[acs[0]] = 1
            acs_agent2[acs[1]] = 1
            acs_agent3[acs[2]] = 1
            acs_n = [acs_agent1, acs_agent2, acs_agent3]
            #print(acs_n)
            # ======================================

            next_obs_n, reward_n, done_n, _ = env.step(acs_n)

            o_n_next, r_n, d_n, i_n = env.step(acs_n)

            agent1_memory.add(np.vstack([obs_n[0], obs_n[1], obs_n[2]]),
                              np.vstack([acs_agent1, acs_agent2, acs_agent3]),
                              r_n[0], np.vstack([o_n_next[0], o_n_next[1], o_n_next[2]]), d_n[0])

            agent2_memory.add(np.vstack([obs_n[1], obs_n[2], obs_n[0]]),
                              np.vstack([acs_agent2,acs_agent3, acs_agent1]),
                              r_n[1], np.vstack([o_n_next[1], o_n_next[2], o_n_next[0]]), d_n[1])

            agent3_memory.add(np.vstack([obs_n[2], obs_n[0], obs_n[1]]),
                              np.vstack([acs_agent3, acs_agent1, acs_agent2]),
                              r_n[2], np.vstack([o_n_next[2], o_n_next[0], o_n_next[1]]), d_n[2])

            o_n = o_n_next

            if roll_out > 1:
                e *= 0.9999
                # agent1 train
                train_agent(agent1_ddpg, agent1_ddpg_target, agent1_memory, agent1_actor_target_update,
                            agent1_critic_target_update, sess, [agent2_ddpg_target, agent3_ddpg_target])

                train_agent(agent2_ddpg, agent2_ddpg_target, agent2_memory, agent2_actor_target_update,
                            agent2_critic_target_update, sess, [agent3_ddpg_target, agent1_ddpg_target])

                train_agent(agent3_ddpg, agent3_ddpg_target, agent3_memory, agent3_actor_target_update,
                            agent3_critic_target_update, sess, [agent1_ddpg_target, agent2_ddpg_target])

            # Agent Reward Tensorboard ===================================================================
            for agent_index in range(3):
                summary_writer.add_summary(
                    sess.run(reward_op[agent_index], {reward_history[agent_index]: r_n[agent_index]}), roll_out)
            # ============================================================================================

        if roll_out % 1000 == 0:
            Saver.save(sess, './three_weight/' + str(roll_out) + '.cptk')





