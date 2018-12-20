
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