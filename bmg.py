import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import os
import gym
from torch.distributions import Categorical # takes prob output from NN, maps to distribution, so we can do sampling


class ActorNetwork(nn.Module):

    def __init__(self, input_dims, n_actions, alpha, fc1_dims=256, fc2_dims=256):
        super(ActorNetwork, self).__init__()
        self.chkpt_file = os.path.join("todo"+'_td3')

        self.fc1 = nn.Linear(*input_dims,fc1_dims)
        self.fc2 = nn.Linear(fc1_dims,fc2_dims)
        self.fc3 = nn.Linear(fc2_dims,n_actions)

        self.optim = T.optim.Adam(self.parameters(),lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self,x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

    def choose_action(self, observation):
        state = T.tensor([observation], dtype=T.float)
        pi = self.forward(state)
        probs = T.softmax(pi, dim=1)
        dist = Categorical(probs)
        action = dist.sample().numpy()[0]

        return action

    def save_checkpoint(self):
        print("...Saving Checkpoint...")
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        print("...Loading Checkpoint...")
        self.load_state_dict(T.load(self.chkpt_file))

class CriticNetwork(nn.Module):

    def __init__(self, input_dims, n_actions, alpha, fc1_dims=256, fc2_dims=256):
        super(CriticNetwork, self).__init__()
        self.chkpt_file = os.path.join("todo"+'_td3')

        self.fc1 = nn.Linear(*input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, 1)

        self.optim = T.optim.Adam(self.parameters(),lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self,x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

    def save_checkpoint(self):
        print("...Saving Checkpoint...")
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        print("...Loading Checkpoint...")
        self.load_state_dict(T.load(self.chkpt_file))

class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []

    def remember(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.rewards = []

    def sample_memory(self):
        return self.states, self.actions, self.rewards

class Agent:
    def __init__(self, input_dims, n_actions, 
                gamma, alpha, beta, name, env_id, n_steps, n_meta_steps, n_rollout_steps, n_bootstrap_steps, random_seed):
        super(Agent, self).__init__()

        self.actor = ActorNetwork(input_dims, n_actions, alpha, fc1_dims = 256, fc2_dims = 256)
        self.critic = CriticNetwork(input_dims, n_actions, alpha, fc1_dims = 256, fc2_dims = 256)
        self.bs_actor = ActorNetwork(input_dims, n_actions, alpha, fc1_dims = 256, fc2_dims = 256)
        self.bs_critic = CriticNetwork(input_dims, n_actions, alpha, fc1_dims = 256, fc2_dims = 256)
        self.memory = Memory()
        self.gamma = T.tensor(gamma, dtype=T.float, requires_grad=True).to(self.actor.device)
        self.env = gym.make(env_id)
        self.beta = beta
        self.name = f"agent_{name}"
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.n_steps = n_steps
        self.n_meta_steps = n_meta_steps
        self.roll_out_steps = n_rollout_steps
        self.n_bootstrap_steps = n_bootstrap_steps
        self.random_seed = random_seed
        self.env.seed(random_seed)
        self.init_state = self.env.reset()

    def calc_reward(self, rewards, v, final_r):
        R = final_r
        batch_return = []
        for reward in rewards[::-1]:
            R = self.gamma.detach().numpy()*R + reward
            batch_return.append(R)
        batch_return.reverse()
        batch_return = T.tensor(batch_return, dtype=T.float).reshape(v.size()).to(self.actor.device)

        return batch_return

    def calc_reward_grad(self, rewards, final_r):
        R = T.tensor(final_r, dtype=T.float).to(self.actor.device)
        R = T.reshape(R, (1,1))
        batch_return = T.zeros(len(rewards)).to(self.actor.device)
        for i in range(len(rewards)-1, -1, -1):
            R = self.gamma*R + rewards[i]
            batch_return[i] = R
        return batch_return

    def calc_dj_dtheta(self, pi_, values_, return_, actor_network, critic_network):

        theta1 = T.autograd.grad(pi_, actor_network.parameters())
        theta1 = [item.view(-1) for item in theta1]
        theta1 = T.cat(theta1)

        theta2 = T.autograd.grad(values_, critic_network.parameters())
        theta2 = [item.view(-1) for item in theta2]
        theta2 = T.cat(theta2)

        g_dgamma = T.autograd.grad(return_, self.gamma)
        return g_dgamma[0], theta1, theta2

    def calc_djp_dthetap(self, pi_, values_, actor_network, critic_network):

        theta1 = T.autograd.grad(pi_, actor_network.parameters())
        theta1 = [item.view(-1) for item in theta1]
        theta1 = T.cat(theta1)
        
        theta2 = T.autograd.grad(values_, critic_network.parameters())
        theta2 = [item.view(-1) for item in theta2]
        theta2 = T.cat(theta2)
        
        return theta1, theta2

    def roll_out(self):
        obs = self.init_state
        self.memory.clear_memory()
        done, is_done, final_reward = False, False, 0
        for _ in range(self.roll_out_steps):
            action = self.actor.choose_action(obs)
            obs_, reward, done, info = self.env.step(action)
            one_hot_action = [int(k == action) for k in range(self.n_actions)]
            self.memory.remember(obs, one_hot_action, reward)
            f_state = obs_
            obs = obs_
            if done:
                is_done = True
                obs = self.env.reset()
                break

        if not is_done:
            f_state = T.tensor(f_state, dtype=T.float)
            final_reward = self.critic(f_state).cpu().data.numpy()
        
        return final_reward, obs, done

    def run(self, steps):
        
        for step in range(steps):
            final_reward, obs, done = self.roll_out()
            self.init_state = obs
            states, actions, rewards = self.memory.sample_memory()
            actions_var = T.tensor(actions, dtype=T.float).view(-1, self.n_actions).to(self.actor.device)
            states_var = T.tensor(states, dtype=T.float).view(-1, *self.input_dims).to(self.actor.device)

            self.actor.optim.zero_grad()
            self.critic.optim.zero_grad()

            # train actor
            pi = self.actor(states_var)
            log_softmax_actions = F.log_softmax(pi)
            v = self.critic(states_var).detach().squeeze()
            q = self.calc_reward(rewards, v, final_reward)
            advantage = q - v
            actor_network_loss = - T.mean(T.sum(log_softmax_actions*actions_var,dim=1)* advantage)
            actor_network_loss.backward(retain_graph=True)
            #T.nn.utils.clip_grad_norm_(self.actor.parameters(), 1)
            
            # train critic
            target_v = q
            v = self.critic(states_var).squeeze()
            value_network_loss = F.mse_loss(v, target_v)
            value_network_loss.backward(retain_graph=True)
            #T.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)

            pi_ = T.mean(log_softmax_actions*actions_var)
            v_ = (v).pow(2).mean()
            return_ = self.calc_reward_grad(rewards, final_reward).mean()
            dg, f1, f2 = self.calc_dj_dtheta(pi_, v_, return_, self.actor, self.critic)

            self.actor.optim.step()
            self.critic.optim.step()

            # Meta-Gradient 
            final_reward, obs, done = self.roll_out()
            states, actions, rewards = self.memory.sample_memory()
            actions_var = T.tensor(actions, dtype=T.float).view(-1, self.n_actions).to(self.actor.device)
            states_var = T.tensor(states, dtype=T.float).view(-1, *self.input_dims).to(self.actor.device)

            self.actor.optim.zero_grad()
            self.critic.optim.zero_grad()

            pi = self.actor(states_var)
            v = self.critic(states_var).squeeze()
            log_softmax_actions = F.log_softmax(pi)
            pi_ = T.mean(log_softmax_actions*actions_var)
            v_ = (v).pow(2).mean()
            J1, J2 = self.calc_djp_dthetap(pi_, v_, self.actor, self.critic)
            
            #update meta-param (using only gamma)
            with T.no_grad():
                self.gamma -= self.beta*dg*(T.matmul(f1,J1))
        
            self.init_state = self.env.reset()
    
    def bootstrap_run(self):

        self.bs_actor = self.actor
        self.bs_critic = self.critic

        for step in range(self.n_steps):
            self.run(self.n_meta_steps)
            self.run(self.n_bootstrap_steps)

            final_reward, obs, done = self.roll_out()
            self.init_state = obs
            states, actions, rewards = self.memory.sample_memory()
            actions_var = T.tensor(actions, dtype=T.float).view(-1, self.n_actions).to(self.actor.device)
            states_var = T.tensor(states, dtype=T.float).view(-1, *self.input_dims).to(self.actor.device)

            self.actor.optim.zero_grad()
            self.critic.optim.zero_grad()

            # train actor
            pi = self.actor(states_var)
            log_softmax_actions = F.log_softmax(pi)
            v = self.critic(states_var).detach().squeeze()
            q = self.calc_reward(rewards, v, final_reward)
            advantage = q - v
            actor_network_loss = - T.mean(T.sum(log_softmax_actions*actions_var,dim=1)* advantage)
            actor_network_loss.backward(retain_graph=True)
            #T.nn.utils.clip_grad_norm_(self.actor.parameters(), 1)
            
            # train critic
            target_v = q
            v = self.critic(states_var).squeeze()
            value_network_loss = F.mse_loss(v, target_v)
            value_network_loss.backward(retain_graph=True)
            #T.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)

            pi_ = T.mean(log_softmax_actions*actions_var)
            v_ = (v).pow(2).mean()
            return_ = self.calc_reward_grad(rewards, final_reward).mean()
            dg, f1, f2 = self.calc_dj_dtheta(pi_, v_, return_, self.actor, self.critic)

            self.actor.optim.step()
            self.critic.optim.step()

            # Meta-Gradient 
            final_reward, obs, done = self.roll_out()
            states, actions, rewards = self.memory.sample_memory()
            actions_var = T.tensor(actions, dtype=T.float).view(-1, self.n_actions).to(self.actor.device)
            states_var = T.tensor(states, dtype=T.float).view(-1, *self.input_dims).to(self.actor.device)

            self.actor.optim.zero_grad()
            self.critic.optim.zero_grad()

            pi = self.actor(states_var)
            v = self.critic(states_var).squeeze()
            log_softmax_actions = F.log_softmax(pi)
            pi_ = T.mean(log_softmax_actions*actions_var)
            v_ = (v).pow(2).mean()
            J1, J2 = self.calc_djp_dthetap(pi_, v_, self.actor, self.critic)
            
            #update meta-param (using only gamma)
            with T.no_grad():
                self.gamma -= self.beta*dg*(T.matmul(f1,J1))

            # test
            if (step + 1) % 10== 0:
                test_env = gym.make(env_id)
                test_env.seed(self.random_seed)
                score = 0
                for _ in range(100):
                    obs = test_env.reset()
                    for i in range(1000):
                        action = self.actor.choose_action(obs)
                        obs_, reward, done, info = test_env.step(action)
                        score += reward
                        obs = obs_
                        if done:
                            break
                    #scores.append(score)
                avg_score = score/100
                print(f"step: {step+1} avg_100_eps_score: {avg_score} meta_param(gamma): {self.gamma}")
                if avg_score > test_env.spec.reward_threshold:
                    break
        
            self.init_state = self.env.reset()

    def save_models(self):
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

if __name__ == "__main__":
    '''Driver code'''
    steps = 2000
    meta_length = 20
    bootstrap_length = 20
    roll_out_length = 50
    random_seed = 1
    env_id = "CartPole-v0"
    n_actions = 2
    input_dims = [4]
    gamma = 0.99
    alpha = 0.001
    beta = 0.0001
    name = 'meta_agent_bmg' 

    # set seed
    T.cuda.manual_seed(random_seed)
    T.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    agent = Agent(input_dims, n_actions, gamma, alpha, beta, name, env_id, steps, meta_length, roll_out_length, bootstrap_length, random_seed)
    agent.bootstrap_run()
