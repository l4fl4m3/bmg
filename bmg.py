import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import os
import TorchOpt
from torch.distributions import Categorical
from torch.distributions.kl import kl_divergence
import matplotlib.pyplot as plt
from tcgw_env import TwoColorGridWorld

class ActorCritic(nn.Module):

    def __init__(self, input_dims, n_actions, alpha, fc1_dims=256, fc2_dims=256, gamma=0.99):
        super(ActorCritic, self).__init__()
        self.chkpt_file = os.path.join("todo"+'_bmg')

        self.pi1 = nn.Linear(*input_dims,fc1_dims)
        self.v1 = nn.Linear(*input_dims,fc1_dims)
        self.pi2 = nn.Linear(fc1_dims,fc2_dims)
        self.v2 = nn.Linear(fc1_dims,fc2_dims)
        self.pi = nn.Linear(fc2_dims,n_actions)
        self.v = nn.Linear(fc2_dims,1)

        self.optim = TorchOpt.MetaSGD(self,lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self,x):

        pi = F.relu(self.pi1(x))
        v = F.relu(self.v1(x))

        pi = F.relu(self.pi2(pi))
        v = F.relu(self.v2(v))

        pi = self.pi(pi)
        v = self.v(v)

        probs = T.softmax(pi, dim=1)
        dist = Categorical(probs)

        return dist, v

    def choose_action(self, observation):
        state = T.tensor([observation], dtype=T.float)
        dist, v = self.forward(state)
        action = dist.sample().numpy()[0]

        return action

    def save_checkpoint(self):
        print("...Saving Checkpoint...")
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        print("...Loading Checkpoint...")
        self.load_state_dict(T.load(self.chkpt_file))

class MetaMLP(nn.Module):
    def __init__(self, alpha, betas=(0.9, 0.999), eps=1e-4, input_dims=10, fc1_dims=32):
        super(MetaMLP, self).__init__()
        self.chkpt_file = os.path.join("todo"+'_bmg')

        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, 1)

        self.optim = T.optim.Adam(self.parameters(),lr=alpha, betas=betas, eps=eps)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = T.sigmoid(self.fc2(out))
        return out
    
    def save_checkpoint(self):
        print("...Saving Checkpoint...")
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        print("...Loading Checkpoint...")
        self.load_state_dict(T.load(self.chkpt_file))

class Agent:
    def __init__(self, input_dims, n_actions, gamma, alpha, m_alpha, betas, eps, name, 
                    env, steps, K_steps, L_steps, rollout_steps, random_seed):
        super(Agent, self).__init__()
        
        self.actorcritic = ActorCritic(input_dims, n_actions, alpha, fc1_dims = 256, fc2_dims = 256)
        self.ac_k = ActorCritic(input_dims, n_actions, alpha, fc1_dims = 256, fc2_dims = 256)
        self.meta_mlp = MetaMLP(m_alpha, betas, eps, input_dims=10, fc1_dims=32)

        self.env = env
        self.name = f"agent_{name}"
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.steps = steps
        self.K_steps = K_steps
        self.L_steps = L_steps
        self.rollout_steps = rollout_steps
        self.random_seed = random_seed
        self.gamma = gamma
        
        #stats
        self.avg_reward = [0 for _ in range(10)]
        self.accum_reward = 0
        self.cum_reward = []
        self.entropy_rate = []

    def rollout(self, bootstrap=False):
        log_probs, values, rewards, masks, states = [], [], [], [], []
        rollout_reward, entropy = 0, 0
        obs = self.env.reset()
        done = False
        for _ in range(self.rollout_steps):
    
            obs = T.tensor([obs], dtype=T.float).to(self.actorcritic.device)
            dist, v = self.actorcritic(obs)

            action = dist.sample()
            
            obs_, reward, done, _ = self.env.step(action.cpu().numpy()[0])

            log_prob = dist.log_prob(action)
            entropy += -dist.entropy()

            states.append(obs)
            values.append(v)
            log_probs.append(log_prob.unsqueeze(0).to(self.actorcritic.device))
            rewards.append(T.tensor([reward]).to(self.actorcritic.device))

            # non-episodic, (i.e use all rewards)
            #masks.append(T.tensor([1-int(done)], dtype=T.float).to(self.actorcritic.device))
            #rollout_reward += reward*(1-int(done))
            rollout_reward += reward
            self.accum_reward += reward
            self.cum_reward.append(self.accum_reward)
            
            obs = obs_

            # No need, since non-episodic
            '''
            if done:
                break
            '''

        obs_ = T.tensor([obs_], dtype=T.float).to(self.actorcritic.device)
        _, v = self.actorcritic(obs_)
        
        # Calc discounted returns
        R = v
        discounted_returns = []
        for step in reversed(range(len(rewards))):
            #R = rewards[step] + self.gamma * R * masks[step]
            R = rewards[step] + self.gamma * R
            discounted_returns.append(R)
        discounted_returns.reverse()

        self.avg_reward = self.avg_reward[1:] 
        self.avg_reward.append(rollout_reward / self.rollout_steps)
        ar = T.tensor(self.avg_reward, dtype=T.float).to(self.actorcritic.device)
        eps_en = self.meta_mlp(ar)

        entropy = entropy / self.rollout_steps
        self.entropy_rate.append(eps_en.item()) 
        
        log_probs = T.cat(log_probs)
        values = T.cat(values)
        returns = T.cat(discounted_returns)
        advantage = returns - values

        # Compute losses
        actor_loss = -T.mean((log_probs * advantage.detach()))
        critic_loss = 0.5 * T.mean(advantage.pow(2))

        if bootstrap:
            return actor_loss, states
        else:
            return actor_loss + critic_loss + eps_en * entropy

    def kl_matching_function(self, ac_k, tb, states, ac_k_state_dict):
        with T.no_grad():
            dist_tb = [tb(states[i])[0] for i in range(len(states))]

        TorchOpt.recover_state_dict(ac_k, ac_k_state_dict)
        dist_k = [ac_k(states[i])[0] for i in range(len(states))]
        
        # KL Div between dsitributions of TB and AC_K, respectively
        kl_div = sum([kl_divergence(dist_tb[i], dist_k[i]) for i in range(len(states))])

        return kl_div

    def plot_results(self):

        cr = plt.figure(figsize=(10, 10)) 
        plt.plot(self.cum_reward)
        plt.xlabel('Steps')
        plt.ylabel('Cumulative Reward')
        plt.savefig('res/cumulative_reward')
        plt.close(cr)

        er = plt.figure(figsize=(10, 10))
        plt.plot(list(range(1, 18_750 * 16 + 1, 16)), self.entropy_rate[-18_750:])
        plt.xlabel('Steps (Last 300,000 of 4.8M steps)')
        plt.ylabel('Entropy Rate')
        plt.savefig('res/entropy_rate')
        plt.close(er)

    def run(self):

        outer_range = self.steps // self.rollout_steps
        outer_range = outer_range // (self.K_steps + self.L_steps)
        ct = 0
        for _ in range(outer_range):
            for _ in range(self.K_steps):
                loss = self.rollout()
                self.actorcritic.optim.step(loss)
            k_state_dict = TorchOpt.extract_state_dict(self.actorcritic)
            
            for _ in range(self.L_steps-1):
                loss = self.rollout()
                self.actorcritic.optim.step(loss)
            k_l_m1_state_dict = TorchOpt.extract_state_dict(self.actorcritic)

            bootstrap_loss, states = self.rollout(bootstrap=True)
            self.actorcritic.optim.step(bootstrap_loss)

            # KL-Div Matching loss
            kl_matching_loss = self.kl_matching_function(self.ac_k, self.actorcritic, states, k_state_dict)
            
            # MetaMLP update
            self.meta_mlp.optim.zero_grad()
            kl_matching_loss.backward()
            self.meta_mlp.optim.step()

            # Use most recent params and stop grad
            TorchOpt.recover_state_dict(self.actorcritic, k_l_m1_state_dict)
            TorchOpt.stop_gradient(self.actorcritic)
            TorchOpt.stop_gradient(self.actorcritic.optim)

            ct += self.rollout_steps*((self.K_steps + self.L_steps))

            # print stats
            if ct %1000 == 0:
                print(f"CR and ER, step# {ct}:")
                print(self.cum_reward[-1])
                print(self.entropy_rate[-1])
                print("###")

    def save_models(self):
        self.actorcritic.save_checkpoint()
        self.meta_mlp.save_checkpoint()

    def load_models(self):
        self.actorcritic.load_checkpoint()
        self.meta_mlp.load_checkpoint()

if __name__ == "__main__":
    '''Driver code'''
    steps = 4_800_000
    K_steps = 3
    L_steps = 5
    rollout_steps = 16
    random_seed = 5
    env = TwoColorGridWorld()
    n_actions = 4
    input_dims = [env.observation_space.shape[0]]
    gamma = 0.99
    alpha = 0.1
    m_alpha = 1e-4
    betas = (0.9, 0.999)
    eps = 1e-4
    name = 'meta_agent_bmg' 

    # set seed
    T.cuda.manual_seed(random_seed)
    T.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    agent = Agent(input_dims, n_actions, gamma, alpha, m_alpha, betas, eps, name, env, 
                    steps, K_steps, L_steps, rollout_steps, random_seed)
    agent.run()
    print("done")
    agent.plot_results()
