import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import os
import gym
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
        self.env.set_seed(random_seed)
        self.gamma = gamma

        #stats
        self.average_reward = [0 for _ in range(10)]
        self.reward_history = 0
        self.r_per_step = []
        self.cumulative_rewards = []
        self.entropy_rate = []

    def rollout(self, bootstrap=False):
        log_probs, values, rewards, masks, states = [], [], [], [], []
        rollout_reward = 0
        entropy = 0
        obs = self.env.reset()
        done = False
        for _ in range(self.rollout_steps):
    
            obs = T.tensor([obs], dtype=T.float).to(self.actorcritic.device)
            dist, v = self.actorcritic(obs)

            action = dist.sample()

            obs_, reward, done, _ = self.env.step(action.numpy()[0])
            self.reward_history += reward

            log_prob = dist.log_prob(action)
            entropy += -dist.entropy()
        
            states.append(obs)
            log_probs.append(log_prob.unsqueeze(0).to(self.actorcritic.device))
            values.append(v)
            rewards.append(T.tensor([reward], dtype=T.float).to(self.actorcritic.device))

            # non-episodic, set all masks to 1 (i.e use all rewards)
            #masks.append(T.tensor([1-int(done)], dtype=T.float).to(self.actorcritic.device))
            #rollout_reward += reward*(1-int(done))
            masks.append(T.tensor([1], dtype=T.float).to(self.actorcritic.device))
            rollout_reward += reward
            
            obs = obs_

            self.cumulative_rewards.append(self.reward_history)
            #self.r_per_step.append(self.cumulative_rewards)

            # No need, since non-episodic
            '''
            if done:
                break
            '''

        obs_ = T.tensor([obs_], dtype=T.float).to(self.actorcritic.device)
        _, v = self.actorcritic(obs_)

        # Calc discounted returns
        R = v
        adjusted_returns = []
        for step in reversed(range(len(rewards))):
            R = rewards[step] + self.gamma * R * masks[step]
            adjusted_returns.insert(0, R)

        log_probs = T.cat(log_probs)
        values    = T.cat(values)
        entropy = entropy / self.rollout_steps

        self.average_reward[:-1] = self.average_reward[1:] 
        self.average_reward[-1] = rollout_reward / self.rollout_steps
        eps_en = self.meta_mlp(T.tensor(self.average_reward, dtype=T.float).to(self.actorcritic.device))

        self.entropy_rate.append(float(eps_en)) 
        
        returns = T.cat(adjusted_returns).detach()
        advantage = returns - values

        # Compute losses
        actor_loss  = -(log_probs * advantage.detach()).mean()
        critic_loss = 0.5 * advantage.pow(2).mean()

        # Terms of policy gradient, temporal difference and entropy 
        if bootstrap:
            loss = actor_loss
            return loss, states
        else:
            loss = actor_loss + critic_loss + eps_en * entropy
            return loss

    def matching_function(self, Kth_model, TB_model, states, Kth_state_dict):

        TorchOpt.recover_state_dict(Kth_model, Kth_state_dict)

        dist_K = [Kth_model(states[i])[0] for i in range(len(states))]
        with T.no_grad():
            dist_T = [TB_model(states[i])[0] for i in range(len(states))]

        KL_loss = sum([kl_divergence(dist_T[i], dist_K[i]) for i in range(len(states))])

        return KL_loss

    def plot_results(self):
        
        #cr = np.array(self.cumulative_rewards)
        #rs = np.array(self.r_per_step)
        #er = np.array(self.entropy_rate)

        fig1 = plt.figure(figsize=(20, 20)) 
        plt.plot(self.cumulative_rewards)
        plt.xlabel('Steps')
        plt.ylabel('Cumulative Reward')
        plt.savefig('cumul_reward')
        plt.close(fig1)

        '''
        fig2 = plt.figure(figsize=(20, 20)) 
        plt.plot(self.r_per_step, color = "blue")
        plt.ylim([-0.5, 0.5])
        plt.xlabel('Steps')
        plt.ylabel('Reward/Step')
        plt.savefig('rew_per_step')
        plt.close(fig2)
        '''

        fig3 = plt.figure(figsize=(20, 20))
        plt.plot(self.entropy_rate)
        plt.xlabel('Steps')
        plt.ylabel('Entropy Rate')
        plt.savefig('entropy_rate')
        plt.close(fig3)

    def run(self):

        outer_steps = self.steps // self.rollout_steps // (self.K_steps + self.L_steps)

        for outer_step in range(outer_steps):
            for inner_step in range(self.K_steps + self.L_steps):
                #print(f"outerstep#: {outer_step}")
                if inner_step == self.K_steps + self.L_steps - 1:
                    bootstrap_loss, states = self.rollout(bootstrap=True)
                    self.actorcritic.optim.step(bootstrap_loss)
                else:
                    loss = self.rollout()
                    self.actorcritic.optim.step(loss)

                if inner_step == self.K_steps - 1: 
                    Kth_state_dict = TorchOpt.extract_state_dict(self.actorcritic)            

                if inner_step == self.K_steps + self.L_steps - 2:
                    saved_state_dict = TorchOpt.extract_state_dict(self.actorcritic)

            # KL-Div loss
            KL_loss = self.matching_function(self.ac_k, self.actorcritic, states, Kth_state_dict)
            
            # Meta update
            self.meta_mlp.optim.zero_grad()
            KL_loss.backward()
            self.meta_mlp.optim.step()

            # Use most recent params
            TorchOpt.recover_state_dict(self.actorcritic, saved_state_dict)
            TorchOpt.stop_gradient(self.actorcritic)
            TorchOpt.stop_gradient(self.actorcritic.optim)

            '''
            # test
            if (outer_step + 1) % 10== 0:
                test_env = gym.make(env_id)
                test_env.seed(self.random_seed)
                score = 0
                for _ in range(100):
                    obs = test_env.reset()
                    for i in range(1000):
                        action = self.actorcritic.choose_action(obs)
                        obs_, reward, done, info = test_env.step(action)
                        score += reward
                        obs = obs_
                        if done:
                            break
                    #scores.append(score)
                avg_score = score/100
                print(f"step: {outer_step+1} avg_100_eps_score: {avg_score} meta_param(entropy): {entropy}")
                if avg_score > test_env.spec.reward_threshold:
                    break
            '''

    def save_models(self):
        self.actorcritic.save_checkpoint()
        self.meta_mlp.save_checkpoint()

    def load_models(self):
        self.actorcritic.load_checkpoint()
        self.meta_mlp.load_checkpoint()

if __name__ == "__main__":
    '''Driver code'''
    steps = 200000
    K_steps = 7
    L_steps = 9
    rollout_steps = 16
    random_seed = 5
    env = TwoColorGridWorld()
    n_actions = 4
    input_dims = [env.observation_space.shape[0]]
    gamma = 0.99
    alpha = 0.1
    m_alpha = 0.0001
    betas=(0.9, 0.999)
    eps=1e-4
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
