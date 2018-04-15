"""
Reinforcement Learning (A3C) using Pytroch + multiprocessing.
The most simple implementation for continuous action.

View more on my Chinese tutorial page [莫烦Python](https://morvanzhou.github.io/).
"""

import torch
import numpy as np
import torch.nn as nn
from utils import v_wrap, set_init, push_and_pull, record
import torch.nn.functional as F
import torch.multiprocessing as mp
from shared_adam import SharedAdam
import gym
import os
os.environ["OMP_NUM_THREADS"] = "1"

UPDATE_GLOBAL_ITER = 10
GAMMA = 0.9
MAX_EP = 6000

env = gym.make('CartPole-v0')
N_S = env.observation_space.shape[0]
N_A = env.action_space.n


class Net(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Net, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.b1 = nn.Linear(s_dim, 32)
        # self.bn1 = nn.BatchNorm1d(32, momentum=0.5)
        self.b2 = nn.Linear(32, 24)
        # self.bn2 = nn.BatchNorm1d(24, momentum=0.5)
        # self.b3 = nn.Linear(24,16)
        # self.bn3 = nn.BatchNorm1d(16, momentum=0.5)
        self.pi = nn.Linear(24, a_dim)
        self.v = nn.Linear(24, 1)
        set_init([self.b1, self.b2, self.pi, self.v])
        self.distribution = torch.distributions.Categorical

    def forward(self, x):

        b1_o = F.leaky_relu(self.b1(x))
        b2_o = F.leaky_relu(self.b2(b1_o))
        # b3_o = F.leaky_relu(self.b3(b2_o))
        logits = self.pi(b2_o)
        values = self.v(b2_o)

        return logits, values

    def choose_action(self, s):
        self.eval()
        logits, _ = self.forward(s)
        prob = F.softmax(logits, dim=1).data
        m = self.distribution(prob)
        return m.sample().numpy()[0]

    def loss_func(self, s, a, v_t):
        self.train()
        logits, values = self.forward(s)
        td = v_t - values
        c_loss = td.pow(2)
        
        probs = F.softmax(logits, dim=1)
        m = self.distribution(probs)
        exp_v = m.log_prob(a) * td.detach()
        a_loss = -exp_v
        total_loss = (c_loss + a_loss).mean()
        return a_loss.mean(), c_loss.mean(), total_loss


class Worker(mp.Process):
    def __init__(self, gnet, opt, global_ep, global_ep_r, res_queue, name):
        super(Worker, self).__init__()
        self.name = 'w%i' % name
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.gnet, self.opt = gnet, opt
        self.lnet = Net(N_S, N_A)           # local network
        self.env = gym.make('CartPole-v0').unwrapped

    def run(self):
        total_step = 1
        while self.g_ep.value < MAX_EP:
            s = self.env.reset()
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0.
            while True:
                # if self.name == 'w0':
                #     self.env.render()
                a = self.lnet.choose_action(v_wrap(s[None, :]))
                s_, r, done, _ = self.env.step(a)
                if done: r = -1
                ep_r += r
                buffer_a.append(a)
                buffer_s.append(s)
                buffer_r.append(r)

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net
                    # sync
                    a_loss, c_loss = push_and_pull(self.opt, self.lnet, self.gnet, done, s_, buffer_s, buffer_a, buffer_r, GAMMA, self.g_ep.value)
                    buffer_s, buffer_a, buffer_r = [], [], []

                    if done:  # done and print information
                        record(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name, a_loss.data.numpy()[0], c_loss.data.numpy()[0])
                        break
                s = s_
                total_step += 1
        self.res_queue.put(None)


if __name__ == "__main__":
    gnet = Net(N_S, N_A)        # global network
    gnet.share_memory()         # share the global parameters in multiprocessing
    opt = SharedAdam(gnet.parameters(), lr=5e-3)      # global optimizer
    global_ep, global_ep_r, res_queue, q_lock = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue(), mp.Lock()

    # parallel training
    workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, i) for i in range(mp.cpu_count())]
    # workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, i) for i in range(1)]

    [w.start() for w in workers]
    res = []                    # record episode reward to plot
    while True:
        r = res_queue.get()
        if r is not None:
            res.append(r)
        else:
            break
    [w.join() for w in workers]

    import matplotlib.pyplot as plt
    res = np.array( res )
    plt.plot(res[:,0], label='episode reward')
    plt.plot(res[:,1], label='actor net loss')
    plt.plot(res[:,2], label='critic net loss')
    plt.ylabel('Moving average ep reward, actor net loss, cretic net loss')
    plt.xlabel('Step')
    plt.legend()
    plt.show()

    # for _ in range(3):
    #     s = env.reset()
    #     while True:
    #         env.render()
    #         a = gnet.choose_action(v_wrap(s[None, :]))
    #         s_, r, done, _ = env.step(a)
    #         s = s_
    #         if done:
    #             break