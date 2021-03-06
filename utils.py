"""
Functions that use multiple times
"""

from torch import nn
import torch
import numpy as np
from torch.autograd import Variable
from shared_adam import exp_lr_scheduler


def v_wrap(np_array, dtype=np.float32):
    if np_array.dtype != dtype:
        np_array = np_array.astype(dtype)
    return Variable(torch.from_numpy(np_array))


def set_init(layers):
    for layer in layers:
        nn.init.normal(layer.weight, mean=0., std=.3)
        nn.init.constant(layer.bias, 0.3)


def push_and_pull(opt, lnet, gnet, done, s_, bs, ba, br, gamma, ep):
    if done:
        v_s_ = 0.               # terminal
    else:
        v_s_ = lnet.forward(v_wrap(s_[None, :]))[-1].data.numpy()[0, 0]

    buffer_v_target = []
    for r in br[::-1]:    # reverse buffer r
        v_s_ = r + gamma * v_s_
        buffer_v_target.append(v_s_)
    buffer_v_target.reverse()

    a_loss, c_loss, loss = lnet.loss_func(
        # loss = lnet.loss_func(
        v_wrap(np.vstack(bs)),
        v_wrap(np.array(ba), dtype=np.int64) if ba[0].dtype == np.int64 else v_wrap(np.vstack(ba)),
        v_wrap(np.array(buffer_v_target)[:, None]))

    # calculate local gradients and push local parameters to global
    opt.zero_grad()
    loss.backward()
    # c_loss.backward(retain_graph=True)
    # a_loss.backward()
    nn.utils.clip_grad_norm(lnet.parameters(), 10.0)
    for lp, gp in zip(lnet.parameters(), gnet.parameters()):
        gp._grad = lp.grad
    # if not ep%50:
        # for lp in lnet.parameters():
        #     print( lp.grad )
        # print( a_loss, c_loss )
    exp_lr_scheduler( opt, ep ).step()
    # opt.step()

    # pull global parameters
    lnet.load_state_dict(gnet.state_dict())

    return a_loss, c_loss


def record(global_ep, global_ep_r, ep_r, res_queue, name, a_loss, c_loss):
    with global_ep.get_lock():
        global_ep.value += 1
    with global_ep_r.get_lock():
        global_ep_r.value = ep_r
        # if global_ep_r.value == 0.:
        #     global_ep_r.value = ep_r
        # else:
        #     global_ep_r.value = global_ep_r.value * 0.99 + ep_r * 0.01
    res_queue.put([global_ep_r.value, a_loss, c_loss])
    if not global_ep.value%100:
        print(
            name,
            "Ep:", global_ep.value,
            "| Ep_r: %.0f" % global_ep_r.value,
        )