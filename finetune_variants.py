"""Parametrized fine-tuning of ARAS with goal inference in the loop.

Variants over token noise (fixed level or per-episode mixed curriculum),
training scenario mix, and exploration. Resumes from the shipped ARAS_v7.
"""
import argparse, collections, datetime, os, random
from itertools import count

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple

from jaco_env import jacoDiverseObjectEnv
from utils import ReplayMemory
from networks import DQN
from noisy_common import InferenceMasker, to_tensors

parser = argparse.ArgumentParser()
parser.add_argument("--out", required=True, help="output checkpoint path")
parser.add_argument("--kind", default="flip", help="flip|drop|mixed")
parser.add_argument("--p", default="0.0", help="float or 'mixed'")
parser.add_argument("--scenarios", default="fixed", help="comma list, per-episode random choice")
parser.add_argument("--episodes", type=int, default=5000)
parser.add_argument("--eps-start", type=float, default=0.15)
parser.add_argument("--eps-end", type=float, default=0.05)
parser.add_argument("--eps-decay", type=int, default=1000)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)

PRETRAINED = "./models/ARAS_v7_bs64_ss4_rb30000_gamma0.5_decaylf20000_lr1e-05.pt"
BATCH_SIZE, GAMMA, LR = 64, 0.5, 1e-5
REPLAY, TARGET_UPDATE, STACK = 30000, 1000, 4
SCENARIOS = args.scenarios.split(",")
P_LEVELS = [0.0, 0.05, 0.10, 0.20]
KINDS = ["flip", "drop"]

env = jacoDiverseObjectEnv(actionRepeat=80, renders=False, isDiscrete=True, maxSteps=70,
                           dv=0.02, AutoXDistance=False, AutoGrasp=True, width=64, height=64,
                           numObjects=3, numContainers=3, scenario=SCENARIOS[0])
env.reset()
masker = InferenceMasker(env, kind="flip", p=0.0, rng=np.random.default_rng(999 + args.seed))

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))
memory = ReplayMemory(REPLAY, Transition)
policy_net = DQN(64, 64, env.action_space.n, stack_size=STACK).to(device)
target_net = DQN(64, 64, env.action_space.n, stack_size=STACK).to(device)
optimizer = optim.Adam(policy_net.parameters(), lr=LR)
ck = torch.load(PRETRAINED, map_location=device)
policy_net.load_state_dict(ck["policy_net_state_dict"])
target_net.load_state_dict(ck["target_net_state_dict"])
optimizer.load_state_dict(ck["optimizer_policy_net_state_dict"])
target_net.eval()
print(f"loaded pretrained; variant out={args.out} kind={args.kind} p={args.p} "
      f"scen={SCENARIOS} eps={args.eps_start}->{args.eps_end} n={args.episodes}", flush=True)


def episode_noise():
    kind = random.choice(KINDS) if args.kind == "mixed" else args.kind
    p = random.choice(P_LEVELS) if args.p == "mixed" else float(args.p)
    return kind, p


def select_action(s, y, ep):
    eps = max(args.eps_end, args.eps_start - ep / args.eps_decay * (args.eps_start - args.eps_end))
    if random.random() > eps:
        with torch.no_grad():
            return policy_net(s, y).max(1)[1].view(1, 1)
    return torch.tensor([[random.choice([0, 1, 3])]], device=device, dtype=torch.long)


def optimize():
    if len(memory) < BATCH_SIZE:
        return
    batch = Transition(*zip(*memory.sample(BATCH_SIZE)))
    non_final = torch.tensor([s is not None for s in batch.next_state], device=device, dtype=torch.bool)
    nf = [s for s in batch.next_state if s is not None]
    sb = torch.cat([s[0] for s in batch.state]); yb = torch.cat([s[1] for s in batch.state])
    ab = torch.cat(batch.action); rb = torch.cat(batch.reward)
    q = policy_net(sb, yb).gather(1, ab)
    nxt = torch.zeros(BATCH_SIZE, device=device)
    if nf:
        nxt[non_final] = target_net(torch.cat([s[0] for s in nf]), torch.cat([s[1] for s in nf])).max(1)[0].detach()
    loss = F.smooth_l1_loss(q, (nxt * GAMMA + rb).unsqueeze(1))
    optimizer.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1)
    optimizer.step()


total_rewards, best_mean = [], None
t0 = datetime.datetime.now()
for ep in range(args.episodes):
    env._scenario = random.choice(SCENARIOS)
    masker.kind, masker.p = episode_noise()
    env.reset()
    masker.reset()
    seg, tok, _, _ = masker.view(env._observation[1])
    s, y = to_tensors(seg, tok, device)
    st_s = collections.deque(STACK * [s], maxlen=STACK)
    st_y = collections.deque(STACK * [y], maxlen=STACK)
    for t in count():
        s_t = torch.cat(tuple(st_s), 1); y_t = torch.cat(tuple(st_y), 1)
        a = select_action(s_t, y_t, ep)
        _, reward, done, _ = env.step(a.item())
        r = torch.tensor([reward[2]], device=device, dtype=torch.float32)
        seg, tok, _, _ = masker.view(env._observation[1])
        ns, ny = to_tensors(seg, tok, device)
        if not done:
            nst_s = st_s.copy(); nst_s.append(ns)
            nst_y = st_y.copy(); nst_y.append(ny)
            next_pair = (torch.cat(tuple(nst_s), 1), torch.cat(tuple(nst_y), 1))
        else:
            next_pair = None
        memory.push((s_t, y_t), a, next_pair, r)
        if not done:
            st_s, st_y = nst_s, nst_y
        optimize()
        if done:
            total_rewards.append(float(reward[2] == 1))
            break
    if ep % TARGET_UPDATE == 0 and ep > 0:
        target_net.load_state_dict(policy_net.state_dict())
    mean100 = np.mean(total_rewards[-100:])
    if ep > 200 and (best_mean is None or mean100 > best_mean):
        best_mean = mean100
        torch.save({"policy_net_state_dict": policy_net.state_dict(),
                    "target_net_state_dict": target_net.state_dict(),
                    "optimizer_policy_net_state_dict": optimizer.state_dict()}, args.out)
    if (ep + 1) % 50 == 0:
        print(f"ep {ep+1}/{args.episodes} SR(100)={mean100:.3f} best={best_mean} "
              f"elapsed={datetime.datetime.now()-t0}", flush=True)
print("done; best mean SR:", best_mean, flush=True)
