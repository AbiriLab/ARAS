"""Noisy-input robustness evaluation for ARAS (inference in the loop) and HO.

Noise corrupts only what the controller sees; ground-truth metrics untouched.
Usage: python test_noisy.py --method ARAS --scenario fixed --kind flip --p 0.10
"""
import argparse, collections, datetime, json, os, random

import numpy as np

from noisy_common import InferenceMasker, corrupt, to_tensors

parser = argparse.ArgumentParser()
parser.add_argument("--method", choices=["ARAS", "HO"], required=True)
parser.add_argument("--scenario", choices=["fixed", "dynamic_pickup", "dynamic_dropoff", "dynamic_both"], required=True)
parser.add_argument("--kind", choices=["flip", "drop"], default="flip")
parser.add_argument("--p", type=float, default=0.0)
parser.add_argument("--episodes", type=int, default=500)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--model", default="./models/ARAS_v8_inference.pt")
parser.add_argument("--outdir", default="./noise_results")
parser.add_argument("--tag", default="")
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
noise_rng = np.random.default_rng(12345 + args.seed)


def run_aras():
    import torch
    from jaco_env import jacoDiverseObjectEnv
    from networks import DQN

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    STACK = 4
    env = jacoDiverseObjectEnv(actionRepeat=80, renders=False, isDiscrete=True, maxSteps=70,
                               dv=0.02, AutoXDistance=False, AutoGrasp=True, width=64, height=64,
                               numObjects=3, numContainers=3, scenario=args.scenario)
    env.reset()
    torch.manual_seed(args.seed)
    masker = InferenceMasker(env, kind=args.kind, p=args.p, rng=noise_rng)

    net = DQN(64, 64, env.action_space.n, stack_size=STACK).to(device)
    ck = torch.load(args.model, map_location=device)
    net.load_state_dict(ck["policy_net_state_dict"])
    net.eval()

    metrics = []
    for ep in range(args.episodes):
        env.reset()
        masker.reset()
        seg, tok, gate_open, correct = masker.view(env._observation[1])
        s, y = to_tensors(seg, tok, device)
        st_s = collections.deque(STACK * [s], maxlen=STACK)
        st_y = collections.deque(STACK * [y], maxlen=STACK)
        steps = inputs = errs = amps = gate_n = inf_ok = inf_n = 0
        done = False
        while not done:
            steps += 1
            gate_n += int(gate_open)
            if correct is not None:
                inf_n += 1
                inf_ok += int(correct)
            with torch.no_grad():
                a = net(torch.cat(tuple(st_s), 1), torch.cat(tuple(st_y), 1)).max(1)[1].item()
            _, reward, done, info = env.step(a)
            prog, _, total_reward = reward
            if a in (0, 1):
                inputs += 1
            else:
                amps += 1
            if prog < 0:
                errs += 1
            seg, tok, gate_open, correct = masker.view(env._observation[1])
            s, y = to_tensors(seg, tok, device)
            st_s.append(s)
            st_y.append(y)
        metrics.append({
            "episode": ep, "steps": steps, "success": bool(total_reward == 1),
            "total_inputs": inputs / 0.05, "error_actions": errs, "amplified_actions": amps,
            "gate_open_frac": gate_n / steps,
            "infer_acc": (inf_ok / inf_n) if inf_n else None,
        })
        if (ep + 1) % 25 == 0:
            print(f"ep {ep+1}/{args.episodes} SR={np.mean([m['success'] for m in metrics]):.3f}", flush=True)
    return metrics


def run_ho():
    from jaco_env import jacoDiverseObjectEnv
    from hindsight_optimizer import HindsightOptimizer

    np.random.seed(42)
    random.seed(42)
    env = jacoDiverseObjectEnv(actionRepeat=80, renders=False, isDiscrete=True, maxSteps=70,
                               dv=0.02, AutoXDistance=False, AutoGrasp=True, width=64, height=64,
                               numObjects=3, numContainers=3, scenario=args.scenario)
    env.reset()
    opt = HindsightOptimizer(env)
    NAME2TOK = {"left": 1, "neutral": 0, "right": -1}
    TOK2NAME = {v: k for k, v in NAME2TOK.items()}

    metrics = []
    for ep in range(args.episodes):
        env.reset()
        opt.reset()
        steps = inputs = errs = amps = 0
        done = False
        while not done:
            u = opt.generate_synthetic_user_input()
            u = TOK2NAME[corrupt(NAME2TOK[u], args.kind, args.p, noise_rng)]
            a = opt.select_action(u)
            if a in (0, 1):
                inputs += 1
            else:
                amps += 1
            _, reward, done, info = env.step(a)
            if reward[0] < 0:
                errs += 1
            steps += 1
        ok = info.get("task_success", 0) > 0
        metrics.append({"episode": ep, "steps": steps, "success": bool(ok),
                        "total_inputs": inputs / 0.05, "error_actions": errs,
                        "amplified_actions": amps})
        if (ep + 1) % 25 == 0:
            print(f"ep {ep+1}/{args.episodes} SR={np.mean([m['success'] for m in metrics]):.3f}", flush=True)
    return metrics


metrics = run_aras() if args.method == "ARAS" else run_ho()

os.makedirs(args.outdir, exist_ok=True)
mean = lambda k: float(np.mean([m[k] for m in metrics if m.get(k) is not None]))
summary = {
    "method": args.method, "scenario": args.scenario, "noise_kind": args.kind,
    "noise_p": args.p, "num_episodes": args.episodes, "seed": args.seed,
    "model": args.model if args.method == "ARAS" else None,
    "inference_in_loop": args.method == "ARAS",
    "success_rate": mean("success"), "avg_steps": mean("steps"),
    "avg_user_inputs": mean("total_inputs"), "avg_error_actions": mean("error_actions"),
    "avg_amplified_actions": mean("amplified_actions"),
    "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
}
if args.method == "ARAS":
    summary["avg_gate_open_frac"] = mean("gate_open_frac")
    summary["avg_infer_acc"] = mean("infer_acc")
tag = args.tag or f"{args.method}_{args.scenario}_{args.kind}{int(round(args.p*100)):02d}"
json.dump(summary, open(os.path.join(args.outdir, f"summary_{tag}.json"), "w"), indent=1)
json.dump(metrics, open(os.path.join(args.outdir, f"episodes_{tag}.json"), "w"))
print(json.dumps(summary, indent=1))
