"""Shared machinery for inference-in-the-loop evaluation and fine-tuning.

Implements the paper's Sec. II goal inference (window tau=4, sigma^2=0.5,
kappa=0.8) over the lateral command channel, plus token corruption models.
"""
import collections

import numpy as np

TAU, SIGMA2, KAPPA, DEADBAND = 4, 0.5, 0.8, 0.025


def corrupt(token, kind, p, rng):
    """token in {-1,0,1}; flip -> random wrong token, drop -> neutral."""
    if p <= 0 or rng.random() >= p:
        return int(token)
    if kind == "drop":
        return 0
    return int(rng.choice([t for t in (-1, 0, 1) if t != token]))


class BayesGoalInference:
    """Windowed Bayesian inference over lateral goal candidates."""

    def __init__(self):
        self.lik_window = collections.deque(maxlen=TAU)
        self.phase = None

    def reset(self):
        self.lik_window.clear()
        self.phase = None

    def step(self, token, cand_ys, grip_y, phase):
        if phase != self.phase:            # object -> bin transition resets belief
            self.lik_window.clear()
            self.phase = phase
        mus = np.array([0 if abs(y - grip_y) < DEADBAND else np.sign(y - grip_y)
                        for y in cand_ys])
        lik = np.exp(-((token - mus) ** 2) / (2 * SIGMA2))
        self.lik_window.append(lik)
        post = np.prod(np.stack(self.lik_window), axis=0)
        s = post.sum()
        post = post / s if s > 0 else np.full(len(cand_ys), 1.0 / len(cand_ys))
        conf = post.max()
        goal_idx = int(post.argmax()) if conf > KAPPA else None
        return goal_idx, conf, post


class InferenceMasker:
    """Builds the policy's observation from the inferred (not oracle) goal.

    Captures the raw segmentation by monkeypatching jaco_env.modify_segmentation,
    then re-masks it from the goal inferred from the (possibly corrupted) token.
    """

    NONE_ID = -999                          # matches no segmentation pixel

    def __init__(self, env, kind="flip", p=0.0, rng=None):
        import jaco_env as je
        from utils import modify_segmentation as orig
        self.env = env
        self.kind, self.p = kind, p
        self.rng = rng or np.random.default_rng(0)
        self.orig_modify = orig
        self.captured = {}
        je.modify_segmentation = self._capture

    def _capture(self, seg, obj_id, cont_id, gstate):
        seg = np.array(seg)
        if len(seg.shape) == 1:
            side = int(np.sqrt(seg.shape[0]))
            seg = seg.reshape(side, side)
        self.captured["raw"] = seg.copy()
        return self.orig_modify(seg, obj_id, cont_id, gstate)

    def reset(self):
        self.inference = BayesGoalInference()

    def view(self, token):
        """-> (seg64x64 uint8, noisy_token int, gate_open bool, infer_correct or None)"""
        import pybullet as pb
        env = self.env
        phase = env._gripperState
        noisy = corrupt(token, self.kind, self.p, self.rng)
        cands = list(env._objectUids) if phase == "open" else list(env.container_uid)
        ys = [pb.getBasePositionAndOrientation(c)[0][1] for c in cands]
        gy = env._getGripper()[1]
        gidx, conf, _ = self.inference.step(noisy, ys, gy, phase)
        gate_open = gidx is not None
        held = env.intention_object if phase == "close" else self.NONE_ID
        if gate_open:
            goal_id = cands[gidx]
            truth = env.intention_object if phase == "open" else env.intention_container
            correct = goal_id == truth
            obj_id = goal_id if phase == "open" else held
            cont_id = goal_id if phase == "close" else self.NONE_ID
        else:
            correct = None
            obj_id, cont_id = held, self.NONE_ID       # goal mask withheld
        seg = self.orig_modify(self.captured["raw"], obj_id, cont_id, phase)
        return seg, noisy, gate_open, correct


def to_tensors(seg, token, device):
    import torch
    s = torch.from_numpy(seg.copy()).unsqueeze(0).unsqueeze(0).to(device)
    y = torch.tensor([[[float(token)]]], dtype=torch.float32, device=device)
    return s, y
