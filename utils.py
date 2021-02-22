import math
import os
import random
from collections import deque, defaultdict
import pickle as pkl

import gym
import pathlib
import numpy as np
import re
from omegaconf import OmegaConf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as pyd


class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def make_dir(*path_parts):
    dir_path = os.path.join(*path_parts)
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path


def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias


def chain(*iterables):
    for it in iterables:
        yield from it


def save(obj, file_path):
    with open(file_path, 'wb') as f:
        torch.save(obj, f)


def load(file_path):
    if isinstance(file_path, str):
        file_path = pathlib.Path(file_path).expanduser()
    with file_path.open('rb') as f:
        return torch.load(f)


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


def mlp(input_dim,
        hidden_dim,
        output_dim,
        hidden_depth,
        output_mod=None,
        use_ln=False):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim)]
        if use_ln:
            mods += [nn.LayerNorm(hidden_dim), nn.Tanh()]
        else:
            mods += [nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk


def to_np(t):
    if t is None:
        return None
    elif t.nelement() == 0:
        return np.array([])
    else:
        return t.cpu().detach().numpy()


def parse_run_overrides(exp_dirs):
    exp_dirs = exp_dirs.split(':')

    def parse_cfg(cfg_list):
        cfg = {}
        for item in cfg_list:
            parts = item.split('=')
            cfg[parts[0]] = parts[1]
        return cfg

    runs = {}
    for exp_dir in exp_dirs:
        exp_dir = pathlib.Path(exp_dir).expanduser()
        for override in exp_dir.glob('**/overrides.yaml'):
            with override.open('rb') as f:
                cfg = parse_cfg(OmegaConf.load(f))
            path = override.parents[1]
            runs[path] = cfg
    return runs


def find_available_seeds(runs, env):
    avail_seeds = {}
    for path, cfg in runs.items():
        if cfg['env'] == env:
            snapshots = {}
            model_dir = path / 'model'
            for snap in model_dir.glob('expl_agent_*.pt'):
                snap_id = int(
                    re.match(r'expl_agent_(\d+).pt', snap.name).group(1))
                snapshots[snap_id] = snap
            avail_seeds[int(cfg['seed'])] = snapshots
    return avail_seeds


def find_pretrained_agent(exp_dirs, env, seed, step):
    runs = parse_run_overrides(exp_dirs)
    avail_seeds = find_available_seeds(runs, env)
    if len(avail_seeds) == 0:
        raise f'cannot find a pretrained agent for {env} {seed}'

    if seed in avail_seeds and step in avail_seeds[seed]:
        return avail_seeds[seed][step]

    for snapshots in avail_seeds.values():
        if step in snapshots:
            return snapshots[step]

    raise f'cannot find a pretrained agent for {env} {seed}'
    return None


class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

        self.base_dist = pyd.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu


class ClippedNormal(pyd.Normal):
    def __init__(self, loc, scale):
        super().__init__(loc, scale)

    def sample(self, sample_shape=torch.Size()):
        x = super().sample(sample_shape)
        return torch.clamp(x, -1.0, 1.0)

    def rsample(self, sample_shape=torch.Size()):
        x = super().rsample(sample_shape)
        return torch.clamp(x, -1.0, 1.0)