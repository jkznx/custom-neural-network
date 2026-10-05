from pathlib import Path
import json
import random
import numpy as np
import torch
from torch.nn import functional as F
from model import LaneNet


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device(value):
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu') if value == 'auto' else torch.device(value)


def loss_function(logits, targets):
    probs, truth = logits.sigmoid().flatten(1), targets.flatten(1)
    dice = (2*(probs*truth).sum(1)+1e-6)/(probs.sum(1)+truth.sum(1)+1e-6)
    return .5*F.binary_cross_entropy_with_logits(logits, targets)+.5*(1-dice.mean())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def load_checkpoint(path, device):
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state.get('architecture') != 'LaneNet-v1':
        raise ValueError('Checkpoint must be LaneNet-v1')
    model = LaneNet(state['config']['base'])
    model.load_state_dict(state['model_state'])
    state.pop('optimizer_state', None)
    state.pop('model_state', None)
    return model.to(device).eval(), state
