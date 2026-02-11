import torch
import torch.nn as nn
from copy import deepcopy

class ModelEMA:

    def __init__(self, model, decay=0.999, device=None):
        self.ema = deepcopy(model)
        self.ema.eval()
        self.decay = decay
        self.device = device

        if self.device is not None:
            self.ema.to(device=device)

        for param in self.ema.parameters():
            param.requires_grad = False

        self.updates = 0

    def update(self, model):

        self.updates += 1

        decay = min(self.decay, (1 + self.updates) / (10 + self.updates))

        with torch.no_grad():
            model_state = model.state_dict()
            ema_state = self.ema.state_dict()

            for key in ema_state.keys():
                if ema_state[key].dtype.is_floating_point:
                    ema_state[key].copy_(
                        ema_state[key] * decay + model_state[key] * (1 - decay)
                    )
                else:
                    ema_state[key].copy_(model_state[key])

    def update_buffer(self, model):

        with torch.no_grad():
            model_buffers = dict(model.named_buffers())
            ema_buffers = dict(self.ema.named_buffers())

            for key in ema_buffers.keys():
                if key in model_buffers:
                    ema_buffers[key].copy_(model_buffers[key])

    def state_dict(self):

        return {
            'ema_state_dict': self.ema.state_dict(),
            'decay': self.decay,
            'updates': self.updates,
        }

    def load_state_dict(self, state_dict):

        self.ema.load_state_dict(state_dict['ema_state_dict'])
        self.decay = state_dict['decay']
        self.updates = state_dict['updates']

    def module(self):

        return self.ema

    def eval(self):

        self.ema.eval()

    def __call__(self, *args, **kwargs):

        return self.ema(*args, **kwargs)

class TeacherEMA(ModelEMA):

    def __init__(self, model, decay=0.999, device=None, warmup_steps=0):
        super().__init__(model, decay, device)
        self.warmup_steps = warmup_steps

    def update(self, model, step=None):

        if step is not None and step < self.warmup_steps:
            warmup_decay = self.decay * (step / max(1, self.warmup_steps))
            original_decay = self.decay
            self.decay = warmup_decay
            super().update(model)
            self.decay = original_decay
        else:
            super().update(model)

    def synchronize(self, model):

        with torch.no_grad():
            model_state = model.state_dict()
            ema_state = self.ema.state_dict()

            for key in ema_state.keys():
                ema_state[key].copy_(model_state[key])

        self.updates = 0
