from torch import nn, optim
from torch.optim import lr_scheduler
from torchvision import transforms as T

from .schedulers import LinearWarmupScheduler

def make_criterion(config):
    crit_config = config['training']['criterion']
    cls = getattr(nn, crit_config['name'])
    return cls(**crit_config.get('params', {}))

def make_optimizer(config, model):
    opt_config= config['training']['optimizer']
    cls = getattr(optim, opt_config['name'])
    return cls(model.parameters(), **opt_config.get('params', {}))

def make_schedulers(config, optimizer, num_epochs, warmup_steps):
    sched_config = config['training']['lr_scheduler']
    main = sched_config['main']
    warm = sched_config['warmup']
    main_cls = getattr(lr_scheduler, main['name'])

    main_kwargs = dict(main.get('params', {}),
                       T_max=num_epochs - config['training']['warmup_epochs'])
    warm_kwargs = dict(warm.get('params', {}),
                       warmup_steps=warmup_steps,
                       start_lr=config['training']['warmup_initial_learning_rate'],
                       target_lr=config['training']['warmup_final_learning_rate'])

    return {
        'main': main_cls(optimizer, **main_kwargs),
        'warmup': LinearWarmupScheduler(optimizer, **warm_kwargs),
    }

def make_transforms(sequence):
    ops = []
    for entry in sequence:
        cls = getattr(T, entry['name'])
        params = entry.get('params') or {}
        ops.append(cls(**params))
    return T.Compose(ops)