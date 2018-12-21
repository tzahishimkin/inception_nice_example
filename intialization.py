from torch.nn import init


def xavier_initialize(model):
    modules = [
        m for n, m in model.named_modules() if
        'conv' in n or 'fc' in n
    ]

    parameters = [
        p for
        m in modules for
        p in m.parameters()
    ]

    for p in parameters:
        if p.dim() >= 2:
            init.xavier_normal(p)
        else:
            init.constant(p, 0)


def gaussian_intiailize(model, std=.01):
    modules = [
        m for n, m in model.named_modules() if
        'conv' in n or 'fc' in n
    ]

    parameters = [
        p for
        m in modules for
        p in m.parameters()
    ]

    for p in parameters:
        if p.dim() >= 2:
            init.normal_(p, std=std)
        else:
            init.constant_(p, 0)
