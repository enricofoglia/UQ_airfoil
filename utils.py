def count_parameters(model):
    if model.kind == 'ensemble':
         return sum(p.numel() for p in model[0].parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)