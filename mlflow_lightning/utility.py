import torch.optim as optim

def get_optimizer(model, optimizer_params):
    if optimizer_params["optimizer"] == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), 
                                lr=optimizer_params["learning_rate"],
                                weight_decay=optimizer_params["weight_decay"])
        return optimizer
    elif optimizer_params["optimizer"] == "SGD":
        optimizer = optim.SGD(model.parameters(),
                              lr=optimizer_params["learning_rate"],
                              momentum=optimizer_params["momentum"],
                              weight_decay=optimizer_params["weight_decay"])
        return optimizer
    else:
        assert False, f'Unknown optimizer: {optimizer_params["optimizer"]}'

    
def get_scheduler(optimzer, scheduler_params):
    if scheduler_params["scheduler"] == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimzer, 
                                                         mode=scheduler_params["mode"],
                                                         factor=scheduler_params["factor"],
                                                         patience=scheduler_params["patience"],
                                                         )
        return scheduler
    elif scheduler_params['scheduler'] == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimzer, 
                                              step_size=scheduler_params["step_size"],
                                              gamma=scheduler_params["gamma"])
        return scheduler
    else:
        assert False, f'Unknown scheduler: {scheduler_params["scheduler"]}'