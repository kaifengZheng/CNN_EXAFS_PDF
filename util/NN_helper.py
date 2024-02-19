import torch
from lightning.pytorch.tuner.tuning import Tuner
import lightning.pytorch as pl

def load_from_checkpoint(model,path):
    checkpoint=torch.load(path)
    loaded_dict = checkpoint['state_dict']
    # n_clip = len(prefix)
    adapted_dict = {k: v for k, v in loaded_dict.items()}
    model.load_state_dict(adapted_dict)  

def learning_rate_tunner(model,train_loader,val_loader, method='fit',min_lr=1e-5,max_lr=1,num_training=1000,mode='exponential'):
    trainer = pl.Trainer()
    tuner=Tuner(trainer)
    lr_finder=tuner.lr_find(model,train_dataloaders=train_loader,val_dataloaders=val_loader,method=method,min_lr=min_lr, max_lr=max_lr,num_training=num_training,mode=mode)
    print(lr_finder.results)
    fig=lr_finder.plot(suggest=True)
    fig.show()
