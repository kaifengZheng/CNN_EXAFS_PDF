import torch
from torch import nn
from torch.optim import Adam, SGD
import lightning.pytorch as pl
from torchmetrics import MeanMetric,MinMetric,SumMetric

def loss_percentage(y_hat,y):
    l1=torch.abs(y_hat-y)
    maxl1=torch.max(l1)
    minl1=torch.min(l1)
    return torch.mean(l1[torch.where(l1>(minl1+0.3*(maxl1-minl1)))[0]])

class NN_model(pl.LightningModule):
    def __init__(self,net:torch.nn.Module,loss_func,lr=1e-1,scheduler=False):
        super().__init__()
        self.net=net
        self.loss=loss_func
        self.train_loss=MeanMetric()
        self.val_loss=MeanMetric()
        self.val_loss_best=MinMetric()
        self.scheduler=scheduler
        self.lr=lr
    #     self.apply(self._init_weights)
    # def _init_weights(self, module):
    #     if isinstance(module, nn.Linear):
    #         module.weight.data.normal_(mean=0.0, std=1.0)
    #         if module.bias is not None:
    #             module.bias.data.zero_()

    def forward(self,x):
        return self.net(x)
    def on_train_start(self):
        self.val_loss.reset()
        self.val_loss_best.reset()
    def model_step(self,batch):
        x,y=batch
        y_hat=self(x)
        
        loss=self.loss(y_hat,y)
        return loss,y,y_hat
        
    def training_step(self,batch,batch_idx):
        loss,y,y_hat=self.model_step(batch)

        self.train_loss(loss)
        self.log('train_loss', self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    def on_train_epoch_end(self)->None:
        pass
    def validation_step(self,batch,batch_idx):
        
        loss,y,y_hat=self.model_step(batch)
        self.val_loss(loss)
        self.log('val_loss', self.val_loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    def on_validation_epoch_end(self)->None:
        loss=self.val_loss.compute()
        self.val_loss_best(loss)
        self.log('val_loss_best', self.val_loss_best, on_step=False, on_epoch=True)
    def configure_optimizers(self):
        optimizer = SGD(self.parameters(), momentum=0.9,lr=self.lr,nesterov=True,weight_decay=1e-4)#,momentum=0.7,nesterov=True
        # optimizer = Adam(self.parameters(), lr=self.lr)
        if self.scheduler==True:
            lr_scheduler = {
            # 'scheduler': torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-5, max_lr=self.lr,step_size_up=5,step_size_down=50,mode="exp_range",gamma=0.99, cycle_momentum=False),
            'scheduler': torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.lr,total_steps=500, pct_start=1e-8,anneal_strategy='cos',three_phase=False,div_factor=20,final_div_factor=1e3, last_epoch=-1,verbose=False),
            #'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5),
            'name': 'my_cyclic_scheduler'
            }
            return [optimizer],[lr_scheduler]#, [lr_scheduler]
        else:
            return [optimizer]
    
