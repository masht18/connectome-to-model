import argparse

import torch
import torch.utils.data as tdata
from torchvision import datasets, transforms

import optuna
import wandb
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from model.graph import Graph, Architecture
from utils.audio_dataset import AudioVisualDataset

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# Seed for reproducibility
torch.manual_seed(42)

class NeuralGraphModule(pl.LightningModule):
    def __init__(self, graph, train_loader, val_loader, test_loader,
                 lr, weight_decay, 
                 logger=WandbLogger(), rep=1,
                 criterion=torch.nn.CrossEntropyLoss()):
        super().__init__()
        #self.hparams.update(vars(hparams))
        self.lr = lr
        self.weight_decay = weight_decay
        self.criterion = criterion
        #self.batch_sz = batch_sz
        self.transform = transforms.Compose([transforms.ToTensor(), 
                                             transforms.Normalize((0.5,), (0.5,))])
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = train_loader
        
        input_dims = [1, 0, 1]
        input_sizes = [(32, 32), (0, 0), (32, 32)]
        self.model = Architecture(graph, input_sizes, input_dims, rep=rep).cuda().float()
        #self.device = device

    def forward(self, x):
        # turn input into a list of image sequences (b,t,c,h,w) if it's not already
        #print(x)
        x = [x] if not isinstance(x, list) else x
        x = [torch.unsqueeze(inp, 1).float() for inp in x if inp.ndim != 5]
        #x = [torch.unsqueeze(inp, 1) for inp in x]
        
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss)
        
        if batch_idx % hparams.log_step == 0:
            self.test_step(batch, batch_idx)
            
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch      
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss)
            
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        
        if batch_idx < len(self.test_loader)/3:
            self.log('amb_match_test_acc', acc)
        elif batch_idx > len(self.test_loader)/3 and batch_idx < (len(self.test_loader)/3)*2:
            self.log('clean_mismatch_test_acc', acc)
        else:
            self.log('amb_mismatch_test_acc', acc)
            
        return acc

    #def validation_epoch_end(self, outputs):
    #    avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
    #    avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
    #    self.log('val_loss', avg_loss)
    #    self.log('val_acc', avg_acc)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader
    
    def test_dataloader(self):
        return self.test_loader
    
def load_mixed_dataset(split, *root):
    datasets = [AudioVisualDataset(None, None, cache_dir=cache_dir, split=split) for cache_dir in root]
    combined = tdata.ConcatDataset(datasets)
    return combined

def objective(trial):
    # Logger
    logger = WandbLogger(project=hparams.trial_name, entity=hparams.entity)
    
    # Load graph
    graph = Graph(hparams.graph_loc, input_nodes=[0, 2], output_node=1)
    
    # Load datasets
    amb_match_root='/home/mila/m/mashbayar.tugsbayar/datasets/ambvisual_multimodal'
    clean_mismatch_root='/home/mila/m/mashbayar.tugsbayar/datasets/multimodal_clean_mismatch'
    amb_mismatch_root='/home/mila/m/mashbayar.tugsbayar/datasets/multimodal_amb_mismatch'
    
    trainset = load_mixed_dataset('train', amb_match_root, clean_mismatch_root)
    valset = load_mixed_dataset('val', amb_match_root, clean_mismatch_root)
    testset = load_mixed_dataset('test', amb_match_root, clean_mismatch_root, amb_mismatch_root)
    
    train_loader = tdata.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = tdata.DataLoader(valset, batch_size=32, shuffle=False, num_workers=2)
    test_loader = tdata.DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)
    
    # Hyperparams
    lr = trial.suggest_loguniform('lr', 1e-4, 1e-2)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-2)
    batch_sz = trial.suggest_int('batch_sz', 16, 128)
    #rep = trial.suggest_int('rep', 1, 3)
    #dropout_p = trial.suggest_float('dropout_p', 0, 0.5)
    
    model = NeuralGraphModule(graph, train_loader, val_loader, test_loader, lr, weight_decay, logger=logger).to(device)
    logger.watch(model)
    early_stopping = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=3, verbose=False, mode="max")
    trainer = pl.Trainer(callbacks=[early_stopping], max_epochs=hparams.epochs, gpus=1, 
                         logger=logger, log_every_n_steps=hparams.log_step)
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)
    #wandb.finish()
    
    return trainer.callback_metrics['val_loss'].item()
    
def main(hparams):
    # Run training
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=hparams.num_trials)

    print("Best trial:")
    best_trial = study.best_trial
    print("  Value: ", best_trial.value)
    print("  Params: ")
    for key, value in best_trial.params.items():
        print("    {}: {}".format(key, value))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_loc', type = str, default ='/home/mila/m/mashbayar.tugsbayar/convgru_feedback/graphs/test/topdown_test_mult_only.csv')
    parser.add_argument('--trial_name', type=str, default='graph_to_model_recovery_mult')
    parser.add_argument('--num_trials', type=int, default=10)
    parser.add_argument('--log_step', type=int, default=500)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--entity', type=str, default='tmshbr')
    hparams = parser.parse_args()

    main(hparams)
