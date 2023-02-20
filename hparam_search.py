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
#from ambiguous import SequenceDataset

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# Seed for reproducibility
torch.manual_seed(42)

class NeuralGraphModule(pl.LightningModule):
    def __init__(self, graph, trainset, valset,
                 lr, weight_decay, 
                 dropout_p, logger=WandbLogger(), rep=1,
                 criterion=torch.nn.CrossEntropyLoss()):
        super().__init__()
        #self.hparams.update(vars(hparams))
        self.lr = lr
        self.weight_decay = weight_decay
        self.criterion = criterion
        #self.batch_sz = batch_sz
        self.transform = transforms.Compose([transforms.ToTensor(), 
                                             transforms.Normalize((0.5,), (0.5,))])
        
        self.trainloader = trainset
        self.valloader = valset
        #self.logger = logger
        
        input_dims = [1, 0, 0, 0]
        input_sizes = [(28, 28), (0, 0), (0, 0), (0, 0)]
        self.model = Architecture(graph, input_sizes, input_dims, dropout_p=dropout_p, rep=rep).cuda()
        #self.device = device

    def forward(self, x):
        # turn input into a list of image sequences (b,t,c,h,w) if it's not already
        #print(x)
        x = [x] if not isinstance(x, list) else x
        x = [torch.unsqueeze(inp, 1) for inp in x if inp.ndim != 5]
        #x = [torch.unsqueeze(inp, 1) for inp in x]
        
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch      
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log('val_loss', loss)
        self.log('val_acc', acc)
        return {'val_loss': loss, 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        self.log('val_loss', avg_loss)
        self.log('val_acc', avg_acc)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
    def train_dataloader(self):
        #train_loader = DataLoader(self.trainset, batch_size=self.batch_sz, shuffle=True)
        return self.trainloader

    def val_dataloader(self):
        #val_loader = DataLoader(self.valset, batch_size=self.batch_sz, shuffle=False)
        return self.valloader
    
    #def on_train_epoch_end(self):
    #    metrics = {
    #        'train_loss': self.trainer.callback_metrics['train_loss'],
    #        'val_loss': self.trainer.callback_metrics['val_loss'],
    #        'val_acc': self.trainer.callback_metrics['val_acc']
    #    }
    #    wandb.log(metrics)

def objective(trial):
    # Logger
    logger = WandbLogger(project="graph_to_model", entity=hparams.entity)
    
    # Load graph
    graph_loc = '/home/mila/m/mashbayar.tugsbayar/convgru_feedback/sample_graph.csv'
    graph = Graph(graph_loc, input_nodes=[0], output_node=3)
    
    # Load dataset
    MNIST_path='/home/mila/m/mashbayar.tugsbayar/datasets'
    dataset = datasets.MNIST(root=MNIST_path, download=True, train=True, transform=transforms.ToTensor())
    trainset, valset = tdata.random_split(dataset, [int(len(dataset)*0.8), len(dataset)-int(len(dataset)*0.8)])
    train_loader = tdata.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=1)
    val_loader = tdata.DataLoader(valset, batch_size=32, shuffle=False, num_workers=1)
    
    lr = trial.suggest_loguniform('lr', 1e-6, 1e-2)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-2)
    #batch_sz = trial.suggest_int('batch_sz', 32, 256)
    rep = trial.suggest_int('rep', 1, 3)
    dropout_p = trial.suggest_float('dropout_p', 0, 0.5)
    
    model = NeuralGraphModule(graph, train_loader, val_loader, lr, weight_decay, dropout_p, logger=logger).to(device)
    logger.watch(model)
    early_stopping = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=10, verbose=False, mode="max")
    trainer = pl.Trainer(callbacks=[early_stopping], max_epochs=hparams.epochs, gpus=1)
    trainer.fit(model, train_loader, val_loader)
    
    return trainer.callback_metrics['val_loss'].item()
    
def main(hparams):
    # Run training
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100)

    print("Best trial:")
    best_trial = study.best_trial
    print("  Value: ", best_trial.value)
    print("  Params: ")
    for key, value in best_trial.params.items():
        print("    {}: {}".format(key, value))
        
    testset = datasets.MNIST(root=MNIST_path, download=True, train=False, transform=T.ToTensor())
    testloader = tdata.DataLoader(testset, batch_size=32, shuffle=True, num_workers=32)
    trainer.test(model, testset)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--dropout_p', type=float, default=0.1)
    parser.add_argument('--entity', type=str, default='tmshbr')
    hparams = parser.parse_args()

    main(hparams)
