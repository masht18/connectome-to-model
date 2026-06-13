"""
Config-driven training entry point.

Lets a non-coder run a new connectome end-to-end without editing Python: point it at a
YAML config that names the connectome CSV, the input/output areas, and the usual
hyperparameters. Currently supports MNIST digit classification (the documented quick-start
task); add new datasets in `build_dataloaders`.

Usage:
    connectome-train --config configs/mnist.yaml
    # or, without installing the console script:
    python -m connectome_to_model.train --config configs/mnist.yaml
"""

import os
import pickle
import argparse

import torch
import yaml
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as T

from connectome_to_model.model.graph import Graph
from connectome_to_model.model.architectures import ConnectomicsConvGRU
from connectome_to_model.model.readouts import ClassifierReadout


# Defaults applied for any key the user omits from the YAML config.
DEFAULT_CONFIG = {
    'dataset': 'mnist',
    'data_path': './data',
    'input_nodes': [0],
    'output_nodes': [2],
    'epochs': 10,
    'batch_size': 32,
    'lr': 0.001,
    'seed': 42,
    'topdown': True,
    'dropout': True,
    'dropout_p': 0.25,
    'proj_hidden_dim': 32,
    'n_classes': 10,
    'num_workers': 4,
    'image_size': 32,
    'model_save': None,
    'results_save': None,
}


def load_config(path):
    with open(path, 'r') as f:
        user_cfg = yaml.safe_load(f) or {}
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(user_cfg)
    if 'graph_loc' not in cfg or not cfg['graph_loc']:
        raise ValueError("Config must set 'graph_loc' (path to the connectome CSV).")
    return cfg


def build_dataloaders(cfg):
    """Return (trainloader, testloader). Extend here to add datasets beyond MNIST."""
    if cfg['dataset'].lower() != 'mnist':
        raise ValueError(
            f"Unsupported dataset '{cfg['dataset']}'. Only 'mnist' is wired up in train.py; "
            f"add a branch in build_dataloaders() for others."
        )
    transform = T.Compose([T.Resize((cfg['image_size'], cfg['image_size'])), T.ToTensor()])
    train_data = datasets.MNIST(root=cfg['data_path'], download=True, train=True, transform=transform)
    test_data = datasets.MNIST(root=cfg['data_path'], download=True, train=False, transform=transform)
    trainloader = DataLoader(train_data, batch_size=cfg['batch_size'], shuffle=True,
                             num_workers=cfg['num_workers'])
    testloader = DataLoader(test_data, batch_size=cfg['batch_size'], shuffle=False,
                            num_workers=cfg['num_workers'])
    return trainloader, testloader


@torch.no_grad()
def evaluate(model, readout, dataloader, device):
    correct = total = 0
    for imgs, label in dataloader:
        imgs = torch.unsqueeze(imgs, 1).to(device)
        label = label.to(device)
        output = readout(model([imgs]))
        _, predicted = torch.max(output.data, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()
    return correct / total


def train_one_epoch(model, readout, dataloader, optimizer, criterion, device):
    running_loss = 0.0
    for imgs, label in dataloader:
        optimizer.zero_grad()
        imgs = torch.unsqueeze(imgs, 1).to(device)
        label = label.to(device)
        output = readout(model([imgs]))
        loss = criterion(output, label)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
    return running_loss


def run(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg['seed'])
    print(f"Using device: {device}")

    trainloader, testloader = build_dataloaders(cfg)

    graph = Graph(cfg['graph_loc'],
                  input_nodes=cfg['input_nodes'],
                  output_nodes=cfg['output_nodes'])
    input_sizes = graph.find_input_sizes()
    input_dims = graph.find_input_dims()

    model = ConnectomicsConvGRU(
        graph, input_sizes, input_dims,
        topdown=cfg['topdown'],
        dropout=cfg['dropout'], dropout_p=cfg['dropout_p'],
        proj_hidden_dim=cfg['proj_hidden_dim'],
    ).to(device).float()
    readout = ClassifierReadout(model.output_sizes[0], n_classes=cfg['n_classes']).to(device).float()

    if cfg['model_save'] and os.path.exists(cfg['model_save']):
        model.load_state_dict(torch.load(cfg['model_save'], map_location=device))
        print(f"Loaded existing model weights from {cfg['model_save']}")

    params = [{'params': model.parameters(), 'lr': cfg['lr']},
              {'params': readout.parameters(), 'lr': cfg['lr']}]
    optimizer = optim.Adam(params)
    criterion = nn.CrossEntropyLoss()

    history = {'loss': [], 'test_acc': []}
    for epoch in range(cfg['epochs']):
        loss = train_one_epoch(model, readout, trainloader, optimizer, criterion, device)
        test_acc = evaluate(model, readout, testloader, device)
        print(f"| epoch {epoch:3d} | loss {loss:8.4f} | test acc {test_acc:.4f} |")
        history['loss'].append(loss)
        history['test_acc'].append(test_acc)

        if cfg['results_save']:
            with open(cfg['results_save'], "wb") as f:
                pickle.dump(history, f)
        if cfg['model_save']:
            torch.save(model.state_dict(), cfg['model_save'])

    return history


def main():
    parser = argparse.ArgumentParser(description="Train a connectome-defined ConvGRU from a YAML config.")
    parser.add_argument('--config', required=True, help='Path to a YAML config file.')
    args = parser.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
