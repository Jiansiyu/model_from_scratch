import os
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from model import build_model
from utils import load_config, get_dataloader, set_seed
from datetime import datetime


def train_one_epoch(model, loader, criterion, optimier, device):
    model.train()

    running_loss = 0.0

    for inputs, target in tqdm(loader, desc='Training'):
        inputs, target = inputs.to(device), target.to(device)

        optimier.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, target)
        loss.backward()
        optimier.step()

        running_loss += loss.item()*inputs.size(0)
    return running_loss / len(loader.dataset)


def validate(model, loader, criterion, device):
    model.eval()

    correct, total, val_loss = 0, 0, 0.0
    with torch.no_grad():
        for inputs, target in tqdm(loader, desc='Validating'):
            inputs, target = inputs.to(device), target.to(device)

            outputs = model(inputs)

            loss = criterion(outputs, target)
            val_loss += loss.item()*inputs.size(0)

            _, predicted = outputs.max(1)
            correct += predicted.eq(target).sum().item()
            total += target.size(0)

    return val_loss / len(loader.dataset), correct / total


def main():
    cfg = load_config(os.getenv('CONFIG_PATH', "config.yaml"))
    set_seed(cfg['seed'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, test_loader = get_dataloader(cfg)
    model = build_model(cfg['model']['name'], cfg['model']['num_classes'], cfg['model']['pretrained']).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=cfg['train']['lr'], momentum=cfg['train']['momentum'])

    best_acc = 0.0
    os.makedirs(cfg['train']['checkpoint_path'], exist_ok=True)

    for epoch in range(cfg['train']['epochs']):

        start_time = datetime.now()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, test_loader, criterion, device)
        print(f'Epoch {epoch+1} - Train loss: {train_loss:.4f} - Val loss: {val_loss:.4f} - Val acc: {val_acc:.4f} - Time: {datetime.now() - start_time}s')

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(cfg['train']['checkpoint_path'], f'{cfg["model"]["name"].lower()}_best_model.pth'))
    print(f'Best val acc: {best_acc:.4f}')

    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f'Test loss: {test_loss:.4f} - Test acc: {test_acc:.4f}')

if __name__ == '__main__':
    main()