import torch
import torch.optim as optim
import torch.nn as nn
import os
from torch.optim.lr_scheduler import StepLR


def train_model(model, train_loader, test_loader, epochs=50, learning_rate=0.005, save_dir='models'):
    """ Função para treinar o modelo e salvar o melhor modelo"""
    print("Inicio do treinamento")
    # Criar o diretório para salvar o modelo, caso não exista
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Diretório {save_dir} criado!")

    print("Criando criterion...")
    criterion = nn.CrossEntropyLoss()
    print("Criando optimizer...")
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Device: {device}")

    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    print("Inicio do loop de epochs...")
    for epoch in range(epochs):
        print("Chamando model.train()...")
        model.train()
        print("Fim de model.train()...")
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}")

        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (torch.Tensor(predicted) == torch.Tensor(labels)).sum().item()

        print(f"Validation Loss: {val_loss / len(test_loader)}, Accuracy: {100 * correct / len(test_loader.dataset):.2f}")
        scheduler.step()

    # Salvar o modelo
    save_path = save_dir + '/trained_model.pth'
    torch.save(model.state_dict(), save_path)
    print(f"Modelo salvo em: {save_path}")

    print("Fim do treinamento")
    return model

