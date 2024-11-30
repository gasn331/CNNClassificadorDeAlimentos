import torch


def evaluate_model(model, test_loader):
    """Função para fazer a avaliação do modelo treinado."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct += (torch.Tensor(predicted) == torch.Tensor(labels)).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}')
    return accuracy
