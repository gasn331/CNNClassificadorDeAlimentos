from PIL import Image
import torch
from torchvision import transforms
from torch import nn
from model import VegetableCNN  # Certifique-se de importar seu modelo

# Dicionário que mapeia os índices das classes para seus nomes
class_names = ['Apple', 'Banana', 'Carrot', 'Corn', 'Cucumber', 'GingerRoot', 'Limes',
               'Onion', 'Pepper', 'Pineapple', 'Potato', 'Strawberry', 'Tomato']


def predict_image(model, img_path, img_size=(100, 100), device='cpu'):
    """Função para fazer a previsão de uma imagem usando o modelo treinado."""

    # Carregar e pré-processar a imagem
    img = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalização padrão para CNNs
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)

    # Colocar o modelo em modo de avaliação e fazer a previsão
    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = nn.functional.softmax(output, dim=1)
        predicted_class_index = torch.argmax(probabilities, dim=1).item()
        predicted_probability = probabilities[0, predicted_class_index].item()

    # Obter o nome da classe prevista
    predicted_class_name = class_names[predicted_class_index]

    return predicted_class_name, predicted_probability


if __name__ == "__main__":
    # Instanciar o modelo com o número correto de classes (ajustado para o seu caso, como 13, por exemplo)
    model = VegetableCNN(num_classes=13)

    # Carregar os pesos do modelo
    model_weights = torch.load('../models/trained_model.pth', map_location='cpu', weights_only=True)
    model.load_state_dict(model_weights)

    # Fazer uma previsão
    class_name, probability = predict_image(model, '../data/data_to_predict/tomato1.jpg')
    print(f'Classe prevista: {class_name} com probabilidade de {probability*100:.2f}% de acerto.')
