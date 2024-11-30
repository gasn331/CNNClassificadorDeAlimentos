from src.data_preprocessing import create_data_generators
from src.model import create_model, load_model
from src.train import train_model
from src.evaluate import evaluate_model


def main():
    # Caminhos para o dataset
    train_dir = 'data/train'
    test_dir = 'data/test'
    save_path = 'models/trained_model.pth'

    print("Inicio do pré-processamento dos dados")
    # Preparar os dados
    train_loader, test_loader = create_data_generators(train_dir, test_dir)
    print("Fim do pré-processamento dos dados")
    # Criar o modelo
    model = create_model()

    # Treinar o modelo
    model = train_model(model, train_loader, test_loader)

    # Avaliar o modelo
    model = load_model(model, save_path)
    evaluate_model(model, test_loader)


if __name__ == "__main__":
    main()
