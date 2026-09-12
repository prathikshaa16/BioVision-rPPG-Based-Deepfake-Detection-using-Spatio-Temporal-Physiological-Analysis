import torch
from backend.app.model import build_efficientnet_b4, load_checkpoint_into_model


def main():
    model = build_efficientnet_b4(num_classes=1)
    model.eval()
    model, info = load_checkpoint_into_model(model, 'backend/models/best_model.pt', device='cpu')
    x = torch.rand(1, 3, 224, 224)
    with torch.inference_mode():
        out = torch.sigmoid(model(x)).squeeze().item()
    print('probability', out)
    print('valid_range', 0.0 <= out <= 1.0)
    print('missing', info['missing_keys'])
    print('unexpected', info['unexpected_keys'])


if __name__ == '__main__':
    main()
