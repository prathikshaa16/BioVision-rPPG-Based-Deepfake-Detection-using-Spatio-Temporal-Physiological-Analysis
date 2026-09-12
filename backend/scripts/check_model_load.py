import torch
from backend.app.model import build_efficientnet_b4, load_checkpoint_into_model
from backend.app.config import MODEL_PATH


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device:', device)
    model = build_efficientnet_b4(num_classes=1)
    model.to(device)
    print('Built EfficientNet-B4; attempting to load checkpoint at', MODEL_PATH)
    try:
        model, info = load_checkpoint_into_model(model, MODEL_PATH, device=device)
        print('Load succeeded. Info:')
        print('  missing_keys_count:', len(info.get('missing_keys', [])))
        print('  unexpected_keys_count:', len(info.get('unexpected_keys', [])))
        # Print examples (up to 20)
        mk = info.get('missing_keys', [])
        uk = info.get('unexpected_keys', [])
        if mk:
            print('  missing_keys (sample):')
            for k in mk[:20]:
                print('   ', k)
        if uk:
            print('  unexpected_keys (sample):')
            for k in uk[:20]:
                print('   ', k)
    except FileNotFoundError as e:
        print('Checkpoint not found:', e)
    except Exception as e:
        print('Error while loading checkpoint:', repr(e))


if __name__ == '__main__':
    main()
