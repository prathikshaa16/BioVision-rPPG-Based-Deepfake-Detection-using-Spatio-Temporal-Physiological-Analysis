"""Verify a Colab cached-feature checkpoint before backend integration."""

import argparse

import torch

from backend.app.multimodal_model import build_cached_biovision_visual_rppg_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint')
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_cached_biovision_visual_rppg_model().to(device)
    model.load_state_dict(state.get('model_state_dict', state.get('model', state)), strict=True)
    model.eval()
    with torch.inference_mode():
        output = model(torch.randn(1, 32, 1792, device=device), torch.randn(1, 240, device=device))
    print('checkpoint compatible')
    print('output_shape=', tuple(output.shape))
    print('epoch=', state.get('epoch', 'unknown'))
    print('best_val_auc=', state.get('best_val_auc', 'unknown'))


if __name__ == '__main__':
    main()