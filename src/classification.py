from PIL import Image
import torch
from torchvision.models import vgg11, VGG11_Weights
import argparse

DEVICE = 'cuda'
torch.set_default_device(DEVICE)

MODEL = vgg11(weights=VGG11_Weights.DEFAULT).to(DEVICE)
MODEL.eval()
IMG_SIZE = ((224, 224))

SOFTMAX = torch.nn.Softmax(dim=1)
TRANSFORM = VGG11_Weights.DEFAULT.transforms(antialias=True)

def get_image_tensor_from(image_path:str) -> torch.Tensor:
    with Image.open(image_path) as image:
        img_tensor = TRANSFORM(image.resize(IMG_SIZE)).unsqueeze(0)
    return img_tensor

def get_model_predictions_for(image_tensor:torch.Tensor) -> torch.Tensor:
    output = SOFTMAX(MODEL(image_tensor.to(DEVICE).clip(-3, 3)))
    return output

def get_viz_for(image_tensor:torch.Tensor, save_to:str='visualization.png', show_k=10, max_len=20):
    image_tensor = image_tensor.clip(-3, 3)
    assert image_tensor.size(0) == 1

    probabilities = get_model_predictions_for(image_tensor)
    topk_results = torch.topk(probabilities[0], k=show_k)
    print(topk_results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Object Classification')
    parser.add_argument('-f', '--filename', required=True)
    args = parser.parse_args()

    image = get_image_tensor_from(args.filename)
    probabilities = get_model_predictions_for(image)
    topk_results = torch.topk(probabilities[0], k=10)
    print(topk_results)