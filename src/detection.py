from torchvision.io.image import read_image
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
import argparse


def initialize_model():
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.9)
    model.eval()
    return weights, model

def visualize(weights, model, img):
    preprocess = weights.transforms()
    batch = [preprocess(img)]
    prediction = model(batch)[0]
    labels = [weights.meta["categories"][i] for i in prediction["labels"]]
    box = draw_bounding_boxes(img, boxes=prediction["boxes"],
                            labels=labels,
                            colors="yellow",
                            width=6, font_size=40)
    im = to_pil_image(box.detach())
    return im, prediction, labels

def crop_trucks(img, prediction, labels):
    pil_img = to_pil_image(img)
    for i, label in enumerate(labels):
        if label == 'truck':
            box = prediction["boxes"][i]
            xmin, ymin, xmax, ymax = float(box[0]), float(box[1]), float(box[2]), float(box[3])
            
            cropped_img = pil_img.crop((xmin, ymin, xmax, ymax))
            cropped_img.save(f'detected_truck_{i}.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Bounding Boxes')
    parser.add_argument('-f', '--filename', required=True)
    args = parser.parse_args()

    image_path = args.filename
    weights, model = initialize_model()

    img = read_image(args.filename)

    im, prediction, labels = visualize(weights, model, img)
    crop_trucks(img, prediction, labels)

    im.save("bounding_box.png")