import argparse
import os

def run_upscaling(img_path):
    cmd = f'python upscaler.py -f {img_path}'
    print(cmd)
    os.system(cmd)

def run_object_detection(img_path):
    cmd = f'python detection.py -f {img_path}'
    print(cmd)
    os.system(cmd)

def run_object_classification(img_path):
    cmd = f'python classification.py -f {img_path}'
    print(cmd)
    os.system(cmd) 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Scenario Runner')
    parser.add_argument('-f', '--filename', required=True)
    args = parser.parse_args()

    img_path = args.filename
    run_upscaling(img_path)
    run_object_detection("upscaled.png")
    run_object_classification("detected_truck_0.png")