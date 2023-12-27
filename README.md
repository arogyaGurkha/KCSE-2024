# KCSE-2024
Replication code for the ambulance detection case scenario.

# Ambulance Detection Scenario
The provided scenario does three things: 
1. It uses an upscaler model to upscale a **93x62** image of an ambulance to **512x512**.
2. The upscaled image is then processed by object detection model to provide bounding box information. Patches inside the bounding boxes are cropped.
3. The classifier processes the cropped image to predict probability scores for ambulance. 

## Models Used
1. Upscaling: [stabilityai/stable-diffusion-x4-upscaler](https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler) (Hugging Face)
2. Bounding Boxes: [fasterrcnn_resnet50_fpn_v2](https://pytorch.org/vision/stable/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn_v2.html) (PyTorch)
3. Final Classification: [vgg11](https://pytorch.org/vision/stable/models/generated/torchvision.models.vgg11.html) (PyTorch)

## Running the scenario
The necessary environment for running the scenario has been pre-built into a docker image. 
1. Clone the repository
2. Run `docker compose up -d` on the root directory.
3. Run `docker exec -it kcse-2024-cuda-service-1 /bin/bash` to enter the container.
4. Enter the `src` directory and run `python run_scenario -f ../assets/boston-ambulance-2.png`. This should download the necessary model checkpoints and run the scenario.
