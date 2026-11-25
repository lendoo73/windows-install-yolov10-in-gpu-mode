# [22-133. Installation: Install YOLOv10 in GPU Mode](https://www.udemy.com/course/yolo-masterclass-deep-learning-computer-vision-course/)

## Installation

[Clone YOLOv10 repository](https://github.com/THU-MIG/yolov10)

## Install the required dependencies

```shell
pip install -r requirements.txt
pip install -e .  # Install the project in editable (development) mode.
```

## [Install PyThorch CUDA version](https://pytorch.org/get-started/previous-versions/)

```shell
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
```

## Test the installation

```shell
python app.py
```

Copy the URL then open in a browser.

## Check if GPU used

While the app is running, open another terminal and run the following command to check if GPU is being used:

```shell
nvidia-smi -l 1
```

On the gradio UI click on `Detect Objects` and you should see the GPU usage in the terminal.