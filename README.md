# Overview

I've trained a model with PyTorch to differentiate between my two cats, Kvarg and Jarlsberg.

The model can be run on images uploaded via an API. In a separate repo, I have a React frontend to send these images.

## Architecture

These are some ways this might sit in dev or production environments.
![architecture diagram](assets/architecture-diagram.png)

# How to run the API

## If running locally without Docker:

- Create a virtual environment running Python 3.11
- Install dependencies: `pip install -r requirements.txt`
- Run the server: `python app.py`
- It should run on port:80

## To run from Docker

- Get the image from AWS ECR (requires aws cli)
  `aws ecr get-login-password --region eu-west-2 | docker login --username AWS --password-stdin public.ecr.aws/i8x4p3a5`
  `docker pull public.ecr.aws/i8x4p3a5/cat-ai-service:latest`

- Note: there are currently issues with the image size being much larger than it should be.

- Then run the container as normal. For help with Docker see their [documentation](https://docs.docker.com/).

# To train a new model

`build_model.ipynb` is a Python script for training a deep learning model, specifically a ResNet-50 architecture, for image classification using PyTorch. It assumes you have a dataset of images organized into train and test folders. The code also includes a learning rate scheduler and saves the trained model's parameters to a file. To use with your own images:

1. **Import Required Libraries**:

   - This script uses several Python libraries for deep learning and image processing. You need to ensure that you have PyTorch and torchvision installed.

2. **Data Preparation**:

   - The script defines a set of transformations to be applied to your images. In the provided code, it resizes the images to 224x224 pixels and converts them to tensors.
   - You should organize your own image dataset in two folders: 'train' and 'test,' and set the `train_dataset` and `test_dataset` accordingly, specifying the path to your dataset and applying the defined transformations.

3. **Model Setup**:

   - The script uses a pre-trained ResNet-50 model from torchvision, which is commonly used for image classification tasks.
   - It modifies the last fully connected layer (`model.fc`) to have two output units, assuming you want to classify images into two classes. You can adjust this based on your specific classification needs.

4. **Loss and Optimizer**:

   - It uses the Cross-Entropy loss and the Adam optimizer for training. You can modify the optimizer or loss function if needed.

5. **Learning Rate Scheduling**:

   - The script defines a learning rate scheduler that reduces the learning rate by a factor of `gamma` every `step_size` epochs. You can adjust these parameters to fine-tune your training.

6. **Training Loop**:

   - It trains the model for `num_epochs` epochs. During training, it alternates between the training and evaluation modes for the model.
   - It prints out test accuracy and learning rate after each epoch.

7. **Model Saving**:
   - After training, the script saves the model's parameters to a file named 'cat_recognition_model4.pth'. You should change this filename to something meaningful for your use case.

To use this code with your own images and class labels, you need to:

1. Organize your image dataset into 'train' and 'test' folders.
2. Update the `train_dataset` and `test_dataset` paths to point to your dataset.
3. Adjust the number of output units in the model's last fully connected layer (`model.fc`) to match your classification task.
4. Set the loss function, optimizer, learning rate scheduler parameters, and the number of epochs according to your specific needs.
5. Change the model saving filename to something relevant to your project.

Remember to install the required Python libraries (PyTorch, torchvision) and ensure your environment is properly configured before running the code. Additionally, consider using a GPU if you have one available, as deep learning training can be computationally intensive.

## Challenges

- first time training any kind of model myself
- not enough data initially
- learning how to use the model with an API
- model files are large, need to hosted somewhere (like S3)
- torch is a massive library, too big for lambda, even though the code is so simple. Containerizing worked, but even then it's several GBs.
- AWS kicked me out:(
