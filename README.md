# Salient Object Detection Using CNN

## Project Overview

This project implements a Salient Object Detection system from scratch using Python and PyTorch.

Salient Object Detection focuses on identifying the most visually important object or region in an image. The model takes an RGB image as input and outputs a binary saliency mask that highlights the dominant object.

The project includes the full machine learning pipeline:

- Dataset loading and preprocessing
- Image and mask resizing
- Normalization
- Train, validation, and test split
- CNN encoder-decoder model
- Custom loss function
- Training and validation loop
- Evaluation metrics
- Visualizations
- Demo application

## Dataset

The ECSSD dataset was used for this project.

The dataset contains natural images and their corresponding ground-truth saliency masks.

Images and masks were resized to 128x128 pixels. Image pixel values were normalized to the range 0-1. Masks were converted into binary format.

The dataset was split as follows:

- Training set: 70%
- Validation set: 15%
- Test set: 15%

Basic augmentation was applied during training, including horizontal flipping and brightness variation.

## Model Architecture

Two CNN models were implemented.

### Baseline CNN

The baseline model is a simple encoder-decoder CNN.

Encoder:

- Conv2D
- ReLU
- MaxPooling

Decoder:

- ConvTranspose2D
- ReLU
- Sigmoid output layer

The output is a one-channel saliency mask with the same height and width as the input image.

### Improved CNN

The improved model uses the same encoder-decoder structure but adds:

- Batch Normalization
- Dropout

These changes were added to improve training stability and reduce overfitting.

## Loss Function

The model was trained using a custom loss function:

Binary Cross-Entropy + 0.5 * (1 - IoU)

This combines pixel-level classification loss with region-overlap quality.

## Evaluation Metrics

The model was evaluated using:

- Intersection over Union
- Precision
- Recall
- F1-score
- Inference time per image

## Results

| Model | IoU | Precision | Recall | F1-score | Inference Time |
|---|---:|---:|---:|---:|---:|
| Baseline CNN | 0.3632 | 0.6096 | 0.4779 | 0.5280 | 0.0015 sec |
| Improved CNN | 0.3891 | 0.6012 | 0.5344 | 0.5558 | 0.0016 sec |

The improved CNN achieved better IoU, Recall, and F1-score than the baseline model. Precision decreased slightly, which means the improved model detected more salient regions but also included slightly more background pixels.

Overall, the improved model performed better.

## Project Files


    app.py
    README.md
    data/
        ecssd/
    src/
        data_loader.py
        sod_model.py
        sod_model_improved.py
        train.py
        train_improved.py
        evaluate.py
        evaluate_improved.py
        utils.py
    outputs/
        checkpoints/
        visualizations/
        visualizations_improved/
        model_comparison.csv

## How to Run

Train baseline model:

python src/train.py

Evaluate baseline model:

python src/evaluate.py

Train improved model:

python src/train_improved.py

Evaluate improved model:

python src/evaluate_improved.py

Run demo app:

python app.py

## Demo

The demo application allows the user to upload an image and returns:

- Input image
- Predicted saliency mask
- Overlay visualization
- Inference time

## Running the Project in Google Colab

### This project can be tested in Google Colab without setting up everything locally.

- !git clone https://github.com/yllkanuredini/sod_project.git
- %cd sod_project
- !pip install torch torchvision opencv-python matplotlib numpy scikit-learn tqdm gradio
- !python src/evaluate_improved.py

### To launch the demo app:
!python app.py

## Conclusion

This project successfully implemented a complete Salient Object Detection pipeline from scratch. The baseline CNN learned to identify general salient object regions, while the improved CNN achieved better overall performance after adding Batch Normalization and Dropout.

The project demonstrates the full deep learning workflow, including preprocessing, model design, training, evaluation, visualization, experimentation, and deployment through a simple demo.
