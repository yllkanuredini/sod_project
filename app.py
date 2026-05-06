import sys
import time
import torch
import gradio as gr
import numpy as np
import cv2

sys.path.append("sod_project/src")

from sod_model_improved import ImprovedSODModel


MODEL_PATH = "sod_project/outputs/checkpoints/best_improved_sod_model.pth"
IMAGE_SIZE = 128


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ImprovedSODModel().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()


def preprocess_image(input_image):
    original_image = input_image.copy()

    resized_image = cv2.resize(input_image, (IMAGE_SIZE, IMAGE_SIZE))
    normalized_image = resized_image.astype(np.float32) / 255.0

    tensor_image = np.transpose(normalized_image, (2, 0, 1))
    tensor_image = np.expand_dims(tensor_image, axis=0)
    tensor_image = torch.tensor(tensor_image, dtype=torch.float32).to(device)

    return original_image, resized_image, tensor_image


def create_overlay(image, mask):
    red_overlay = np.zeros_like(image)
    red_overlay[:, :, 0] = mask * 255

    overlay = 0.7 * image + 0.3 * red_overlay
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    return overlay


def predict_saliency(input_image):
    original_image, resized_image, tensor_image = preprocess_image(input_image)

    start_time = time.time()

    with torch.no_grad():
        prediction = model(tensor_image)

    end_time = time.time()

    inference_time = end_time - start_time

    prediction = prediction.squeeze().cpu().numpy()
    binary_mask = (prediction > 0.5).astype(np.uint8)

    mask_display = binary_mask * 255

    overlay = create_overlay(resized_image, binary_mask)

    time_text = f"Inference time: {inference_time:.4f} seconds"

    return resized_image, mask_display, overlay, time_text


demo = gr.Interface(
    fn=predict_saliency,
    inputs=gr.Image(type="numpy", label="Upload Input Image"),
    outputs=[
        gr.Image(type="numpy", label="Input Image"),
        gr.Image(type="numpy", label="Predicted Saliency Mask"),
        gr.Image(type="numpy", label="Overlay"),
        gr.Textbox(label="Inference Time")
    ],
    title="Salient Object Detection Demo",
    description="Upload an image and the trained CNN model will predict the salient object mask."
)


if __name__ == "__main__":
    demo.launch(share=True)