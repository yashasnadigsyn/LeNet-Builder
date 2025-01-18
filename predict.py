import torch
from torch import nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import io

class LeNet5(nn.Module):
    def __init__(self, num_classes):
        super(LeNet5, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Linear(400, 120)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(84, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out

def load_model(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LeNet5(num_classes=10).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, device

def preprocess_canvas_image(canvas_image_data):
    """
    Preprocess the canvas image data for inference.
    Args:
        canvas_image_data: The image data from the canvas (numpy array).
    Returns:
        Preprocessed image tensor.
    """

    image = Image.fromarray(canvas_image_data.astype('uint8'), 'RGBA')


    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    ])

    image = transform(image).unsqueeze(0)
    return image

def predict(model, image, device):
    """
    Predict the digit from the preprocessed image tensor.
    Args:
        model: The trained LeNet5 model.
        image: Preprocessed image tensor.
        device: The device (CPU or GPU) to run the model on.
    Returns:
        predicted_digit: The predicted digit (0-9).
    """
    with torch.no_grad():
        image = image.to(device)
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)
        return predicted.item()

def main(canvas_image_data):
    """
    Main function to run inference on the canvas image.
    Args:
        canvas_image_data: The image data from the canvas (numpy array).
    """

    model_path = 'lenet_model.pth'
    model, device = load_model(model_path)

    processed_image = preprocess_canvas_image(canvas_image_data)

    prediction = predict(model, processed_image, device)
    return prediction
