from flask import Flask, request, jsonify
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io

app = Flask(__name__)

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the model
class CustomCNNModel(nn.Module):
    def __init__(self, num_classes=26):
        super(CustomCNNModel, self).__init__()
        self.model = models.resnet50(pretrained=True)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# Instantiate the model
num_classes = 25
model = CustomCNNModel(num_classes=num_classes).to(device)

# Load the model checkpoint
def load_model(filepath='./trained_model.pth'):
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    print(f"Model loaded from {filepath}")

load_model()

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dic = {0: 'Aerosal Cans', 1: 'Aluminium Soda Cans', 2: 'Cardboard Boxes', 3: 'Disposable Plastic Cutlery', 
       4: 'Eggshells', 5: 'Glass Bottles', 6: 'Glass Cosmetic Containers', 7: 'Hand Bag', 8: 'Hoodies', 9: 'Jacket',
       10: 'Magazines', 11: 'Newspaper', 12: 'Office Paper', 13: 'Paper Cups', 14: 'Plastic Bottles', 15: 'Plastic Detergent Bottles', 
       16: 'Plastic Food Containers', 17: 'Plastic Shopping Bags', 18: 'Plastic Straws', 19: 'Polo Shirt',
       20: 'Shirt', 21: 'Shoes', 22: 'Styrofoam Food Containers', 23: 'T-shirt', 24:'Tank Top'}

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if file and allowed_file(file.filename):
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        img = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img)
            _, predicted = torch.max(outputs, 1)
            prediction = predicted.item()

        return jsonify({"prediction": dic[prediction]})
    else:
        return jsonify({"error": "Invalid file format"}), 400

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

if __name__ == '__main__':
    app.run(debug=False, port=5000)
