import os
import torch
from flask import Flask, render_template, request, redirect, url_for
from PIL import Image
import torchvision.transforms as transforms
from model import PhysicalNN  # Import the updated model

# Initialize the Flask app
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ENHANCED_FOLDER'] = 'static/enhanced/'

# Ensure the uploads folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model and weights
model = PhysicalNN().to(device)
model.load_state_dict(torch.load('model.pth', map_location=device))
model.eval()  # Set to evaluation mode

# Image transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Match training normalization
])
to_image = transforms.ToPILImage()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        # Save the uploaded image
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Load and preprocess the input image
        input_image = Image.open(filepath).convert('RGB')
        input_tensor = transform(input_image).unsqueeze(0).to(device)  # Add batch dimension


        # Perform inference
        with torch.no_grad():
            output_tensor = model(input_tensor)

        # Post-process the output image
        output_tensor = (output_tensor.squeeze(0).cpu() * 0.5 + 0.5).clamp(0, 1)  # Rescale from [-1, 1] to [0, 1]
        output_image = to_image(output_tensor)

        # Save the enhanced image
        enhanced_image_path = os.path.join(app.config['ENHANCED_FOLDER'], 'enhanced_' + file.filename)
        output_image.save(enhanced_image_path)

        return render_template('result.html', input_image=file.filename, enhanced_image='enhanced_' + file.filename)

if __name__ == '__main__':
    app.run(debug=True)
