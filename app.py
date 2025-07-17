import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn

# === Config ===
model_path = "heart_disease_model.pth"
img_size = 224
class_names = ["Heart Disease", "No Heart Disease"]


# === Load Model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# === Image Transform ===
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# === Streamlit UI ===
st.title("🩺 Retinal Image Heart Disease Predictor")
st.markdown("Upload a retinal image to check heart disease risk using a trained deep learning model.")

uploaded_file = st.file_uploader("Choose a retinal image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)


    if st.button("Predict"):
        with st.spinner("Analyzing with model..."):
            # Preprocess image
            input_tensor = transform(image).unsqueeze(0).to(device)

            # Predict
            with torch.no_grad():
                outputs = model(input_tensor)
                probs = torch.softmax(outputs, dim=1)
                pred = torch.argmax(probs, dim=1).item()
                confidence = probs[0][pred].item() * 100

            label = class_names[pred]
        st.success(f"Prediction: **{label}**")
        st.info(f"Confidence: {confidence:.2f}%")
