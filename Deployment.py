import numpy as np
import torch
import streamlit as st
from swin_model import Swin_V2_B_Weights, swin_v2_b 
from torchvision import transforms
from PIL import Image
import torch.nn as nn

np.random.seed(0)
torch.manual_seed(0)
# Set page configuration
st.set_page_config(
    page_title="NeuroVision Animal Classifier",
    page_icon="🐾",
    #layout="wide",
    #initial_sidebar_state="collapsed"
)

index_to_animal = {
    0: "Beaver", 1: "Butterfly", 2: "Cougar", 3: "Crab",
    4: "Crayfish", 5: "Crocodile", 6: "Dolphin", 7: "Dragonfly",
    8: "Elephant", 9: "Flamingo", 10: "Kangaroo", 11: "Leopard",
    12: "Llama", 13: "Lobster", 14: "Octopus", 15: "Pigeon",
    16: "Rhino", 17: "Scorpion"
}

# App Banner
st.image("Cover.png", use_column_width=True)

# Streamlit App Title
st.markdown("""
    <h1 style='text-align: center; color: #bb86fc;'>🐾 Image Classification with SwinV2 Model</h1>
""", unsafe_allow_html=True)

st.markdown("""
    <h3 style='text-align: center; color:rgb(126, 230, 219);'>🔮 The model can predict the following 18 animals:</h3>
""", unsafe_allow_html=True)
st.markdown("<hr style='border: 1px solid #bb86fc;'>", unsafe_allow_html=True)

# Display animals in a card-based layout
animal_emojis = [
    "🦫", "🦋", "🐆", "🦀", "🦞", "🐊", "🐬", "🐝",
    "🐘", "🦩", "🦘", "🐆", "🦙", "🦞", "🐙", "🕊️",
    "🦏", "🦂"
]

# Define CSS for dark-themed cards with fade animation
st.markdown("""
    <style>
    @keyframes fadeColor {
        0% { background-color: #1e1e1e; }
        100% { background-color: #bb86fc; }
    }
    .card {
        background-color: #1e1e1e;
        padding: 15px;
        margin: 10px;
        border-radius: 12px;
        box-shadow: 0 4px 8px rgba(255, 255, 255, 0.1);
        text-align: center;
        font-size: 20px;
        font-weight: bold;
        color: #ffffff;
        transition: transform 0.3s, background-color 0.8s;
    }
    .card:hover {
        transform: scale(1.05);
        animation: fadeColor 0.8s forwards;
    }
    </style>
""", unsafe_allow_html=True)

# Create grid layout for cards
cols = st.columns(3)
for i, (animal, emoji) in enumerate(zip(index_to_animal.values(), animal_emojis)):
    with cols[i % 3]:  # Distribute items into 3 columns
        st.markdown(f'<div class="card">{emoji} {animal}</div>', unsafe_allow_html=True)

st.markdown("<hr style='border: 1px solid #bb86fc;'>", unsafe_allow_html=True)

# Cached function to load model and weights
@st.cache_resource
def load_model():
    weights_file = "swin_epoch_1_97_99%_875.pth"
    
    weights = Swin_V2_B_Weights.IMAGENET1K_V1
    model = swin_v2_b(weights=weights)

    # Update the model head for the desired number of classes
    num_classes = 18
    model.head = nn.Linear(model.head.in_features, num_classes)

    # Load weights into the model
    model.load_state_dict(torch.load(weights_file, map_location=torch.device('cpu')))
    model.eval()
    
    return model

# Load model once
model = load_model()
st.success("✅ Model weights loaded successfully!")

# Upload an image for prediction
uploaded_image = st.file_uploader("📸 Upload an image for prediction", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # Display the uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption="🖼 Uploaded Image", use_column_width=True)

    # Transform the image
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Prepare image for the model
    input_tensor = preprocess(image).unsqueeze(0)

    # Make prediction
    if st.button("🔍 Predict"):
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            top_prob, top_class = torch.max(probabilities, dim=1)

        # Get the predicted animal
        predicted_animal = index_to_animal.get(top_class.item(), "Unknown")
        st.success(f"🦴 **Predicted Animal: {predicted_animal}**")
        st.info(f"📊 **Confidence: {top_prob.item():.2f}**")



st.markdown("""
    <div style='text-align: center; margin-top: 50px; padding: 25px;
                border-top: 1px solid #bb86fc; color: #bb86fc;'>
        🧪 Neural Network Project Team | Advanced Computer Vision System v2.1
    </div>
""", unsafe_allow_html=True)