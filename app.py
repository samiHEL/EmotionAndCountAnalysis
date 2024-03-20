import gradio as gr
import torch
from models import vgg19
from PIL import Image
from torchvision import transforms
import gradio as gr
import cv2
import numpy as np

model_path = "model_sh_B.pth"
device = torch.device('cpu')  # device can be "cpu" or "gpu"

model = vgg19()
model.to(device)
model.load_state_dict(torch.load(model_path, device))
model.eval()


def predict(image_path):
    inp = Image.open(image_path)
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_tensor = trans(inp)
    inp = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs, _ = model(inp)

    count = torch.sum(outputs).item()
    vis_img = outputs[0, 0].cpu().numpy()
    vis_img = (vis_img - vis_img.min()) / (vis_img.max() - vis_img.min() + 1e-5)
    vis_img = (vis_img * 255).astype(np.uint8)
    vis_img = cv2.applyColorMap(vis_img, cv2.COLORMAP_JET)
    vis_img = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
    return vis_img, int(count)
image_path = 'le-mazette-3-_2_1200.jpg'  # Chemin de votre image
vis_img, count = predict(image_path)
print(f"Nombre de personnes détectées : {count}")
# Si vous souhaitez voir l'image avec les densités, décommentez la ligne suivante
cv2.imshow('Predicted Density Map', vis_img)
cv2.waitKey(5000)  # Attend 5000 millisecondes = 5 secondes
cv2.destroyAllWindows()






import streamlit as st
import torch
from models import vgg19
from PIL import Image
from torchvision import transforms
import cv2
import numpy as np

# Chargement du modèle
model_path = "model_sh_B.pth"
device = torch.device('cpu')  # Changez ici pour "cuda" si vous utilisez GPU

model = vgg19()
model.to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Fonction de prédiction adaptée pour Streamlit
def predict(image):
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_tensor = trans(image)
    inp = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs, _ = model(inp)

    count = torch.sum(outputs).item()
    vis_img = outputs[0, 0].cpu().numpy()
    vis_img = (vis_img - vis_img.min()) / (vis_img.max() - vis_img.min() + 1e-5)
    vis_img = (vis_img * 255).astype(np.uint8)
    vis_img = cv2.applyColorMap(vis_img, cv2.COLORMAP_JET)
    vis_img = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
    return vis_img, int(count)

# Interface Streamlit
st.title("Compteur de Foule")
st.write("Cette application estime le nombre de personnes dans une image en utilisant un modèle de deep learning.")

uploaded_file = st.file_uploader("Choisissez une image...", type=["jpg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Image téléchargée", use_column_width=True)
    st.write("Prédiction en cours...")
    vis_img, count = predict(image)
    st.image(vis_img, caption="Carte de densité prédite", use_column_width=True)
    st.write(f"Nombre de personnes détectées : {count}")










# title = "Crowd Counter"
# desc = "A demo of Crowd counting model"
# examples = [
#     ["IMG_1.jpg"],
#     ["IMG_2.jpg"],
#     ["IMG_107.jpg"],
# ]
# inputs = gr.inputs.Image(label="Image of Crowd")
# outputs = [gr.outputs.Image(label="Predicted Density Map",type = "numpy"), gr.outputs.Label(label="Predicted Count",type = "numpy")]
# gr.Interface(fn=predict, inputs=inputs, outputs=outputs, title=title, description=desc, examples=examples,
#              allow_flagging=False).launch()