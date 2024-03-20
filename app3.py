# import torch
# gpu_id = torch.cuda.current_device()
# print("GPU", gpu_id)
# print(torch.__path__)
# print(torch.cuda.is_available())
# print(torch.cuda.get_arch_list())
# print(torch.cuda.device_count())

import streamlit as st
from transformers import pipeline
from PIL import Image
import torch
# Assurez-vous que models.py est correctement défini et dans votre PATH
from models import vgg19  
from torchvision import transforms

# Initialisation des pipelines Hugging Face
caption_pipeline = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")
emotion_pipeline = pipeline("image-classification", model="RickyIG/emotion_face_image_classification_v3")

# Chemin vers votre modèle local et initialisation
model_path = "model_sh_B.pth"
device = torch.device('cpu')  # Changez pour 'cuda' si vous utilisez un GPU

# Charger le modèle vgg19
model = vgg19()
model.to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

def predict_count(image):
    # Prétraitement de l'image
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_tensor = trans(image)
    inp = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs, _ = model(inp)

    count = torch.sum(outputs).item()
    return int(count)

def generate_caption_emotion_and_people_count(image):
    # Générer une légende 1 ere etape
    caption_result = caption_pipeline(image)
    caption = caption_result[0]["generated_text"]

    # Classification des émotions 2 eme etape
    emotion_result = emotion_pipeline(image)
    emotions = ", ".join([f"{res['label']}: {res['score']:.2f}" for res in emotion_result])

    # Comptage des personnes dans la photo 3 eme etaoe
    count = predict_count(image)

    # Combinaison des résultats
    # combined_result = f"Caption: {caption}\nEmotions: {emotions}\nNumber of People: {count}"
    # return combined_result

    return caption, emotions, count
def main():
    # Interface Streamlit
    st.title("Analyse d'Image avec IA pour Titouan")

    # Upload d'image
    uploaded_image = st.file_uploader("Choisissez une image...", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        image = Image.open(uploaded_image)

        # Affichage de l'image uploadée
        st.image(image, caption='Image Uploadée', use_column_width=True)
        
        # # Bouton pour lancer l'analyse
        # if st.button('Analyser l\'image'):
        #     # Génération de la légende, des émotions et du comptage des personnes
        #     result = generate_caption_emotion_and_people_count(image)
        #     st.write(result)
        # Bouton pour lancer l'analyse
        if st.button('Analyser l\'image'):
            caption, emotions, count = generate_caption_emotion_and_people_count(image)
            with st.expander("Voir les résultats !!"):
                st.write(f"**Légende**: {caption}")
                st.write(f"**Émotions**: {emotions}")
                st.write(f"**Nombre de personnes**: {count}")

if __name__ == '__main__':
    main()
