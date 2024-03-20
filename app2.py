from transformers import pipeline
from PIL import Image
import requests
from io import BytesIO
import torch
from models import vgg19  # Assurez-vous que models.py est correctement défini et dans votre PATH
from torchvision import transforms
###### Nous utiliserons deux models Hugging face et un model local #####
# Chemin vers votre modèle local
model_path = "model_sh_B.pth"
device = torch.device('cpu')  # Changez pour 'cuda' si vous utilisez un GPU

# Charger le modèle vgg19
model = vgg19()
model.to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Pipeline pour faire une legende et pour detecter emotions -> importées grâce à transformers
caption_pipeline = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")
emotion_pipeline = pipeline("image-classification", model="RickyIG/emotion_face_image_classification_v3")

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
    combined_result = f"Caption: {caption}\nEmotions: {emotions}\nNumber of People: {count}"
    return combined_result

# URL de l'image à tester en attendanrt version final
image_url = "https://img.freepik.com/photos-premium/equipe-souriante-fond-transparent_894067-17853.jpg"

# Chargement de l'image
response = requests.get(image_url)
image = Image.open(BytesIO(response.content))

# Génération de la légende, des émotions et du comptage des personnes
result = generate_caption_emotion_and_people_count(image)
print(result)
