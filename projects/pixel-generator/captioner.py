# This does an ATROCIOUS job with pixel art. Keeping it here for my future reference for other projects

import os
import pandas as pd
from transformers import pipeline
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

# Initialize the BLIP captioning model
# captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base", prompt="Describe the appearance of this pixel art character, including gender, clothing, skin color, and any unique features.")

image_folder = "../../training-data/character-forward"
output_data = []
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

labels = [
    "a male character",
    "a female character",
    "child",
    "a skeleton",
    "an orc",
    "a robot",
    "a frog",
    "old person",
    "naked",
    "wearing sunglasses",
    "wearing eyeglasses",
    "wearing armor",
    "wearing a helmet",
    "wearing a hat",
    "wearing a robe",
    "wearing a dress",
    "wearing a cloak",
    "wearing a cape",
    "wearing a hood",
    "holding a spear",
    "holding a sword",
    "wearing green clothes",
    "wearing white clothes",
    "wearing blue clothes",
    "wearing grey clothes",
    "wearing red clothes",
    "wearing yellow clothes",
    "wearing orange clothes",
    "wearing purple clothes",
    "wearing pink clothes",
    "wearing black clothes",
    "not wearing a shirt",
    "not wearing pants",
    "white skin",
    "light brown skin",
    "dark brown skin",
    "blue skin",
    "green skin",
    "bald",
    "mustache",
    "beard",
    "partially bald",
    "horns on their head",
    "pigtails",
    "red hair",
    "orange hair",
    "yellow hair",
    "green hair",
    "blue hair",
    "purple hair",
    "white hair",
    "grey hair",
    "long hair",
    "short hair",
    "spiky hair",
    "pointy ears",
    "warrior",
    "king",
    "princess",
    "archer",
    "rogue",
    "wizard",
    "cleric",
    "priest",
    "goblin",
    "soldier",
    "scientist",
    "monster"
]


for filename in sorted(os.listdir(image_folder)):
    if filename.lower().endswith(('.png')):
        image_path = os.path.join(image_folder, filename)
        try:
            image = Image.open(image_path)
            inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)

            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
            caption = ''
            # Top matches
            for label, prob in zip(labels, probs[0]):
                if prob.item() > 0.5:
                    print(f"{label}: {prob.item():.2f}")
                    caption += f"{label}, "
            # caption = captioner(image_path)[0]['generated_text']
            print(caption)
            image_id = os.path.splitext(filename)[0]
            output_data.append({"image": image_id, "caption": caption})
        except Exception as e:
            output_data.append({"image": filename, "caption": f"ERROR: {str(e)}"})

# Save to CSV
df = pd.DataFrame(output_data)
df.to_csv("generated_captions.csv", index=False)
print("CSV saved!")










