import base64
import os
import pandas as pd
from openai import OpenAI, files
import time
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
image_folder = "../../training-data/character-forward"
output_data = []

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


for filename in sorted(os.listdir(image_folder)):
    if filename.lower().endswith(('.png')):
            if int(filename.split(".")[0]) > -1:
                image_path = os.path.join(image_folder, filename)
                image = encode_image(image_path)
                try:
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    { "type": "text", "text": "You are looking at a fantasy style pixel art character from a video game. Describe the following attributes about this character: hair color, hair style, skin color, all discernable clothing or armor (including the color of each piece), any discernable accessories or items in hand, gender, and their possible fantasy class or job (which can also just be something like being a commoner, townsperson, merchant, or other adjectives for an average person). Do not mention that this is a fantasy pixel art character. The format should be a series of sentences- no bullet lists or any other formatting. IMPORTANT: If you cannot discern a particular trait, just don't mention it." },
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{image}"
                                        }
                                        
                                    },
                                ],
                            }
                        ],
                    )
                    caption = response.choices[0].message.content
                    print(caption)
                    image_id = os.path.splitext(filename)[0]
                    output_data.append({"image": image_id, "caption": caption})
                    time.sleep(1)
                except Exception as e:
                    print(e)
                    pass

# Save to CSV
df = pd.DataFrame(output_data)
# csv = pd.read_csv("generated_captions.csv")
# combined = pd.concat([df, csv], ignore_index=True)
df.to_csv("generated_captions.csv", index=False)
print("CSV saved!")

