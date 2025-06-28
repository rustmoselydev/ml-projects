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
            if int(filename.split(".")[0]) > 1112:
                image_path = os.path.join(image_folder, filename)
                image = encode_image(image_path)
                try:
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    { "type": "text", "text": "You are looking at a 32x32 pixel art character. Describe its gender, outfit, age, skin color, hair, and colors as best you can. If the character seems to have an obviously specific job like king or soldier, mention that. If they are not human, describe what creature they resemble. If they don't meet a specific criteria mentioned in this prompt, do not mention that criteria (e.g. if they aren't a king, don't say so explicitly)" },
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
csv = pd.read_csv("generated_captions.csv")
combined = pd.concat([df, csv], ignore_index=True)
df.to_csv("generated_captions.csv", index=False)
print("CSV saved!")

