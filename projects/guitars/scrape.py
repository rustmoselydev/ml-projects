from google_images_search import GoogleImagesSearch
from dotenv import load_dotenv
import os
load_dotenv()
API_KEY = os.getenv("API_KEY")
APP_ID = os.getenv("APP_ID")
q = "ibanez ar"

gis = GoogleImagesSearch(API_KEY, APP_ID)
def get_search_params(query):
    return  {
    'q': query,
    'num': 200
    }
params = get_search_params(q)

gis.search(search_params=params, path_to_dir='./guitars/' + q.replace(" ", "_"))