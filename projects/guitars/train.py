from fastai.vision.all import *
from matplotlib import pyplot
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

files = get_image_files("guitars")
print(len(files))

def get_category(filename):
    return filename.parent.name.replace("_", " ")

# Data not in repository because of file size of 2000+ images
# Trained on Fender, Gibson, and Ibanez electric guitars
dataloader = ImageDataLoaders.from_path_func("./guitars", files, get_category, item_tfms=Resize(800, method="squish"), bs=64, batch_tfms=aug_transforms())
dataloader.device = torch.device("mps")
dataloader.show_batch()
#you must close the pyplot window to begin the training
pyplot.show()
learner = vision_learner(dataloader, resnet34, metrics=error_rate)

learner.fine_tune(40)
learner.export('guitar-ai-model.pkl')
