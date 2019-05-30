import os
import re
import nltk
import cv2

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()


def text_prepare(text):
    text = text.lower()
    text = re.sub(REPLACE_BY_SPACE_RE, " ", text)
    text = re.sub(BAD_SYMBOLS_RE, "", text)
    #text = ' '.join([word for word in text.split() if word not in STOPWORDS])
    text = ' '.join([LEMMATIZER.lemmatize(word) for word in text.split()])
    return text

def gray(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray

def prepare_pascal(path):

    # Prepare text descriptors
    word_description, image_description = {}, {}

    for folder_cls in os.listdir(path):
        folder_cls_path = os.path.join(path, folder_cls)

        for example in os.listdir(folder_cls_path):
            example_path = os.path.join(folder_cls_path, example)
            txt_path = os.path.join(example_path, "description.txt")
            image_path = os.path.join(example_path, "image.jpg")
            full_name = "{}_{}".format(folder_cls, example)
            full_name += "_{}"
            with open(txt_path, "r") as f:
                text_descriptions = f.read().strip().split('\n')
                for i in range(len(text_descriptions)):
                    word_description[full_name.format(i)] = text_prepare(text_descriptions[i])
                    image_description[full_name.format(i)] = gray(cv2.imread(image_path))

    return word_description, image_description


