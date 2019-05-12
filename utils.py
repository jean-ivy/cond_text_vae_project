import os
import re
import nltk
import cv2

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


nltk.download('stopwords')

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

def text_prepare(text):
    text = text.lower()
    text = re.sub(REPLACE_BY_SPACE_RE, " ", text)
    text = re.sub(BAD_SYMBOLS_RE, "", text)
    text = ' '.join([word for word in text.split() if word not in STOPWORDS])
    return text

def gray(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray

def prepare_pascal(path):
    lemmatizer = WordNetLemmatizer()

    # Prepare text descriptors
    word_description = {}
    for folder_cls in os.listdir(path):
        folder_cls_path = os.path.join(path, folder_cls)
        for example in os.listdir(folder_cls_path):
            example_path = os.path.join(folder_cls_path, example)
            txt_path = os.path.join(example_path, "description.txt")
            full_name = "{}_{}".format(folder_cls, example)
            with open(txt_path, "r") as f:
                word_description[full_name] = f.read().strip().split('\n')

    for img_name, text_info in word_description.items():
        for i, sent in enumerate(text_info):
            lemmatized_sent = []
            for idx, word in enumerate(text_prepare(sent).split()):
                lemmatized_sent.append(lemmatizer.lemmatize(word))
            text_info[i] = lemmatized_sent

    # Prepare image descriptors
    image_description = {}
    for folder_cls in os.listdir(path):
        folder_cls_path = os.path.join(path, folder_cls)
        for example in os.listdir(folder_cls_path):
            example_path = os.path.join(folder_cls_path, example)
            image_path = os.path.join(example_path, "image.jpg")
            full_name = "{}_{}".format(folder_cls, example)
            image_description[full_name] = cv2.imread(image_path)

    for img_name, img in image_description.items():
        image_description[img_name] = gray(img)

    return word_description, image_description


