""" Module to send requests to backend server """
import base64
from io import BytesIO
import requests
from PIL import Image


TIMEOUT_VALUE = 10


class ServiceError(Exception):
    """ Class to set status's code """
    def __init__(self, status_code):
        self.status_code = status_code


def get_images_from_backend(prompt, backend_url):
    """ Function to get requested image """
    request = requests.post(backend_url, json={"prompt": prompt}, timeout=TIMEOUT_VALUE)

    # if success return image and version else error code
    if request.status_code == 200:
        json = request.json()
        images = json["images"]
        images = [Image.open(BytesIO(base64.b64decode(img))) for img in images]
        version = json.get("version", "unknown")
        return {"images": images, "version": version}
    else:
        raise ServiceError(request.status_code)


def get_model_version(url):
    """ Function to get model's version """
    request = requests.get(url, timeout=TIMEOUT_VALUE)

    # if success, return model's version else error code
    if request.status_code == 200:
        version = request.json()["version"]
        return version
    else:
        raise ServiceError(request.status_code)
