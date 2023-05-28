""" Project's details """

from setuptools import setup

NAME = "ImageGenerator"
VERSION = "1.0.0"
DESCRIPTION = "The first (trial) deep learning of image generation based on Craiyon"
URL = "https://github.com/fedorajuandy/tugas-akhir"
EMAIL = "fedorajuandy@gmail.com"
AUTHOR = "Fedora Yoshe Juandy"
REQUIRES_PYTHON = ">=3.7.0"
VERSION = "1.0.0"

REQUIRED = [
    "transformers",
    "einops",
    "unidecode",
    "ftfy",
    "emoji",
    "pillow",
    "jax",
    "flax",
    "wandb",
    "tqdm",
    "optax"
    "braceexpand",
    "datasets[streaming]",
    "black[jupyter]",
    "isort"
    "gradio"
]

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    install_requires=REQUIRED,
    license='Apache Software License',
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.7",
    ],
)
