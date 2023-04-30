---
title: FaceGenerator
emoji: 😄
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: 3.17.0
app_file: app.py
pinned: false
---

# Face Generator

## Description

### About

A simple application to generate human face images with the limitation of the algorithm, dataset, hardware specification, and other outside variables.

![UI](/img/ui.png)

1. "Text prompt" textbox ,to enter text description of desired image
2. "Gender" radio button, to choose desired gender
3. "Attributes" checkboxes, to choose whether to include certain attributes or not
4. "RUN" button (or keyboard ENTER button), to confirm input
5. Result placeholder, the place for generated image

### Notes

- Loading and saving models are configured for training from scratch only
- Loading and saving models are configured to my own use; change them to your suitable one
- Optimisation only using Distributed Shampoo
- The dataset is limited to CelebA-HQ

### Dependencies

Main:

- python >=3.7
- gradio
- transformers
- einops
- unidecode
- ftfy
- emoji
- pillow
- jax
- flax
- tqdm
- optax
- braceexpand
- datasets[streaming]
- black[jupyter]
- isort

Additionals:

- wandb

---

## Guide

### How to use

Users input either free-form text in the textbox or choose one or several attribute options in the form of radio buttons and checkboxes then press the RUN or ENTER button to confirm them. Then the user waits until the desired image is generated and shown in the previously empty placeholder.

---

## Files Structure

- app

Web application files.

- encode

Notebook for encoding dataset including the output history.

- img

Additional images for README files or other purposes.

- old_results

Old notebooks or other files for tracking missing issues. Or other things.

- train

Files for main training.

  - config

  Configuration file.

  - dalle_mini

  The model based on.

  - scalable_shampoo

  The optimiser files from [Google Research](https://github.com/google-research/google-research/tree/master/scalable_shampoo)

  - train.py

  The main training script,

  - train.ipynb

  The notebook for running training script and setting up used workplace; configure it to your own uses or just immediately run the training script. This is just for convinient uses in case of failures.

- README.md

This file; which is the explanations.

---

## References

### Papers

```text
@misc{
  title={Zero-Shot Text-to-Image Generation}, 
  author={Aditya Ramesh and Mikhail Pavlov and Gabriel Goh and Scott Gray and Chelsea Voss and Alec Radford and Mark Chen and Ilya Sutskever},
  year={2021},
  eprint={2102.12092},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  link={[]()}
}
```

### Datasets

```text
@inproceedings{CelebAMask-HQ,
  title={MaskGAN: Towards Diverse and Interactive Facial Image Manipulation},
  author={Lee, Cheng-Han and Liu, Ziwei and Wu, Lingyun and Luo, Ping},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2020}
}
```

```text
@inproceedings{xia2021tedigan,
  title={TediGAN: Text-Guided Diverse Face Image Generation and Manipulation},
  author={Xia, Weihao and Yang, Yujiu and Xue, Jing-Hao and Wu, Baoyuan},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2021}
}

@article{xia2021open,
  title={Towards Open-World Text-Guided Face Image Generation and Manipulation},
  author={Xia, Weihao and Yang, Yujiu and Xue, Jing-Hao and Wu, Baoyuan},
  journal={arxiv preprint arxiv: 2104.08910},
  year={2021}
}

@inproceedings{karras2017progressive,
  title={Progressive growing of gans for improved quality, stability, and variation},
  author={Karras, Tero and Aila, Timo and Laine, Samuli and Lehtinen, Jaakko},
  journal={International Conference on Learning Representations (ICLR)},
  year={2018}
}

@inproceedings{liu2015faceattributes,
 title = {Deep Learning Face Attributes in the Wild},
 author = {Liu, Ziwei and Luo, Ping and Wang, Xiaogang and Tang, Xiaoou},
 booktitle = {Proceedings of International Conference on Computer Vision (ICCV)},
 year = {2015} 
}
```

### Codes, and Libraries

```text
@misc{Dayma_DALL·E_Mini_2021,
      author = {Dayma, Boris and Patil, Suraj and Cuenca, Pedro and Saifullah, Khalid and Abraham, Tanishq and Lê Khắc, Phúc and Melas, Luke and Ghosh, Ritobrata},
      doi = {10.5281/zenodo.5146400},
      month = {7},
      title = {DALL·E Mini},
      url = {https://github.com/borisdayma/dalle-mini},
      year = {2021}
}
```

```text
@software{jax2018github,
  author = {James Bradbury and Roy Frostig and Peter Hawkins and Matthew James Johnson and Chris Leary and Dougal Maclaurin and George Necula and Adam Paszke and Jake Vander{P}las and Skye Wanderman-{M}ilne and Qiao Zhang},
  title = {{JAX}: composable transformations of {P}ython+{N}um{P}y programs},
  url = {http://github.com/google/jax},
  version = {0.3.13},
  year = {2018},
}
```

### Others

- [Gradio documentation](https://gradio.app/docs)

### Helping Hands

https://www.youtube.com/@TheAIEpiphany

---

## Tools Used

- [Google Colab](https://colab.research.google.com/)
- [Paperspace Gradient](https://www.paperspace.com/gradient)
- [Kaggle](https://www.kaggle.com/)
- [Figma](https://www.figma.com/)
