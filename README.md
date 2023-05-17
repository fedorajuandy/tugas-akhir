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

A simple application to generate human face images with the limitation of the algorithm, dataset, hardware specification, chosen limited configuration, and other various variables.

![UI](/img/ui.png)

1. "Text prompt" textbox ,to enter text description of desired image
2. "Gender" radio button, to choose desired gender
3. "Attributes" checkboxes, to choose whether to include certain attributes or not
4. "RUN" button (or keyboard ENTER button), to confirm input
5. Result placeholder, the place for generated image

### Notes

- Optimisation only using Distributed Shampoo
- The dataset is limited to CelebA-HQ

---

## Guide

### How to use

Users input either free-form text in the textbox or choose one or several attribute options in the form of radio buttons and checkboxes then press the RUN or ENTER button to confirm them. Then the user waits until the desired image is generated and shown in the previously empty placeholder.

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

```text
@misc{esser2020taming,
  title={Taming Transformers for High-Resolution Image Synthesis}, 
  author={Patrick Esser and Robin Rombach and Björn Ommer},
  year={2020},
  eprint={2012.09841},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```

### Others

- [Gradio documentation](https://gradio.app/docs)

### Papers' explanation

- 

---

## Tools Used

- [Amazon SageMaker Studio Lab](https://studiolab.sagemaker.aws/)
- [Google Colab](https://colab.research.google.com/)
- [Paperspace Gradient](https://www.paperspace.com/gradient)
- [Kaggle](https://www.kaggle.com/)
- [Figma](https://www.figma.com/)
- [Visual Studio Code Space in GitHub](https://github.com/)
- [Weights & Biases](https://wandb.ai/home)
