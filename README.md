[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.md)
![Python 3.7](https://img.shields.io/badge/python-3.7-green.svg)
# Handwriting Generation: Improving AFFGANwriting by Exploring Deep Learning Models for Style Encoders and Image Generation of Sentence-Level Handwriting

The goal of this work is to enhance handwriting generation by fine-tuning recent backbone models to improve the style encoder’s capability in capturing style-specific features. Additionally, the aim was to extend the system to line-level generation and deploy it as a web application, allowing users to select a writing style, input text, and instantly generate a corresponding handwriting image.

![Architecture](GAN_word/coverimage.png)

## Setup

 To install the required dependencies, enter the respective folder, create the conda environment and run the following command:
 `pip install -r requirements.txt`

 ## Dataset
 The experiments were conducted using the IAM dataset, a multi-writer dataset widely used for handwriting research. The IAM word-level subset was used for the word-level generation experiments and the IAM line-level subset for line-level handwriting generation.

## Training
For training, navigate to the respective folder and follow the instructions provided in its README.md file.


## Acknowledgments

This work builds upon and extends ideas and code from the following repositories:

- [AFFGANwriting](https://github.com/omni-us/research-GANwriting) – served as the baseline architecture for word-level handwriting synthesis, where we explored alternative backbone models.
- [Handwriting_line_generation](https://github.com/herobd/handwriting_line_generation/) – provided the foundation for line-level handwriting generation, which we integrated into our web application.
- [Emuru-autoregressive-text-img](https://github.com/aimagelab/Emuru-autoregressive-text-img) – offered an alternative approach to text-to-image generation.

I thank the authors for making their code publicly available to use.
