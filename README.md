# Distribution Filling Back-Propagation

​	This repository provides an implement of Distribution Filling Back-Propagation.  We use stable diffusion model to generate more virtual samples from uniform design points rather than random points in the latent space, in order to fine-tuning the classification model. 

## Requirement

This work use CLIP as the condition model. Ones can get the weights from:

* [HuggingFace](https://huggingface.co/openai/clip-vit-large-patch14/tree/main)
* [Baidupan](https://pan.baidu.com/s/1EBptJ2v9inq9A5LEYFfBMg)：**Access Code** dh2b

Other python environment:

|                    Environment                    | version |
| :-----------------------------------------------: | :-----: |
|                      PyTorch                      |  2.0.1  |
|                    torchvision                    | 0.18.0  |
|                       numpy                       | 1.23.5  |
| [pyunidoe](https://github.com/ZebinYang/pyunidoe) |   0.1   |
|                   transformers                    | 4.49.0  |
|                      typing                       | 3.7.4.3 |

