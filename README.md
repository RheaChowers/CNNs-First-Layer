[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RheaChowers/first-layer-representations/blob/main/pretrained_models_demo.ipynb)
# What do CNNs Learn in the First Layer and Why? A Linear Systems Perspective

This repository is the official implementation of of our paper ["What do CNNs Learn in the First Layer and Why? A Linear Systems Perspective"](https://arxiv.org/abs/2206.02454), by Rhea Chowers and Yair Weiss, which has been accepted to ICML 2023.

# Summary
We show that trained networks learn consistent representations that are far from their initialization despite the fact that CNNs with commonly used architectures can be trained with fixed, random filters in the first layer and still yield comparable performance to full learning. We also show that the same energy profile is obtained when the network is trained to predict random labels. We then show that under realistic assumptions on the statistics of the input and labels, consistency also occurs
in simple, linear CNNs, and derive an analytical form for its energy profile. We show that as the number of iterations goes to infinity, this profile takes the form of a first layer that performs whitening of the input image patches. Finally, we show that the analytical formula which we derived for linear CNNs gives an excellent fit to the energy profile of real-world CNNs as well, when trained with either true or random labels.

# Results on Pretrained Models
As explained in greater detail in the paper, we found a great similarity beween the first layers of various pretrained networks by projecting the filters onto the principal components and measuring correlations between the projection vectors. 

An example of such results is presented here, and can be reproduced in this [demo](pretrained_models_demo.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RheaChowers/first-layer-representations/blob/main/pretrained_models_demo.ipynb).



# Citation
```
@misc{chowers2023cnns,
      title={What do CNNs Learn in the First Layer and Why? A Linear Systems Perspective}, 
      author={Rhea Chowers and Yair Weiss},
      year={2023},
      eprint={2206.02454},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

```
