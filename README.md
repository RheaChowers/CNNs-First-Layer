[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RheaChowers/first-layer-representations/blob/main/pretrained_models_demo.ipynb)
# Why do CNNs Learn Consistent Representations in their First Layer Independent of Labels and Architecture?

A demo illustrating the results of ["Why do CNNs Learn Consistent Representations in their First Layer Independent of Labels and Architecture?"](https://arxiv.org/abs/2206.02454)

# Summary
We show consistency in the representation learnt in the first layer of various CNNs by measuring the average projection of the filters in the first layer in the input's patches' principal components (PCA). We show empirically that this consistency is independent of initialization, width, architecture and even **labels** - whether true or sampled randomly. 
We continue by proving these properties on a linear CNN with a single hidden layer. We conclude by showing that this model can predict the sensitivity in the first layer to changes in the input statistics. 

# Results on Pretrained Models
As explained in greater detail in the paper, we found a great similarity beween the first layers of various pretrained networks by projecting the filters onto the principal components and measuring correlations between the projection vectors. 

An example of such results is presented here, and can be reproduced in this [demo](pretrained_models_demo.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RheaChowers/first-layer-representations/blob/main/pretrained_models_demo.ipynb).



# Citation
If you find this research interesting, feel free to cite:
```
@misc{https://doi.org/10.48550/arxiv.2206.02454,
  doi = {10.48550/ARXIV.2206.02454},
  url = {https://arxiv.org/abs/2206.02454},
  author = {Chowers, Rhea and Weiss, Yair},
  title = {Why do CNNs Learn Consistent Representations in their First Layer Independent of Labels and Architecture?},
  publisher = {arXiv},
  year = {2022},  
  copyright = {Creative Commons Attribution 4.0 International}
}

```
