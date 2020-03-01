## ProGAN For Humans
A Pytorch implementation of the model presented in "Progressive Growing of GANs for Improved Quality, Stability, and Variation". The video short-training.mp4 shows a short training run of 4 passes over the dataset (120,000 images shown at each phase of training, compared to 800,000 in the original paper). The results are mildly terrifying, but hopefully get the point across.</br>
Although the paper is well-written and clear, in my experience the [original code](https://github.com/tkarras/progressive_growing_of_gans) is dense and difficult to follow for people not already deeply familiar with Tensorflow and machine learning research code. This implementation is an attempt to improve on the accessibility of the techniques in use.</br>

Thanks to [Animesh Karnewar's Pytorch implementation of ProGAN](https://github.com/akanimax/pro_gan_pytorch) for some inspiration and [Erik Linder-Norén](https://github.com/eriklindernoren/PyTorch-GAN) for inspiration on implenting straightforward GANs.

Because this code's readability was the highest priority, some other features of the original project are lacking and the project is likely not "production-ready". For instance:</br>

* The code is not platform agnostic - it assumes a machine with 16 CPU cores and 2 GPUs with 8 GB VRAM apiece. Instead, comments point out these assumptions when they are made and gives guidance on changing them for different machines.
* The code assumes that the dataset consists of 1024x1024 images and does not check to ensure that there is adequate disk space for preprocessed images.
* The code is not particularly well-optimized - optimizations that have been made are either common PyTorch idioms or were necessary to make the model fit.
* Some comments do assume you're using default configurations - in particular, if you see the magic number 512 come up in a comment, it's referring to the dimensionality of the latent space.
</br>
In general, I try to assume you have the paper on hand - comments don't generally explain "why"s that are covered in the paper, but instead try to give enough information that you know what part of the paper is relevant. Likewise, the code assumes a basic level of familiarity with convolutional neural networks. If you don't have that yet, I strongly recommend the following resources:

* For an introduction to deep learning in general:
** [MIT Course 6.034 Lecture 12A: Neural Nets](https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-034-artificial-intelligence-fall-2010/lecture-videos/lecture-12a-neural-nets)
** [MIT Course 6.034 Lecture 12B: Deep Neural Nets](https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-034-artificial-intelligence-fall-2010/lecture-videos/lecture-12b-deep-neural-nets)

* For an introduction to convolutional neural networks and modern training techniques:
** [Stanford CS234n: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/)

# Requirements:

The following Python modules are required. I recommend installing the Python requirement with ``venv`` or something similar.
* ``Pillow``
* ``numpy``
* ``torch``
* ``torchvision``

You will also require a copy of the CelebA-HQ dataset (or other dataset of 1024x1024 images). Getting this dataset in the form of images is actually more challenging than it really should be. One way of getting hold of it is [this project](https://github.com/willylulu/celeba-hq-modified). Alternatively, you may be able to get the data from this [Google Drive folder](https://drive.google.com/drive/folders/11Vz0fqHS2rXDb5pprgTjpD7S2BAJhi1P). Finally, if you don't care that much about replication, you can just use the [FFHQ dataset](https://github.com/NVlabs/ffhq-dataset) which is larger, higher-quality, and in PNG format by default.</br>

You may also find it helpful to set the environment variable ``CUDA_VISIBLE_DEVICES=1,0`` - your OS may be using some VRAM on your 0th GPU, so reversing them for Pytorch can give you just a bit more room on the GPU used for unparallelized scratch space.</br>
