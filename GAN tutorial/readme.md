# Generative Adversarial Networks (GANs)
---

Demo on how to build a Generative Adversarial Network (GAN)[1] using the paper on DCGAN[2].

## Setup
---

We recommend running the code in a virtual environment with Python > 3.5.x (fully tested on Python 3.6.5):
```
virtualenv -p python3 venv
source venv/bin/activate
pip3 install -r requirements.txt
```

Run `jupyter notebook`

## Data
---

For training the DCGAN network, we have used the MNIST[3] dataset. It contains images of handwritten numbers along with the labels. For training the DCGAN network, we will only use the images and not the labels. The size of the image is 1x28x28. 

![mnist](https://user-images.githubusercontent.com/30028589/46747656-df0ec480-cc7f-11e8-8a08-c84689631f84.png "MNIST Dataset")
