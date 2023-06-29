# GANs
Implementation from scratch: Simple GAN, DCGAN, WGAN, WGAN-GP, Cycle GAN, ESRGAN, Pix2Pix, ProGAN, SRGAN, StyleGAN

## Create a new environment
```bash
conda create -n GANs python=3.10
conda activate GANs
```

## Install dependencies
```bash
pip install -r requirements.txt
```

## Run
```bash
python 1.SimpleGAN/fc_gan.py
python 2.DCGAN/train.py
python 3.WGAN/train.py
python 4.WGAN-GP/train.py
```

## 查看tensorboard
```bash
tensorboard --logdir='1.SimpleGAN/logs'
tensorboard --logdir='2.DCGAN/logs'
tensorboard --logdir='3.WGAN/logs'
tensorboard --logdir='4.WGAN-GP/logs'
```