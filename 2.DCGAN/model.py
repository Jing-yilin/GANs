"""
Discriminator and Generator implementation from DCGAN paper
paper url: https://arxiv.org/pdf/1511.06434.pdf
tutorial: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
reference video: https://www.youtube.com/watch?v=Tk5B4seA-AU&list=PLJV_el3uVTsPy9oCRY30oBPNLCo89yu49&index=26&ab_channel=Hung-yiLee
Author: 景风眠
"""

import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, channles_img, features_d):
        '''
        input: (N, channels_img, 64, 64)
        '''
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(channles_img, features_d, kernel_size=4, stride=2, padding=1), # 64*64 -> 32*32
            nn.LeakyReLU(0.2),
            # _block(self, in_channels, out_channels, kernel_size, stride, padding)
            self._block(features_d, features_d*2, 4, 2, 1), # 32*32 -> 16*16
            self._block(features_d*2, features_d*4, 4, 2, 1), # 16*16 -> 8*8
            self._block(features_d*4, features_d*8, 4, 2, 1), # 8*8 -> 4*4
            # After all _block img output is 4x4 (Conv2d below makes into 1x1)
            nn.Conv2d(features_d*8, out_channels=1, kernel_size=4, stride=2, padding=0),
            nn.Sigmoid()
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=False
            ),
            # nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.disc(x)
    

class Generator(nn.Module):
    def __init__(self, channels_noise, channels_img, features_g):
        '''
        input: (N, channels_noise, 1, 1)
        '''
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            #  _block(self, in_channels, out_channels, kernel_size, stride, padding)
            self._block(channels_noise, features_g*16, 4, 1, 0), # 1*1 -> 4*4   1-4 + 2*(4-1-0) + 1=4
            self._block(features_g*16, features_g*8, 4, 2, 1), # 4*4 -> 8*8   4-4+2*(4-1-1) + (4-1)*(2-1) + 1= 8
            self._block(features_g*8, features_g*4, 4, 2, 1), # 8*8 -> 16*16  8-4+2*(4-1-1) + (8-1)*(2-1) + 1= 16
            self._block(features_g*4, features_g*2, 4, 2, 1), # 8*8 -> 16*16  16-4+2*(4-1-1) + (16-1)*(2-1) + 1= 32
            nn.ConvTranspose2d(
                features_g * 2, channels_img, kernel_size=4, stride=2, padding=1
            ),
            # Output: N x channels_img x 64 x 64
            nn.Tanh(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
    
    def forward(self, x):
        return self.net(x)

    

if __name__ == "__main__":
    disc = Discriminator(channles_img=3, features_d=8)
    x = torch.randn((2, 3, 64, 64))
    out = disc(x)
    print(out.size()) # torch.Size([2, 1, 1, 1])

    gen = Generator(channels_noise=100, channels_img=3, features_g=4)
    x = torch.randn((2, 100, 1, 1))
    out = gen(x)
    print(out.size()) # torch.Size([2, 3, 64, 64])
