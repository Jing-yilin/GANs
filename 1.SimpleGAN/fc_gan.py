"""
使用全连接层实现一个简单的GAN
Author: 景风眠
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

# 判别器
class Discriminator(nn.Module):
    def __init__(self, in_features):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.disc(x)

# 生成器
class Generater(nn.Module):
    '''
    z_dim: 噪音维度
    img_dim: 生成的图片的维度
    '''
    def __init__(self, z_dim, img_dim):
        super(Generater, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, img_dim),
            nn.Tanh() # 正则化输出
        )
    def forward(self, x):
        return self.gen(x)

# 超参设置
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'mps'
lr = 0.0003
z_dim = 64
image_dim = 28 * 28 * 1
batch_size = 32
num_epochs = 100

# 创建生成器和判别器的实例
disc = Discriminator(image_dim).to(device)
gen = Generater(z_dim, image_dim).to(device)
fixed_noise = torch.randn((batch_size, z_dim)).to(device)
transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ]
)

# 加载数据集，设置优化器
dataset = datasets.MNIST(root="dataset/", transform=transforms, download=True)
loader = DataLoader(dataset, batch_size, shuffle=True)
opt_disc = optim.Adam(disc.parameters(), lr=lr)
opt_gen = optim.Adam(gen.parameters(), lr=lr)
criterion = nn.BCELoss()
writer_fake = SummaryWriter(f"1.SimpleGAN/logs/fake")
writer_real = SummaryWriter(f"1.SimpleGAN/logs/real")
step = 0

print(f"Running on {device}")
for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(loader):
        '''
        real.size(): [32, 1, 28, 28]
        '''
        # flatten
        real = real.view(-1, 784).to(device)
        # 获取本次batch的大小
        batch_size = real.shape[0]

        #------------------#
        #      训练disc     #
        #------------------#
        # 保证gen的参数不更新
        gen.eval()
        disc.train()
        noise = torch.randn(batch_size, z_dim).to(device)
        # 生成假图像(784,)
        fake = gen(noise)
        # 判别真图像的结果 0~1
        disc_real = disc(real).view(-1)
        # 判别器鉴别真图像得到的损失
        lossD_real = criterion(disc_real, torch.ones_like(disc_real))
        # 判别假图像的结果 0~1
        disc_fake = disc(fake).view(-1)
        # 判别器鉴别假图像得到的损失
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        # 总计损失
        lossD = (lossD_real + lossD_fake) / 2
        # 更新disc参数
        disc.zero_grad()
        lossD.backward(retain_graph=True)
        opt_disc.step()

        #------------------#
        #      训练gen      #
        #------------------#
        # 保证disc的参数不更新
        disc.eval()
        gen.train()
        # 这里我们希望disc把假照片当成真的
        disc_fake = disc(fake).view(-1)
        lossG = criterion(disc_fake, torch.ones_like(disc_fake))
        # 更新gen参数
        gen.zero_grad()
        lossG.backward()
        opt_gen.step()

        # 在每个epoch首轮打印信息
        if batch_idx == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(loader)} \
                      Loss D: {lossD:.4f}, loss G: {lossG:.4f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise).reshape(-1, 1, 28, 28)
                data = real.reshape(-1, 1, 28, 28)
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(data, normalize=True)

                writer_fake.add_image(
                    "Mnist Fake Images", img_grid_fake, global_step=step
                )
                writer_real.add_image(
                    "Mnist Real Images", img_grid_real, global_step=step
                )
                step += 1
        


