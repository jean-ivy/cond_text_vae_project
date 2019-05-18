import torch
import torch.nn as nn
import torch.nn.functional as F
from TextEncoder import TextEncoder


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input, size=(128, 7, 7)):
        return input.view(input.size(0), *size)


class TextCondVAE(nn.Module):
    def __init__(self, image_channels, img_h_dim, text_h_dim, z_dim,
                 vocab_size, emb_dim, cnn_params):
        super(TextCondVAE, self).__init__()

        # encoder
        self.conv1 = nn.Conv2d(image_channels, 32, kernel_size=4, stride=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, return_indices=True)
        self.pool_size1 = None

        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, return_indices=True)
        self.pool_size2 = None

        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, return_indices=True)
        self.pool_size3 = None

        self.flatten = Flatten()

        self.img_size = 128 * 7 * 7
        self.fc_enc = nn.Linear(self.img_size, img_h_dim)

        # bottleneck
        self.fc1 = nn.Linear(img_h_dim + text_h_dim, z_dim)
        self.fc2 = nn.Linear(img_h_dim + text_h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim + text_h_dim, self.img_size)

        # decoder
        self.unflatten = UnFlatten()

        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2)
        self.uppool1 = nn.MaxUnpool2d(2, stride=2)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')

        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2)
        self.uppool2 = nn.MaxUnpool2d(2, stride=2)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')

        self.deconv3 = nn.ConvTranspose2d(32, image_channels, kernel_size=4, stride=2)
        self.uppool3 = nn.MaxUnpool2d(2, stride=2)
        self.upsample3 = nn.Upsample(scale_factor=2, mode='nearest')

        self.sigm = nn.Sigmoid()

        self.text_encoder = TextEncoder(vocab_size, emb_dim, text_h_dim, **cnn_params)

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size()).cuda()
        z = mu + std * esp
        return z

    def bottleneck(self, img_h_dim, text_h_dim):
        text_img_h_dim = torch.cat((img_h_dim, text_h_dim), 1)
        mu, logvar = self.fc1(text_img_h_dim), self.fc2(text_img_h_dim)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x, text_h_dim):

        out = F.relu(self.conv1(x))
        if self.pool_size1 is None:
            self.pool_size1 = out.size()[2:]

        out, self.indices1 = self.pool1(out)

        out = F.relu(self.conv2(out))
        if self.pool_size2 is None:
            self.pool_size2 = out.size()[2:]

        out, self.indices2 = self.pool2(out)

        out = F.relu(self.conv3(out))

        if self.pool_size3 is None:
            self.pool_size3 = out.size()[2:]
        out, self.indices3 = self.pool3(out)

        out = self.flatten(out)
        h = self.fc_enc(out)

        z, mu, logvar = self.bottleneck(h, text_h_dim)
        return z, mu, logvar

    def decode(self, z, text_h_dim):
        z = torch.cat((z, text_h_dim), 1)
        out = self.fc3(z)
        out = self.unflatten(out, size=(128, 7, 7))

        if self.indices3 is None:
            out = self.upsample1(out)
        else:
            out = self.uppool1(out, self.indices3, output_size=self.pool_size3)
        out = F.relu(self.deconv1(out))

        if self.indices2 is None:
            out = self.upsample2(out)
        else:
            out = self.uppool2(out, self.indices2, output_size=self.pool_size2)
        out = F.relu(self.deconv2(out))

        if self.indices1 is None:
            out = self.upsample3(out)
        else:
            out = self.uppool3(out, self.indices1, output_size=self.pool_size1)
        out = F.relu(self.deconv3(out))

        out = self.sigm(out)

        self.indices1 = None
        self.indices2 = None
        self.indices3 = None

        return out

    def get_h_text(self, caption):
        return self.text_encoder(caption)

    def forward(self, image, caption):
        text_h_dim = self.get_h_text(caption)
        z, mu, logvar = self.encode(image, text_h_dim)
        z = self.decode(z, text_h_dim)
        return z, mu, logvar


def loss_fn(recon_x, x, mu, logvar):
    MSE = F.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return MSE + KLD, MSE, KLD