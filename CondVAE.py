import torch
import torch.nn as nn
import torch.nn.functional as F
from TextEncoder import TextEncoder


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input, size):
        return input.view(input.size(0), size, 1, 1)


class TextCondVAE(nn.Module):
    def __init__(self, image_channels, img_h_dim, text_h_dim, z_dim,
                 vocab_size, emb_dim, cnn_params):
        super(TextCondVAE, self).__init__()

        # encoder
        self.conv1 = nn.Conv2d(image_channels, 32, kernel_size=4, stride=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, return_indices=True)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, return_indices=True)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, return_indices=True)

        self.flatten = Flatten()
        self.fc_enc = nn.Linear(128 * 7 * 7, img_h_dim)

        # bottleneck
        self.fc1 = nn.Linear(img_h_dim + text_h_dim, z_dim)
        self.fc2 = nn.Linear(img_h_dim + text_h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, img_h_dim)

        # decoder
        self.unflatten = UnFlatten()

        self.deconv1 = nn.ConvTranspose2d(img_h_dim + text_h_dim, 128, kernel_size=7, stride=1)
        self.uppool1 = nn.MaxUnpool2d(2, stride=2)

        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2)
        self.uppool2 = nn.MaxUnpool2d(2, stride=2)

        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2)
        self.uppool3 = nn.MaxUnpool2d(2, stride=2)

        self.deconv4 = nn.ConvTranspose2d(32, 1, kernel_size=6, stride=2)
        # self.uppool4 = nn.MaxUnpool2d(2, stride=2)

        self.sigm = nn.Sigmoid()
        self.text_encoder = TextEncoder(vocab_size, emb_dim, text_h_dim, **cnn_params)

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size()).cuda()
        z = mu + std * esp
        return z

    def bottleneck(self, h, h_text):
        mu, logvar = self.fc1(torch.cat((h, h_text), 1)), self.fc2(torch.cat((h, h_text), 1))
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x, h_text):
        # print('enc_inp', x.shape)
        out = F.relu(self.conv1(x))
        # print('conv1', out.shape)
        out, self.indices1 = self.pool1(out)
        # print('pool1', out.shape)

        out = F.relu(self.conv2(out))
        # print('conv2', out.shape)
        out, self.indices2 = self.pool2(out)
        # print('pool2', out.shape)

        out = F.relu(self.conv3(out))
        # print('conv3', out.shape)
        out, self.indices3 = self.pool3(out)
        # print('pool3', out.shape)

        out = self.flatten(out)
        # print('flat', out.shape)
        h = self.fc_enc(out)

        # print("h", h.shape)
        # print("h_text", h_text.shape)
        z, mu, logvar = self.bottleneck(h, h_text)
        return z, mu, logvar

    def decode(self, z, h_text):
        z = self.fc3(z)
        z = torch.cat((z, h_text), 1)

        # print('dec_inp', z.shape)
        out = self.unflatten(z)
        # print('un_flat', out.shape)

        out = F.relu(self.deconv1(out))
        # print('deconv1', out.shape)
        out = self.uppool1(out, self.indices3)
        # print('uppool1', out.shape)

        out = F.relu(self.deconv2(out))
        # print('deconv2', out.shape)
        out = self.uppool2(out, self.indices2)
        # print('uppool2', out.shape)

        out = F.relu(self.deconv3(out))
        # print('deconv3', out.shape)
        out = self.uppool3(out, self.indices1)
        # print('uppool3', out.shape)

        out = F.relu(self.deconv4(out))
        # print('deconv4', out.shape)
        z = self.sigm(out)

        return z

    def get_h_text(self, caption):
        return self.text_encoder(caption)

    def forward(self, x, caption):
        h_text = self.get_h_text(caption)
        z, mu, logvar = self.encode(x, h_text)
        z = self.decode(z, h_text)
        return z, mu, logvar


def loss_fn(recon_x, x, mu, logvar):
    MSE = F.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return MSE + KLD, MSE, KLD