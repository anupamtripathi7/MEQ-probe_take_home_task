import torch
import torch.nn as nn


class UNet2D(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Sequential(nn.Conv2d(3, 32, (3, 3), padding=(1, 1)),
                                    nn.ReLU(),
                                    nn.Conv2d(32, 32, (3, 3), padding=(1, 1)),
                                    nn.ReLU())

        self.conv_2 = nn.Sequential(nn.Conv2d(32, 32, (3, 3), padding=(1, 1)),
                                    nn.ReLU(),
                                    nn.Conv2d(32, 32, (3, 3), padding=(1, 1)),
                                    nn.ReLU())

        self.conv_3 = nn.Sequential(nn.Conv2d(32, 64, (3, 3), padding=(1, 1)),
                                    nn.ReLU(),
                                    nn.Conv2d(64, 64, (3, 3), padding=(1, 1)),
                                    nn.ReLU())

        self.conv_4 = nn.Sequential(nn.Conv2d(64, 128, (3, 3), padding=(1, 1)),
                                    nn.ReLU(),
                                    nn.Conv2d(128, 128, (3, 3), padding=(1, 1)),
                                    nn.ReLU())

        self.conv_5 = nn.Sequential(nn.Conv2d(128, 256, (3, 3), padding=(1, 1)),
                                    nn.ReLU(),
                                    nn.Conv2d(256, 256, (3, 3), padding=(1, 1)),
                                    nn.ReLU())

        self.conv_6 = nn.Sequential(nn.Conv2d(256, 512, (3, 3), padding=(1, 1)),
                                    nn.ReLU(),
                                    nn.Conv2d(512, 512, (3, 3), padding=(1, 1)),
                                    nn.ReLU())

        self.deconv_1 = nn.ConvTranspose2d(512, 128, (2, 2), stride=(2, 2))
        self.conv_7 = nn.Sequential(nn.Conv2d(384, 256, (3, 3), padding=(1, 1)),
                                    nn.ReLU(),
                                    nn.Conv2d(256, 256, (3, 3), padding=(1, 1)),
                                    nn.ReLU())

        self.deconv_2 = nn.ConvTranspose2d(256, 64, (2, 2), stride=(2, 2))
        self.conv_8 = nn.Sequential(nn.Conv2d(192, 128, (3, 3), padding=(1, 1)),
                                    nn.ReLU(),
                                    nn.Conv2d(128, 128, (3, 3), padding=(1, 1)),
                                    nn.ReLU())

        self.deconv_3 = nn.ConvTranspose2d(128, 32, (2, 2), stride=(2, 2))
        self.conv_9 = nn.Sequential(nn.Conv2d(96, 64, (3, 3), padding=(1, 1)),
                                    nn.ReLU(),
                                    nn.Conv2d(64, 64, (3, 3), padding=(1, 1)),
                                    nn.ReLU())

        self.deconv_4 = nn.ConvTranspose2d(64, 16, (2, 2), stride=(2, 2))
        self.conv_10 = nn.Sequential(nn.Conv2d(48, 32, (3, 3), padding=(1, 1)),
                                     nn.ReLU(),
                                     nn.Conv2d(32, 32, (3, 3), padding=(1, 1)),
                                     nn.ReLU())

        self.deconv_5 = nn.ConvTranspose2d(32, 16, (2, 2), stride=(2, 2))
        self.conv_11 = nn.Sequential(nn.Conv2d(48, 32, (3, 3), padding=(1, 1)),
                                     nn.ReLU(),
                                     nn.Conv2d(32, 32, (3, 3), padding=(1, 1)),
                                     nn.ReLU())

        self.conv_12 = nn.Conv2d(32, 1, (1, 1))
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, x):
        x_1 = self.conv_1(x)
        x = self.pool(x_1)

        x_2 = self.conv_2(x)
        x = self.pool(x_2)

        x_3 = self.conv_3(x)
        x = self.pool(x_3)

        x_4 = self.conv_4(x)
        x = self.pool(x_4)

        x_5 = self.conv_5(x)
        x = self.pool(x_5)

        x = self.conv_6(x)

        x = self.deconv_1(x)
        x = torch.cat((x, x_5), dim=1)
        x = self.conv_7(x)

        x = self.deconv_2(x)
        x = torch.cat((x, x_4), dim=1)
        x = self.conv_8(x)

        x = self.deconv_3(x)
        x = torch.cat((x, x_3), dim=1)
        x = self.conv_9(x)

        x = self.deconv_4(x)
        x = torch.cat((x, x_2), dim=1)
        x = self.conv_10(x)

        x = self.deconv_5(x)
        x = torch.cat((x, x_1), dim=1)
        x = self.conv_11(x)

        return torch.sigmoid(self.conv_12(x))


class UNet1D(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.conv_1 = nn.Sequential(nn.Conv1d(in_channels, 32, (3,), padding=(1,)),
                                    nn.ReLU(),
                                    nn.Conv1d(32, 32, (3,), padding=(1,)),
                                    nn.ReLU())

        self.conv_2 = nn.Sequential(nn.Conv1d(32, 32, (3,), padding=(1,)),
                                    nn.ReLU(),
                                    nn.Conv1d(32, 32, (3,), padding=(1,)),
                                    nn.ReLU())

        self.conv_3 = nn.Sequential(nn.Conv1d(32, 64, (3,), padding=(1,)),
                                    nn.ReLU(),
                                    nn.Conv1d(64, 64, (3,), padding=(1,)),
                                    nn.ReLU())

        self.conv_4 = nn.Sequential(nn.Conv1d(64, 128, (3,), padding=(1,)),
                                    nn.ReLU(),
                                    nn.Conv1d(128, 128, (3,), padding=(1,)),
                                    nn.ReLU())

        self.conv_5 = nn.Sequential(nn.Conv1d(128, 256, (3,), padding=(1,)),
                                    nn.ReLU(),
                                    nn.Conv1d(256, 256, (3,), padding=(1,)),
                                    nn.ReLU())

        self.conv_6 = nn.Sequential(nn.Conv1d(256, 512, (3,), padding=(1,)),
                                    nn.ReLU(),
                                    nn.Conv1d(512, 512, (3,), padding=(1,)),
                                    nn.ReLU())

        self.deconv_1 = nn.ConvTranspose1d(512, 128, (2,), stride=(2,))
        self.conv_7 = nn.Sequential(nn.Conv1d(384, 256, (3,), padding=(1,)),
                                    nn.ReLU(),
                                    nn.Conv1d(256, 256, (3,), padding=(1,)),
                                    nn.ReLU())

        self.deconv_2 = nn.ConvTranspose1d(256, 64, (2,), stride=(2,))
        self.conv_8 = nn.Sequential(nn.Conv1d(192, 128, (3,), padding=(1,)),
                                    nn.ReLU(),
                                    nn.Conv1d(128, 128, (3,), padding=(1,)),
                                    nn.ReLU())

        self.deconv_3 = nn.ConvTranspose1d(128, 32, (2,), stride=(2,))
        self.conv_9 = nn.Sequential(nn.Conv1d(96, 64, (3,), padding=(1,)),
                                    nn.ReLU(),
                                    nn.Conv1d(64, 64, (3,), padding=(1,)),
                                    nn.ReLU())

        self.deconv_4 = nn.ConvTranspose1d(64, 16, (2,), stride=(2,))
        self.conv_10 = nn.Sequential(nn.Conv1d(48, 32, (3,), padding=(1,)),
                                     nn.ReLU(),
                                     nn.Conv1d(32, 32, (3,), padding=(1,)),
                                     nn.ReLU())

        self.deconv_5 = nn.ConvTranspose1d(32, 16, (2,), stride=(2,))
        self.conv_11 = nn.Sequential(nn.Conv1d(48, 32, (3,), padding=(1,)),
                                     nn.ReLU(),
                                     nn.Conv1d(32, 32, (3,), padding=(1,)),
                                     nn.ReLU())

        self.conv_12 = nn.Conv1d(32, 1, (1,))
        self.pool = nn.MaxPool1d(2)

    def forward(self, x):
        x_1 = self.conv_1(x)
        x = self.pool(x_1)

        x_2 = self.conv_2(x)
        x = self.pool(x_2)

        x_3 = self.conv_3(x)
        x = self.pool(x_3)

        x_4 = self.conv_4(x)
        x = self.pool(x_4)

        x_5 = self.conv_5(x)
        x = self.pool(x_5)

        x = self.conv_6(x)

        x = self.deconv_1(x, output_size=x_5.shape)
        x = torch.cat((x, x_5), dim=1)
        x = self.conv_7(x)

        x = self.deconv_2(x, output_size=x_4.shape)
        x = torch.cat((x, x_4), dim=1)
        x = self.conv_8(x)

        x = self.deconv_3(x, output_size=x_3.shape)
        x = torch.cat((x, x_3), dim=1)
        x = self.conv_9(x)

        x = self.deconv_4(x, output_size=x_2.shape)
        x = torch.cat((x, x_2), dim=1)
        x = self.conv_10(x)

        x = self.deconv_5(x, output_size=x_1.shape)
        x = torch.cat((x, x_1), dim=1)
        x = self.conv_11(x)

        return torch.sigmoid(self.conv_12(x))


if __name__ == "__main__":
    inp = torch.zeros((1, 3, 1400))
    model = UNet1D(in_channels=inp.shape[1])
    print(model(inp).shape)

