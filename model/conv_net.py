import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(self, channels, norm, residual):
        assert isinstance(channels, list)
        super().__init__()

        kernel_sz = [150, 40, 10, 3]
        stride = 4
        pad_mode = 'fair'

        self.blocks = []
        for i in range(1, len(channels)):
            self.blocks.append(
                nn.Sequential(
                    ConvLayer(channels[i-1], channels[i], kernel_sz[i-1], stride, pad_mode, norm),
                    ResBlock(channels[i], kernel_sz[i], norm, residual),
                )
            )
        self.fc = nn.Linear(channels[-1], 1)

    def forward(self, x):
        x = self.net(x)
        x = x.sum(dim=(2))
        x = self.fc(x)
        x = x.squeeze()
        return F.sigmoid(x)


class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size, norm, residual):
        super().__init__()
        self.conv1 = self.__make_conv_layer(channels, kernel_size, norm)
        self.conv2 = self.__make_conv_layer(channels, kernel_size, norm)
        self.residual = residual

    @staticmethod
    def __make_conv_layer(channels, kernel_size, norm):
        return ConvLayer(
            input_channels=channels,
            output_channels=channels,
            kernel_size=kernel_size,
            stride=1,
            padding='same',
            norm=norm
        )
        
    def forward(self, x):
        x0 = x
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        if self.residual:
            x = x + x0
        x = F.relu(x)
        return x


class ConvLayer(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride, padding, norm):
        super().__init__()
        assert norm in ('batch', 'layer')
        assert padding in ('fair', 'same')
        
        if padding == 'fair':
            pad = kernel_size - stride
        elif padding == 'same':
            pad = 'same'
        
        self.conv = nn.Conv1d(
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=pad
        )

        if norm == 'batch':
            self.norm = nn.BatchNorm1d(output_channels)
        elif norm == 'layer':
            self.norm = nn.LayerNorm(output_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return x


if __name__ == '__main__':
    net = ConvNet(149, 50, 15, 5, 'batch', residual=True)
    _x = torch.randn(32, 149, 1500)
    _y = net.forward(_x)
