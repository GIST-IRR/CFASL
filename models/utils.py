import torch.nn as nn


class MLPEmbeddingLayer(nn.Module):
    def __init__(self, config):
        super(MLPEmbeddingLayer, self).__init__()
        self.embedding = nn.Linear(config.input_size, config.hidden_size[0])
        self.relu = nn.ReLU(True)

    def forward(self, input):
        output = self.embedding(input)
        output = self.relu(output)
        return output


class MLPFullyConnectedLayer(nn.Module):
    def __init__(self, hidden_input, hidden_output):
        super(MLPFullyConnectedLayer, self).__init__()
        self.dense = nn.Linear(hidden_input, hidden_output)
        self.relu = nn.ReLU(True)

    def forward(self, input):
        output = self.dense(input)
        output = self.relu(output)
        return output


class CNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, padding=0):
        super(CNNLayer, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=4, stride=2, padding=padding
        )
        # self.batch = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(True)

    def forward(self, input):
        output = self.conv(input)
        # output = self.batch(output)
        output = self.relu(output)
        return output


class CNNTrasnposedLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CNNTrasnposedLayer, self).__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=4, stride=2, padding=1
        )
        # self.batch = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(True)

    def forward(self, input):
        output = self.conv(input)
        # output = self.batch(output)
        output = self.relu(output)
        return output
