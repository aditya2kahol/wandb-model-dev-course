import torch.nn as nn

def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
              nn.BatchNorm2d(out_channels), 
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

class ResNet9(nn.Module):
    def __init__(self, in_channels, num_classes, dropout = 0.2):
        super().__init__()
        #in: 3,80,80
        self.conv1 = conv_block(in_channels, 32)
        #out: 32,80,80
        self.conv2 = conv_block(32, 64, pool=True)
        #out: 64,40,40
        self.res1 = nn.Sequential(conv_block(64, 64), conv_block(64, 64))
        #out: 64,40,40
        self.conv3 = conv_block(64, 128, pool=True)
        #out: 128,20,20
        self.conv4 = conv_block(128, 256, pool=True)
        #out: 256,10,10
        self.res2 = nn.Sequential(conv_block(256, 256), conv_block(256, 256))
        #out: 256,10,10
        
        self.classifier = nn.Sequential(nn.MaxPool2d(5), #256,2,2 
                                        nn.Flatten(), #256*2*2,
                                        nn.Dropout(dropout),
                                        nn.Linear(256*2*2, num_classes))
        
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out
