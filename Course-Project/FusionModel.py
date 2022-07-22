#import important modules.
import torch
import torch.nn as nn


#model construction
def convblock(in_channels, out_channels, kernel_size = 3, padding = 1, stride = 1, pool = None):
    layers = [nn.Conv2d(in_channels, 
                        out_channels, 
                        kernel_size = kernel_size, 
                        padding = padding,
                        stride = stride,
                        bias = False
                        ),
             nn.BatchNorm2d(out_channels),
             nn.ReLU(inplace = True)]
    if pool:
        layers.append(nn.MaxPool2d(pool))
    return nn.Sequential(*layers)

def convTblock(in_channels, out_channels, kernel_size = 3, padding = 1, stride = 1):
    layers = [nn.ConvTranspose2d(in_channels, 
                        out_channels, 
                        kernel_size = kernel_size, 
                        padding = padding,
                        stride = stride),
             nn.PReLU(init = -0.05)]
    return nn.Sequential(*layers)

class Fusion(nn.Module):
    """
        Exp: Encoder-Decoder architecture with skip connections.
    """
    def __init__(self, drop = 0):
        super().__init__()
        #in: 3,500,500
        
        self.encoder1 = nn.Sequential(
            convblock(3, 32, kernel_size = 5, padding = 2, pool = 2),
            convblock(32,64, kernel_size = 3, padding = 1, pool = None)
            ) #out: 64,250,250

        #skip connection from encoder to decoder
        self.res_e = convblock(64,64) #out: 64,250,250
        self.res_d = convblock(64, 64, pool = 5) #out: 64,50,50
                                          
        self.encoder2 = nn.Sequential( 
            convblock(64, 128, kernel_size = 2, padding = 1, stride = 2, pool = None), #out: 128,26,26
            convblock(128,128, kernel_size = 3, padding = 2, stride = 3, pool = 2), #out: 128,5,5
            nn.Flatten(), #out: 128*5*5 = 3200
            nn.Linear(128*5*5,800),#out: 800
            nn.Dropout(p = drop)
        )

        self.linear = nn.Linear(1600,128*5*5) #out: 128*5*5
        
        #now change the view of this to 128,5,5
        
        self.decoder1 = nn.Sequential(convTblock(128, 128, kernel_size = 2, stride = 7, padding = 2),
                                      convTblock(128, 64, kernel_size = 2, stride = 2, padding = 1))
        
        self.decoder2 = convTblock(64, 64, kernel_size = 7, stride = 5, padding = 1)

        self.decoder3 = nn.Sequential(
            convTblock(64,32, kernel_size = 4, stride = 2, padding = 1),
            convTblock(32,3, kernel_size = 3, stride = 1, padding = 1)
        )
    
    def forward(self, img1, img2):
        """
            img1 -> 3,500,500
            img2 -> 3,500,500
        """                                         
        #apply encoder1 to image1 and image2
        outI1 = self.encoder1(img1)
        outI2 = self.encoder1(img2)

        #add skip connections within encoder
        res1a = self.res_e(outI1) + outI1
        res1b = self.res_e(outI2) + outI2

        #skip connection from encoder to decoder
        res2a = self.res_d(res1a)
        res2b = self.res_d(res1b)

        #apply encoder2
        outI1 = self.encoder2(res2a) #out: 800,
        outI2 = self.encoder2(res2b) #out: 800,                                 
        
        #concat the encoded images.
        fused_out = torch.hstack([outI1,outI2]) #out: 1600
        
        #bottle-neck
        fused_out = self.linear(fused_out) #out: 128*5*5
        
        #reshape to 128,5,5
        fused_out = fused_out.view(-1,128,5,5) #out: 128,5,5
        
        #decode fused image
        fused_out = self.decoder1(fused_out) + res2a + res2b #out: 64,50,50 
        fused_out = self.decoder2(fused_out) + res1a + res1b #out: 64,250,250
        fused_out = self.decoder3(fused_out) #out: 3,500,500
        
        return fused_out
