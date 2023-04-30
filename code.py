import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import cv2
import time
from google.colab.patches import cv2_imshow
import os

from sklearn.cluster import SpectralClustering

# pytorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
#from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torch.utils.data import TensorDataset, DataLoader
import torchvision

def toy_image(imgSz):
  R_ = np.array([255,0,0]) / 255
  G_ = np.array([0,255,0]) / 255
  B_ = np.array([0,0,255]) / 255

  v1 = [R_, G_, B_, R_ + G_]
  v2 = [1, 2, 3, 4]

  img = np.zeros((*imgSz ,3),dtype=np.float32)
  ref = np.zeros(imgSz)

  arr = [[img, *imgSz, v1], [ref, *imgSz, v2]]
  for [im, H, W, v] in arr:
    x1, y1, r1 = W // 6 , H // 4, W // 8
    x2, y2, r2 = 3*W // 4 , H // 4, H //4
    x3, y3, r3 = W // 4 , 3*H // 4, W // 5
    x4, y4, r4 = 3*W // 4 , 3*H // 4, H // 6

    X,Y = np.meshgrid(range(W), range(H))

    im[(X-x1)**2 + (Y-y1)**2 <= r1**2] = v[0]
    im[(X-x2)**2 + (Y-y2)**2 <= r2**2] = v[1]
    im[(X-x3)**2 + (Y-y3)**2 <= r3**2] = v[2]
    im[(X-x4)**2 + (Y-y4)**2 <= r4**2] = v[3]

  return img, ref
  
  
class splitPatches:
  def __init__(self, sz, st):
    self.hig = sz[0]
    self.wid = sz[1]
    self.st = st
  
  def __call__(self, img, ref):
    H, W = img.shape[:2]
    h = (H - (self.hig - self.st)) // self.st
    w = (W - (self.wid - self.st)) // self.st
    d = 3 if len(img.shape) == 3 else 1
    imgSplit = np.zeros((h,w, self.hig, self.wid, d), dtype=np.float32)
    refSplit = np.zeros((h,w), dtype=np.int32)
    for i in range(h):
      for j in range(w):
        imgSplit[i, j] = img[i*self.st : i*self.st + self.hig, j*self.st : j*self.st + self.wid]
        refSplit[i, j] = ref[i*self.st + (self.hig // 2), j*self.st + (self.wid //2)]
    return np.squeeze(imgSplit.reshape(h*w,*imgSplit.shape[2:])) , refSplit.reshape(h*w,*refSplit.shape[2:])


def histogram(patches, bins=32):
  pt_sz = patches.shape[0]
  patch = patches.reshape(pt_sz, -1)
  hist = np.zeros((pt_sz, bins))
  min = np.min(patch)
  max = np.max(patch)
  patch = (patch - min) / (max - min)
  patch = (255 * patch) // (256/bins)
  for i, p in enumerate(patch):
    for pix in p:
      hist[i][int(pix)] += 1
    
  return hist


class AE_default(nn.Module):
    
    def __init__(self, input_dim=28*28, hidden_dim=256, latent_dim=10):
        super(AE_default, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # define the encoder
        self.encoder = nn.Sequential(nn.Linear(self.input_dim, self.hidden_dim),
                                     nn.ReLU(), nn.Linear(self.hidden_dim, self.hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(self.hidden_dim, self.latent_dim))
        
        # define decoder
        self.decoder = nn.Sequential(nn.Linear(self.latent_dim, self.hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(self.hidden_dim, self.hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(self.hidden_dim, self.input_dim),
                                     nn.Sigmoid())
        
    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
        
    def latent(self, x):
        return self.encoder(x)
        

class AEtrainer:

  def __init__(self, hp):
    self.AE = hp['AE']
    self.hp = hp

    self.crt = nn.BCELoss()
    self.opt = torch.optim.Adam(self.AE.parameters(), lr = hp['lr'])

    self.dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  
  def train(self, patches):
    patch = torch.from_numpy(patches)
    patch = patch.view(patches.shape[0], -1)
    ds = TensorDataset(patch)
    ld = DataLoader(ds, self.hp['batch_size'], shuffle=True)
    for ep in range(self.hp['epochs']):
      for [bt] in ld:
        bt.to(self.dev)
        out = self.AE(bt)
        loss = self.crt(out, bt)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
      if (ep % 20 == 19):
        print("epoch {:3d}/{:3d} - Loss {:.4f}".format(ep, self.hp['epochs'], loss.item()))

  def latent(self, patches):
    patch = torch.from_numpy(patches)
    patch = patch.view(patches.shape[0], -1)
    return self.AE.latent(patch)

  def __call__(self, patches):
    self.train(patches)
    return self.latent(patches)

  def rec(self, patches):
    patch = torch.from_numpy(patches)
    patch = patch.view(patches.shape[0], -1)
    return self.AE(patch)

  def show(self, patches):
    patch = torch.from_numpy(patches)
    sz = patch.size()
    patch = patch.view(patches.shape[0], -1)
    rec =self.AE(patch)

    plt.subplot(211)
    plt.imshow(patches[0])
    plt.subplot(212)

    rec = torch.reshape(rec, sz).detach().numpy()
    plt.imshow(rec[0])


def defaultDict(ptz=32):
  dct = {'name': 'imageDir', 
         'patchSize': (ptz,ptz),
         'patchStep': ptz,
         # AE model
         'AE': AE_default(3*ptz*ptz, ptz , ptz // 4),
         'lr': 0.001,
         'epochs' : 100,
         'batch_size': 128}
  return dct




  

splt = splitPatches((28,28),28)
imgCar = cv2.cvtColor(cv2.imread('./images/car.jpeg'), cv2.COLOR_BGR2RGB) / 255

hp = defaultDict(28)
hp['patchStep'] = 8
hp['lr'] = 0.0005
hp['epochs'] = 300
AEt = AEtrainer(hp)

patches, r= splitPatches((28,28), 8)(imgCar, imgCar[:,:,0])
m,n = patches.shape[0:2]
patchesList = patches.reshape(m*n, 28,28,3)
AEt(patchesList)


patches, r= splt(imgCar, imCar[:,:,0])
m,n = patches.shape[0:2]
patchesList = patches.reshape(m*n, 28,28,3)
lt = AEt.latent(patchesList)
print(lt.size())

lt = lt.detach().numpy()
lt = lt.reshgape(m,n,-1)
np.save('./ae_d_28_28_lt', lt)

rec = AEt.rec(patchesList).detach().numpy()
rec = rec.reshape(m,n,28,28,3)
np.save('./ae_d_28_28_rec', rec)

hist = histogram(patchesList)
hist = hist.reshape(m,n,-1)
print(hist.shape)

np.save('./hist_28_28_32', hist)