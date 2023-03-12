import numpy as np
import cv2

def split_image(img:np.array, size, jump, start = [0,0])->np.array:
  # extract params
  N, M, D = img.shape
  [n, m] = size
  [dn, dm] = jump
  [n0, m0] = start
  # make sure input are ok
  if n + n0 > N or m + m0 > M:
    raise Exception("split is bigger the input image")
  
  # calculate the loops size
  n_loop = int(np.floor((N-n0-n+dn)/dn))
  m_loop = int(np.floor((M-m0-m+dm)/dm))
  splited_image = np.zeros([n_loop, m_loop, n, m, D])
  for i in range(n_loop):
    for j in range(m_loop):
      splited_image[i,j,:,:,:] = img[n0+i*dn : n0+i*dn + n, 
                                     m0+j*dm : m0+j*dm + m,:]
  # reutn splited image
  return np.squeeze(splited_image)
