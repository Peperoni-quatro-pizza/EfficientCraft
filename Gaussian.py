import cv2 
from skimage import io 
import timeit 
import numpy as np 
from math import exp, sqrt, pi
import matplotlib.pyplot as plt 

#Gaussian density function 
standardGaussian = lambda x : exp(-(1/2)*((x)**2))/(sqrt(2*pi))
scaledGaussian = lambda x : exp(-(1/2)*(x**2))

#Parameter 
img_size = 512 
center = img_size/2 
thred = 3.5 

#l2 norm 
def l2_norm_(x1, x2=np.array([img_size/2 , img_size/2])):
    return np.linalg.norm(x1-x2)

#vectorize 
vec_standardGaussian = np.vectorize(standardGaussian)
vec_scaledGaussian = np.vectorize(scaledGaussian)

#GrayScale Image generation -> data type uint8 
isotropic_Gray_Image = np.zeros((img_size, img_size), np.float32)
for x in range(img_size):
    for y in range(img_size):
       isotropic_Gray_Image[x][y] = np.linalg.norm(np.array([x-center , y-center]))

isotropic_Gray_Image /= np.max(isotropic_Gray_Image)
isotropic_Gray_Image *= thred


def main(img_size = img_size, thred = 3.5, test = False):

    #GrayScale Image generation -> data type uint8 
    isotropic_Gray_Image = np.zeros((img_size, img_size), np.float32)
    for x in range(img_size):
        for y in range(img_size):
            isotropic_Gray_Image[x][y] = np.linalg.norm(np.array([x-center , y-center]))

    isotropic_Gray_Image /= np.max(isotropic_Gray_Image)
    isotropic_Gray_Image *= thred

    #Save 
    heatmap = vec_scaledGaussian(isotropic_Gray_Image)
    if test == False: 
        return heatmap

    else: 
        np.save('./heatmap' , heatmap)
        GM = vec_scaledGaussian(isotropic_Gray_Image)*255
        GM = GM.astype(np.uint8)
        GM = cv2.applyColorMap(GM,cv2.COLORMAP_JET)

        #cv2 : BGR, plt : RGB -> 조심조심 
        GM = cv2.cvtColor(GM, cv2.COLOR_BGR2RGB)

        print('Mean of prob : {}'.format(np.mean(GM.flatten())))
        io.imshow(GM)


if __name__ == '__main__': 

    #make heatmap img as npy file 
    main(img_size = img_size, thred = thred, test = False)
