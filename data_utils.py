import scipy.io as scio
import numpy as np 
from skimage import io 
import cv2 
import matplotlib.pyplot as plt 
from skimage import color 
import os 
import re 

img_folder_path = '/data_disk1/Synth/SynthText'
charBB_path = '/data_disk1/Synth/SynthText/charBB.npy'
imnames_path = '/data_disk1/Synth/SynthText/imnames.npy'
txt_path = '/data_disk1/Synth/SynthText/txt.npy'
heatmap_path = '/root/craft_re/sample_image/gaussian_heatmap/sample_heatmap.npy'

def generate_gt(img , heatmap  , bbox_cor): 

    #bbox_cor = num_bbox , 4point , (x,y)

    row = img.shape[0]
    col = img.shape[1]

    gt_image = np.zeros((row, col) , np.float32)

    original_cor = np.float32(  [ [0,0] , [0, heatmap.shape[0]] , [heatmap.shape[1],0] , [heatmap.shape[1],heatmap.shape[0]]  ] )

    for bbox in bbox_cor: 

        bbox = np.float32( [bbox[0] , bbox[1] , bbox[3] , bbox[2] ])

        mtrx = cv2.getPerspectiveTransform(original_cor , bbox)
        dst = cv2.warpPerspective( heatmap, mtrx , (col, row) )

        gt_image += dst 
    
    return gt_image 


#txt 공백제거 
def rm_blank(x): 
    tmp = []
    for word in x: 
        tmp.extend(re.split(' |\n', word))
    # <U19 타입 
    tmp = np.array(tmp , dtype = '<U19')
    tmp = np.delete(tmp, np.where(tmp==''))
    return tmp
    
#txt 파일 공백제거 후 현재위치에 저장 
def save_txt_without_blank(txt_full):
    txt_without_blank = np.empty(txt_full.shape[0] , dtype=object)
    for index, txt in enumerate(txt_full):
        txt_without_blank[index] = rm_blank(txt)
    
    return txt_without_blank
    #np.save(os.getcwd + '/' + 'txt_without_blank' , txt_without_blank)

# txt_without_blank -> Affinity_bounding_box index 
# 이 함수를 통해 나온 어레이에 있는 index , index+1 사이의 Affinity를 구하면된다. 
def make_Affinity_index(txt_without_blank):
    Affinity_index_list = np.empty(txt_without_blank.shape[0] , dtype=object)
    for n , txt in  enumerate(txt_without_blank):
        index_list = np.array([] , dtype=np.int32)
        index = 0 
        for word in txt: 
            if len(word) == 1: 
                index += 1 
            else: 
                for _ in range(len(word) - 1 ):
                    index_list = np.append(index_list , index)
                    index += 1 
                index += 1 
        Affinity_index_list[n] = index_list
    return Affinity_index_list 

#Affinity box 의 좌표 생성 
#사각형의 대각선 교점을 구해야하는데 직접 구하면 너무 복잡해서 
#4개 좌표의 평균값으로 근사한다. -> 사각형이 정/직사각형, 평행사변형 이면 일치한다. 
def affinity_box(bbox1 , bbox2):
    #Aproximation of center of square 
    center1 = np.mean(bbox1 , axis = 0 )
    center2 = np.mean(bbox2 , axis = 0)

    #4 triangular 
    left_top = (bbox1[0] + bbox1[1] + center1)/3
    left_bottom = (bbox1[2] + bbox1[3] + center1)/3
    right_top = (bbox2[0] + bbox2[1] + center2)/3
    right_bottom = (bbox2[2] + bbox2[3] + center2)/3

    return np.array([left_top , right_top , right_bottom , left_bottom])

#Affinity_index를 받아서 Affinity_box 좌표를 저장 
#affinity_index = (num of images , )
#charBB = (num of images, )
def affinity_box_list(affinity_index , charBB):
    #empty affinityBB
    affinityBB = np.empty(affinity_index.shape[0], dtype=object)

    for image_index in range(affinity_index.shape[0]):

        image_BB = np.empty((affinity_index[image_index].shape[0],4,2) , dtype=object)

        t_charBB = charBB[image_index].transpose(2,1,0)

        for index , affinity_index_ in enumerate(affinity_index[image_index]):

            aff_bbox = affinity_box(t_charBB[affinity_index_] , t_charBB[affinity_index_ + 1])

            image_BB[index] = aff_bbox

        affinityBB[image_index] = image_BB

    return affinityBB

#GT를 만들때 이미지를 전부 불러올 필요없이
#이미지의 크기만 있으면 되기 때문에 이미지의 크기만 저장해두는 함수 
#img_size_list.npy 
def make_size_list(img_folder_path, imnames): 
    img_size_list = np.empty(imnames.shape[0], dtype=object)

    for img_index in range(imnames.shape[0]):
        img_name = imnames[img_index][0]

        #im_size -> tuple (row,col)
        img_size = cv2.imread(os.path.join(img_folder_path, img_name)).shape[0:2] #channel -> 3 
        img_size_list[img_index] = img_size

    return img_size_list 


def Gray2RGB(image): 

    # H x W -> H x W x C 
    image = image*255
    image = image.astype(np.uint8)

    image = cv2.applyColorMap(image, cv2.COLORMAP_JET)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


if __name__ =='__main__':

    print("Data load")

    charBB = np.load(charBB_path, allow_pickle=True) 
    imnaems = np.load(imnames_path, allow_pickle=True)
    txt = np.load(txt_path, allow_pickle=True)

    heatmap = np.load(heatmap_path)

    #charBB.shape -> (1,~~~)
    #txt.shape -> (1, ~~~) 
    charBB = charBB[0]
    txt = txt[0]

    print('Data load Complete')

    #txt 파일에서 공백을 제거한다 
    print('delete blank in txt.npy')
    txt_without_blank = save_txt_without_blank(txt)
    np.save('./txt_without_blank' , txt_without_blank)
    print('complete')

    #Affinity Score를 구해야하는 charBB의 index를 저장한다. 
    #ex) [1,2,7,10] -> 1~2 , 2~3, 7~8, 10~11의 Affinity를 구해야한다. 
    print('make Affinity index')
    Affinity_index = make_Affinity_index(txt_without_blank)
    np.save('./Affinity_index' , Affinity_index)
    print('complete')

    #Affinity_index, charBB를 이용해서 char_aff_BB를 만든다 
    print('make affinity score bounding box')
    char_aff_BB = affinity_box_list(Affinity_index , charBB)
    np.save('./char_aff_BB' , char_aff_BB)
    print('complete')

