import face_recognition
import os
import pandas as pd
import numpy as np
import imageio
from skimage.transform import resize
from skimage.draw import polygon2mask
from utils import get_key_points, get_key_points_mouth, get_key_points_eyes, _get_value
import argparse

if __name__ == '__main__':
    ## change your file directory here

    parser = argparse.ArgumentParser()
    parser.description = "please enter png file path, bmt file path and heat table path ..."
    parser.add_argument("png_path",help="png file path", type=str, default="./test_face/png/")    
    parser.add_argument("bmt_path",help="bmt file path", type=str, default="./test_face/BMT1/")    
    parser.add_argument("heat_table",help="heat table path", type=str, default= "./test_face/excel/")    
    args = parser.parse_args()
    png_path = args.png_path
    bmt_path = args.bmt_path
    heat_table =args.heat_table

    path_list = os.listdir(png_path)
    keys = ["nose_bridge", "nose_tip"]

    with open('tf.transform') as f:
        tf = f.read()
    tf = tf.split("\n")
    center_x = int(tf[0])
    center_y = int(tf[1])
    size = float(tf[2])

    data = pd.DataFrame(index=np.arange(0,len(path_list)))
    for i in range(0, len(path_list)):
        basename = os.path.basename(path_list[i])[:-4]
        print(basename)
        im = imageio.imread(os.path.join(png_path, path_list[i]))
        bmt_3c = imageio.imread(os.path.join(bmt_path, path_list[i][:-4]+'.BMT'))
        bmt = pd.read_excel(os.path.join(heat_table, path_list[i][:-4]+'.xlsx'),index_col=None, header=None ).values
        face_landmarks_list = face_recognition.face_landmarks(np.array(im))
        btm_resize = resize(bmt, (int(bmt.shape[0]*size),int(bmt.shape[1]*size)))
        btm_pad = np.zeros(im.shape[0:2])

        start_x = int(center_x-btm_resize.shape[0]/2)
        end_x = int(center_x+btm_resize.shape[0]/2)
        start_y = int((center_y-btm_resize.shape[1]/2))
        end_y = int(center_y+btm_resize.shape[1]/2)

        btm_pad[start_x:end_x, start_y:end_y] = btm_resize
        mask = np.zeros(im.shape[0:2])
        data.loc[i,'name']=basename
        if(len(face_landmarks_list)>0):
            for p in range(0, len(face_landmarks_list)):

                nose = get_key_points(face_landmarks_list[p])
                mask_nose = polygon2mask(im.shape[0:2], nose[:,[1,0]])
                tmp_nose = np.mean(_get_value(mask_nose*btm_pad))
                data.loc[i, str(p)+"_nose"]=tmp_nose

                mouth = get_key_points_mouth(face_landmarks_list[p])
                mask_mouth = polygon2mask(im.shape[0:2], mouth[:,[1,0]])
                tmp_mouth = np.mean(_get_value(mask_mouth*btm_pad))
                data.loc[i, str(p)+"_mouth"]=tmp_mouth

                eyes = get_key_points_eyes(face_landmarks_list[p])
                mask_eyes = polygon2mask(im.shape[0:2], eyes[:,[1,0]])
                tmp_eyes = np.mean(_get_value(mask_eyes*btm_pad))
                data.loc[i, str(p)+"_eyes"]=tmp_eyes
    data.to_csv("output.csv")