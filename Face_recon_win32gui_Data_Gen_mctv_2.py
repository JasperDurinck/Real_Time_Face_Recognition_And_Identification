import cv2 as cv
from WinCap_Win32gui_Class import WindowCapture
from Train_data_get_manage_1_Class import Train_data_get_manage
import os
import torch
from facenet_pytorch import MTCNN
from PIL import Image
import tqdm 

def show_images_for_frame(img_cropped_list):
    
        for i in range(len(img_cropped_list)):
            if i != 0:
                PILimg_path = img_item+'_detected_face_'+str(i+1)+'.png'
            elif i == 0:
                PILimg_path = img_item+'_detected_face.png'
            Image.open(PILimg_path).show()

# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

name = input("Give Name: ")

dir_path = 'C:/Users/PC/Documents/MLdatasets/Database_faces/Train/'
Train_data_get_manage.new_folder(dir_path,name)
timeframe = Train_data_get_manage.file_file_count(dir_path,name)

Frame_Freq = 1
Stop_img_amount = 200
Farme_i = Frame_Freq
prob_threshold = 0.8
Show_Images_Choice = False
Save_Images = True #None
keep_all = False

mtcnn = MTCNN(image_size=240, margin=0, min_face_size=40,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        select_largest=True, selection_method='probability', keep_all=keep_all, device=device)

WinCap = WindowCapture()

name_list = [] # list of names corrospoing to cropped photos
embedding_list = [] # list of embeding matrix after conversion from cropped faces to embedding matrix using resnet

#while input("Next(n): ") == 'n':
while len(embedding_list) <= Stop_img_amount:

    frame = WinCap.get_screenshot()

    if Farme_i >= Frame_Freq:

        if Save_Images is not None:
            img_item = dir_path+name+'/Frame_'+str(timeframe)+'_'+name+'_detected_face.png'
            img_cropped_list, prob_list = mtcnn(frame, return_prob=True, save_path=img_item)
        
            if Show_Images_Choice is True:
                show_images_for_frame(img_cropped_list, prob_list)

        
        face, prob = mtcnn(frame, return_prob=True) 
        if face is not None and prob>prob_threshold: 
            embedding_list.append(face.detach())
            print("added with prob: "+str(prob)) 
            #name_list.append(idx_to_class[idx] 

        Farme_i = 0
        timeframe +=1

    Farme_i +=1

cv.destroyAllWindows()

# loading data.pt file
save_tensors_path = str(dir_path+'/'+name+'/tensors_for_training_save.pt')
load_data = []

if Train_data_get_manage.file_are_u_there(save_tensors_path) is True:
    load_data = torch.load(save_tensors_path)
    print('Amount of tensors already in train data: ' + str(len(load_data)))

# save data
data = load_data + embedding_list
print('Amount of tensors added to train data: ' + str(len(embedding_list)))
print('Total tensors in train data: ' + str(len(data)))
torch.save(data, save_tensors_path) # saving data.pt file