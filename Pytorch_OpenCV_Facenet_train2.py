# importing libraries

from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
import tkinter.filedialog as fd
from tqdm import tqdm 
from Train_data_get_manage_1_Class import Train_data_get_manage

#parameters
prob_parameter = 0.0
image_size =240
margin = 0
min_face_size = 40

#storing new file location
directory_train_data = 'C:/Users/PC/Documents/MLdatasets/Database_faces/Train'
extra_directory = '/Models/'
choose_name=True
while choose_name is True:
    choose_name, path_train_data = Train_data_get_manage.new_file(directory_train_data, str(extra_directory))
Train_data_get_manage.is_folder_there(path_train_data)

# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

InceptionResnetV1 
resnet = InceptionResnetV1(pretrained='vggface2').eval()

List_of_Tensors_list, list_of_Classes_names, List_number_tensors_each_class, list_of_Classes_data_path, list_of_Classes_directory_path = Train_data_get_manage.Get_Data_And_Labels_And_directorys(directory_train_data)

data = Train_data_get_manage.object_ID_tensors_though_NN_embeding_matrix_class_list(list_of_Classes_names, List_of_Tensors_list, resnet)       

# save data
torch.save(data, path_train_data) # saving data.pt file
print("done")