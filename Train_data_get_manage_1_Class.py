import os
import torch
from tqdm import tqdm

class Train_data_get_manage:

    def __init__(self, dir_path):
        #Directory path to where all directory of training data are
        self.dir_path = dir_path

    def new_folder(self, name): #checks if the folder already exits or creats it
        path = self+name
        existence = os.path.exists(path)
        if existence is False:
            os.mkdir(path)
            print("Directory '% s' created" % name)
            print(path)
        elif  existence is True:
            print("Folder already exists")

    def is_folder_there(self): #checks if the folder already exits or creats it
        path = str(os.path.dirname(self))
        existence = os.path.exists(path)
        if existence is False:
            os.mkdir(path)
            print("Directory did not exit yet, so was created: " + path)

    def new_file(self, Extra_directory=''): #checks if the file already exits or need to be created or be overwritten, returns True or False
        name_file = input("Give name and type of file you want to safe, (fileName.ExampleType): ")
        path_store_model = str(os.path.dirname(self)) + str(Extra_directory) + str(name_file)
        existence = os.path.exists(path_store_model)
        if existence is False:
            return existence, path_store_model
        elif  existence is True:
            print("File already exists")
            choice00 = False
            while choice00 != 'Yes' or 'No':
                choice00 = input("Do You Want To OVER WRITE the file? (Yes) or (No): ")
                if choice00 == 'Yes':
                    return False, path_store_model
            return existence, path_store_model

    def file_are_u_there(self, Extra_directory=''): #checks if the file already exits or need to be created or be overwritten, returns True or False
        path_file = self + Extra_directory
        existence = os.path.exists(path_file)
        return existence

    def file_file_count(self, name, file_count=0): #counts the file in a diretory
        # Iterate directory
        for files in os.listdir(self+name):
            # check if current path is a file
            if os.path.isfile(os.path.join(str(self+name), files)):
                file_count += 1
        print('File file_count:', file_count)
        
        return file_count
    
    def  Get_Data_And_Labels_And_directorys(self):
        list_of_Classes_directory_path = []
        list_of_Classes_data_path = []
        list_of_Classes_names = []
        List_of_Tensors = []
        List_number_tensors_each_class = []

        rootdir = self
        for file in os.listdir(rootdir):
            direct = os.path.normpath(str(os.path.join(rootdir, file)))
            if os.path.isdir(direct):
                list_of_Classes_names.append(file)
                list_of_Classes_directory_path.append(direct)
                train_data_path = str(direct+'/'+"tensors_for_training_save.pt")
                print(train_data_path)
                list_of_Classes_data_path.append(train_data_path)
    
        for data in list_of_Classes_data_path:
            # loading data.pt file
            save_tensors_path = str(data)
            load_data = []
            load_data = torch.load(save_tensors_path)
            List_of_Tensors += [load_data]
            List_number_tensors_each_class.append(load_data)
        
        


        return List_of_Tensors, list_of_Classes_names, List_number_tensors_each_class, list_of_Classes_data_path, list_of_Classes_directory_path


    def object_ID_tensors_though_NN_embeding_matrix_class_list(list_of_Classes_names, List_of_Tensors_list, resnet):
    
        name_list = []
        embedding_list = [] # list of embeding matrix after conversion from cropped faces to embedding matrix using resnet

        for Class_number in tqdm(range(len(list_of_Classes_names))):
        
            for tensor in List_of_Tensors_list[Class_number]:
                emb = resnet(tensor.unsqueeze(0)) 
                embedding_list.append(emb.detach()) 
                name_list.append(list_of_Classes_names[Class_number])

        data = [embedding_list, name_list] 

        return data


