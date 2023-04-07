import torch
import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt

def list_to_device(list_current_device):
    # Define the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    list_other_device = []
    for i in list_current_device:
        list_other_device.append(i.to(device))
    return list_other_device

def Face_ID_combined_Tensor(load_data):

    name_list = load_data[1] #list of names ordered by name, and respective to the imagens 
    
    unique_names = np.unique(name_list) #Get respective ordered list of names
    number_img_name_list = []   #list for the amount of images for each name

    for name in unique_names:
        number_img_name_list.append(name_list.count(name))

    print(unique_names)

    list_perons_embedding = []
    list_tensors_per_ID = []

    start_index = 0
    end_index=0

    for name_index in range(len(unique_names)):
        end_index += number_img_name_list[name_index]
        list_tensors_per_ID.append(load_data[0][start_index:end_index])
        list_perons_embedding.append(torch.cat(load_data[0][start_index:end_index]))
        start_index = end_index
    
    name_list = unique_names

    def nomlizery(x):
        #normlizer = 1.1028*np.log(x)-0.4959
        normlizer = x
        return torch.tensor(normlizer)

    list_normilizers = []

    for num in number_img_name_list:
        list_normilizers.append(nomlizery(float(num)))


    list_Condifance_Threshold_list = []

    for i in range(len(list_normilizers)):

        list_dists = []

        for ij in list_tensors_per_ID[0]:
            dist = np.sqrt(torch.dist(list_perons_embedding[i], ij).item())
            norm_coef = list_normilizers[i].item()
            dist_adj = dist / norm_coef
            dist_adj = dist_adj
            list_dists.append(dist)
            #print(dist)


            
            
        # # generate data
        # data =list_dists
        # print(np.std(list_dists))
        # print(np.mean(list_dists))
        # ax = None
        # # plotting a histogram
        # ax = sns.distplot(data,
        #                 bins=20,
        #                 kde=True,
        #                 color='red',
        #                 hist_kws={"linewidth": 15,'alpha':1})
        # ax.set(xlabel='Normal Distribution', ylabel='Frequency')
        # plt.show()
        # plt.close()

    
        list_Condifance_Threshold_list.append((np.mean(list_dists)))#+4*(np.std(list_dists)))
    

    return list_to_device(list_perons_embedding), name_list, list_to_device(list_normilizers), list_tensors_per_ID, list_Condifance_Threshold_list

