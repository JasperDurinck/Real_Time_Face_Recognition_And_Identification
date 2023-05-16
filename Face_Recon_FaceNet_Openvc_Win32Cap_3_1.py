# importing libraries
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import cv2 as cv
from WinCap_Win32gui_Class import WindowCapture
import numpy as np
from time import time
from Image_Confidance_function import Face_ID_combined_Tensor, list_to_device

# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# initializing MTCNN and InceptionResnetV1 
mtcnn0 = MTCNN(image_size=240, margin=0, keep_all=False, min_face_size=40, device=device) # keep_all=False, only returns one found face
mtcnn = MTCNN(image_size=240, margin=0, keep_all=True, min_face_size=40, device=device) # keep_all=True, returns all found faces
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# loading data.pt file
load_data = torch.load('C:/Users/PC/Documents/MLdatasets/Database_faces/Models/model.pt')

list_perons_embedding = load_data[0]
embedding_list_cuda = []
for tens in list_perons_embedding:
    embedding_list_cuda.append(tens.to(device))
list_perons_embedding = embedding_list_cuda
name_list = load_data[1] 

#Capture frame Win32 based class
WinCap = WindowCapture()

def ID_in_frame(frame, name, x1, y1, x2, y2, min_dist, Frame_i):
    frame = cv.rectangle(frame, (int(x1), int(y1)) , (int(x2), int(y2)), (0,255,0), 1)
    frame = cv.putText(frame, name, (int(x2), int(y2)), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),1, cv.LINE_AA)
    frame = cv.putText(frame, str(round(min_dist, 2)), (int(x2), int(y2-20)), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),1, cv.LINE_AA)
    print(name, min_dist, Frame_i)


def Win32Cap_Screen_Cap_Detect():

    Frame_Freq_adjust = 0 #If we reconize a face we wait #Frame_Freq_adjust frames before again trying to reconize
    detected_i = 0 #Number of frames that we show the same detection
    Frame_i = Frame_Freq_adjust #frame number, within the Frame_Freq_adjust loop
    Face_detected = False #Is face reconized, True or False
    prob_threshold = 0.2 #Threshold (needs to be greater than #prob_thershold) for the probability of a face being in a frame (says nothing about person recongization)
    min_dist_threshold = 0.86 #Threshold for the min_dist for which a name is assignt to the face (min_dist must be lower than min_dist_threshold)

    while True:
        #Capture frame with the Win32gui based class
        loop_time = time()
        frame = WinCap.get_screenshot()
        img = frame.copy()

        if Frame_i >= Frame_Freq_adjust: #if statement for configuration how many frames are put through the model (Could be needed for beter fps and lower resource cost)

            #Here we reset the frame counter and detection counter
            Frame_i =0
            detected_i=0
            name_list_IDs = []
            min_dist_list_IDs = []

            img_cropped_list, prob_list = mtcnn(img, return_prob=True)  #Here we put the frame through the mtcnn net for detecting a face in the image 

            if img_cropped_list is not None: #If mtcnn detects face(s) like figures in the frame we next run the cropped version(s) through the trained resnet

                for i, prob in enumerate(prob_list): #if the cropped images hace a high enough prob we run it through the resnet for face recongisiton 
                    if prob>prob_threshold:
                        emb = resnet(img_cropped_list[i].unsqueeze(0)).detach().to(device) #resnet prediction for face recongisiton
                        dist_list = [] # list of matched distances, minimum distance is used to identify the person

                        for idx, emb_db in enumerate(list_perons_embedding): #Highest predicted face get selected
                            dist = ((torch.dist(emb, emb_db))).item() #/(list_normilizers[idx])
                            #dist = np.sqrt(dist)
                            #dist = np.log(dist) - list_Condifance_Threshold_list[idx]
                            #dist = (dist**2) - list_Condifance_Threshold_list[idx]
                            print(dist)
                            dist_list.append(dist)

                        min_dist = min(dist_list) # get minumum dist value
                        min_dist_idx = dist_list.index(min_dist) # get minumum dist index
                        name = name_list[min_dist_idx] # get name corrosponding to minimum dist
                        #min_dist_threshold = list_Condifance_Threshold_list[min_dist_idx]
                        
                        if min_dist<min_dist_threshold:
                            
                            Face_detected = True #if We detect a Face we set the variable to true so that we can keep the prediction and location of a set amount of frames if needed for preforman improment
                            boxes= mtcnn.detect(img) #here we save the cropped image location for rectange
                            x1, y1, x2, y2 = boxes[0][i] 
                            name_list_IDs.append(name)
                            min_dist_list_IDs.append(min_dist)
                            ID_in_frame(frame, name, x1, y1, x2, y2, min_dist, Frame_i)

        if Face_detected == True and Frame_Freq_adjust != 0 and Frame_i != 0: #if we did detect known face, we keep that location and name for few framse on same place 
           for j, name_face in enumerate(name_list_IDs):
            name, min_dist,  = name_list_IDs[j], min_dist_list_IDs[j]
            x1, y1, x2, y2 = boxes[0][j]
            ID_in_frame(frame, name, x1, y1, x2, y2, min_dist, Frame_i)
           
           detected_i += 1 
           
           if detected_i >= Frame_Freq_adjust:
            Face_detected = False
        
        cv.imshow("IMG", frame)  

        # debug the loop rate
        print('FPS {}'.format(1 / (time() - loop_time)))
        loop_time = time()

        k = cv.waitKey(1)
        if k%256==27: # ESC
           print('Esc pressed, closing...')
           break
           
        Frame_i+=1 #Add 1 to frame counter 
        if Face_detected == False and Frame_Freq_adjust != 0:
            Frame_i = Frame_Freq_adjust

    cv.destroyAllWindows()

Win32Cap_Screen_Cap_Detect()