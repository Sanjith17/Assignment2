import random
import sys
import math
import imageio
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
MAX_DISPARITY = 30 # Set this to the maximum disparity in the image pairs you'll use
iter = 50
alpha = 10
w = 2

#update message after every iteration
def update_message(msg_u,msg_d,msg_l,msg_r,disp_cost,alpha):
    #Defining the message numpy array with the shape of height, width and Max_disparity. 
    #Initializing four such matries such that it refers outgoing messages in each direction 
    # for all the values of Max_disparity
    #Upwards
    msg_u = np.zeros(disp_cost.shape)
    #Downards
    msg_d = np.zeros(disp_cost.shape)
    #Left message
    msg_l = np.zeros(disp_cost.shape)
    #Right Message
    msg_r = np.zeros(disp_cost.shape)
    
    h,w,MAX_DISPARITY = disp_cost.shape
    
    #incoming messages 
    #Rolling the messages such that the new arrays represent the incoming messages 
    #to that pixel and disparity
    #rolling by 1 in axis 0 gives incoming message from the upper pixel and so on  
    incoming_U = np.roll(msg_u, 1,axis=0)
    incoming_L = np.roll(msg_l, 1, axis=1)
    incoming_D = np.roll(msg_d, -1, axis=0)
    incoming_R = np.roll(msg_r, -1, axis=1)

    #Calculating the messages going out of the pixel in a direction
    #adding the datacost and incoming messages from the directions other that the message is sent to 
    npqu = disp_cost + incoming_L + incoming_D + incoming_R
    npql = disp_cost + incoming_U + incoming_D + incoming_R
    npqd = disp_cost + incoming_L + incoming_U + incoming_R
    npqr = disp_cost + incoming_L + incoming_D + incoming_U

    
    #Finding the minimum message along the diurection of disparity
    spqU = np.amin(npqu, axis=2)
    spqL = np.amin(npql, axis=2)
    spqD = np.amin(npqd, axis=2)
    spqR = np.amin(npqr, axis=2)

    #Combiining the V function of Potts function with the messages and datacost
    #and returning the result along each disparity
    for p in range(MAX_DISPARITY):
        msg_u[:,:,p]= np.minimum(npqu[:,:,p], alpha+spqU)
        msg_d[:,:,p]= np.minimum(npqd[:,:,p], alpha+spqD)
        msg_l[:,:,p]= np.minimum(npql[:,:,p], alpha+spqL)
        msg_r[:,:,p]= np.minimum(npqr[:,:,p], alpha+spqR)

    return msg_u, msg_d, msg_l, msg_r

#normalize messages 
def normalize_message(msg_u,msg_d,msg_l,msg_r):
    #normalizing using the avg. 
    #Finding the average of the messages along the disparity and subtracting the value from the messages
    #This normalizing is performed so that when these messages are calculated over many iterations, the messages
    #don't go to large values (it goes to e+10 in some rows if normalization is not performed 
    avg = np.mean(msg_u,axis=2)
    msg_u -= avg[:,:,np.newaxis]

    avg = np.mean(msg_d,axis=2)
    msg_d -= avg[:,:,np.newaxis]
    
    avg = np.mean(msg_l,axis=2)
    msg_l -= avg[:,:,np.newaxis]
    
    avg = np.mean(msg_r,axis=2)
    msg_r -= avg[:,:,np.newaxis]

    return msg_u,msg_d,msg_l,msg_r

#compute befiefs
def compute_beliefs(disp_cost,msg_u,msg_d,msg_l,msg_r):
    #Computing the beliefs based on the messages and data cost
    #Initiating beliefs to the disparity cost
    beliefs= disp_cost.copy()
    #ROlling the messages do that the new arrays represesnt the incoming 
    #messages into the pixel along all the disparity values
    incoming_U = np.roll(msg_d,1,axis=0)
    incoming_L = np.roll(msg_r,1,axis=0)
    incoming_D = np.roll(msg_u,-1,axis=0)
    incoming_R = np.roll(msg_l,-1,axis=1)

    #add the incoming messages of all 4 neighbours
    beliefs+= incoming_D + incoming_L + incoming_R + incoming_U
    return np.argmin(beliefs,axis=2)

def mrf_stereo(img1, img2, disp_cost):
    #Calling all the functions in order of uupdate message, normlaizing and computr_beliefs
    h,w,_ = img1.shape
    msg_u = np.zeros((h,w,MAX_DISPARITY))
    msg_d = np.zeros((h,w,MAX_DISPARITY))
    msg_l = np.zeros((h,w,MAX_DISPARITY))
    msg_r = np.zeros((h,w,MAX_DISPARITY))

    for r in range(iter):
        msg_u, msg_d,msg_l,msg_r = update_message(msg_u,msg_d,msg_l,msg_r,disp_cost,alpha)
        msg_u,msg_d,msg_l,msg_r = normalize_message(msg_u,msg_d,msg_l,msg_r)
        disparity = compute_beliefs(disp_cost,msg_u,msg_d,msg_l,msg_r)
       
    return disparity

# This function should compute the function D() in the assignment
def disp_cost_rgb(img1, img2):
    # this placeholder just returns a random cost map
    # initilizing numpy array of size Height, width and max_disparity with zeroes
    result = np.zeros((img1.shape[0], img1.shape[1], MAX_DISPARITY))
    w = 2
    # Changing all the values in the numpy array to infinity so that when we choose the 
    # minimun values in the beliefs the disparities where the window goes out of the range
    #is not considered as the minimum cost disparity.
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            for d in range(MAX_DISPARITY):
              result[i,j,d] = float('inf')
    # looping through the pixels of left view to find the matching of the window size in the right image 
    # and finding the sum of squared error fot different disparity values and saving the resultant error in 
    # result of corresponding disparity values.
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            for d in range(MAX_DISPARITY):
                bin_bound_window=0
                sum_each_d=0
                for u in range(-w,w+1):
                    for v in range(-w,w+1):
                        if i+u <0 or i+u >= img1.shape[0] or j+v<0 or j+v >= img1.shape[1] or j+v-d<0 or j+v-d >= img1.shape[1]:
                            continue
                        #summing error of each colour by squaring the differnce
                        sum_each_d += (int(img1[i+u,j+v,0]) - int(img2[i+u,j+v-d,0])) ** 2
                        sum_each_d += (int(img1[i+u,j+v,1]) - int(img2[i+u,j+v-d,1])) ** 2
                        sum_each_d += (int(img1[i+u,j+v,2]) - int(img2[i+u,j+v-d,2])) ** 2  
                #Dividing the total sum of error for each disparity            
                result[i,j,d] = sum_each_d/3
                
    return result
                
# This function finds the minimum cost at each pixel
def naive_stereo(img1, img2, disp_cost):
    return np.argmin(disp_cost, axis=2)
                     
if __name__ == "__main__":
    if len(sys.argv) != 3 and len(sys.argv) != 4:
        raise Exception("usage: " + sys.argv[0] + " image_file1 image_file2 [gt_file]")
    input_filename1, input_filename2 = sys.argv[1], sys.argv[2]

    # read in images and gt
    #image1 = np.array(Image.open(input_filename1).convert('L'))
    #image2 = np.array(Image.open(input_filename2).convert('L'))

    image1 = np.array(Image.open(input_filename1))
    image2 = np.array(Image.open(input_filename2))
   
    gt = None
    if len(sys.argv) == 4:
        gt = np.array(Image.open(sys.argv[3]))[:,:,0]

        # gt maps are scaled by a factor of 3, undo this...
        gt = gt / 3.0

    # compute the disparity costs (function D_2())
    #disp_cost = disparity_costs(image1, image2)
    disp_cost = disp_cost_rgb(image1, image2)
   
    # do stereo using naive technique
    disp1 = naive_stereo(image1, image2, disp_cost)
    #Image.fromarray(disp1.astype(np.uint8))#.save("output-naive.png")
    imageio.imwrite("output-naive.png",disp1)
    #plt.imshow(disp1, cmap='gray')
    #plt.show()

    # do stereo using mrf
    disp3 = mrf_stereo(image1, image2, disp_cost)
    #plt.imshow(disp3, cmap = "gray")
    #plt.show()
    #Image.fromarray(disp3.astype(np.uint8)).save("output-mrf.png")
    imageio.imwrite("output-mrf.png",disp3)

    # Measure error with respect to ground truth, if we have it...
    if gt is not None:
        err = np.sum((disp1- gt)**2)/gt.shape[0]/gt.shape[1]
        print("Naive stereo technique mean error = " + str(err))

        err = np.sum((disp3- gt)**2)/gt.shape[0]/gt.shape[1]
        print("MRF stereo technique mean error = " + str(err))
       
