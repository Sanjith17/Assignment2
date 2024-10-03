# Assignment 2 : Warping, Matching, Stitching, Blending

## Part 1 : 3d-to-2d

### Aim : 
To generate the first person view of the airplane simulation. 

### Given data :
We are given a data file of containing the airplane coordinates. Some of the sample datapoints are given below: 

```
-40.0 -1000.0 0.0
40.0 -1000.0 0.0
-40.0 -999.0 0.0
40.0 -999.0 0.0
-40.0 -998.0 0.0
40.0 -998.0 0.0
-40.0 -997.0 0.0
40.0 -997.0 0.0
-40.0 -996.0 0.0
40.0 -996.0 0.0
```

###  Implementation and Approach :

The code to read the data file was already given in the initial file and is given below:  

``` sh
# load in 3d point cloud
with open("airport.pts", "r") as f:
    pts3 = [ [ float(x) for x in l.split(" ") ] for l in f.readlines() ]
```

To create the animation, we are generating the subplots and merging them in the frame using the inbuilt python library. The code for the same is given below:
``` sh
# creating the animation!
fig, ax  = plt.subplots()

frame_count = 300 

# creating animation function
ani = animation.FuncAnimation(fig, animate_above, frames=range(0, frame_count), blit=True)

# saving the animation 
ani.save("movie.mp4")
``` 

The <b>animate_above</b> method gets called for each frame. To begin with, we need to set the initial camera parameters for our simulation. The initial camera parameters is given below:
``` sh
variables to declare the initial camera position
foc = 0.003 
(tx, ty, tz) = (0, 0, 50)
(tilt, twist, compass) = (-pi/2, 0, 0)
```

We need to define the parameters for the rotation matrix. We have declared a method named <b>get_rotation_matrix</b> which performs matrix multiplication of tilt, twist and yaw angles and returns the resultant matrix. The code for <b>get_rotation_matrix </b> is given below:

``` sh
# function to get the rotation matrix
def get_rotation_matrix(tilt, twist, compass):
    tilt_matrix = np.array([[1, 0, 0], 
                            [0, np.cos(tilt), -np.sin(tilt)],
                            [0, np.sin(tilt), np.cos(tilt)]])

    twist_matrix = np.array([[np.cos(twist), 0, -np.sin(twist)],
                             [0, 1, 0], 
                             [np.sin(twist), 0, np.cos(twist)]])

    yaw_matrix = np.array([[np.cos(compass), -np.sin(compass), 0],
                           [np.sin(compass), np.cos(compass), 0], 
                           [0, 0, 1]])
    
    rotation_matrix = np.matmul(tilt_matrix, twist_matrix)
    rotation_matrix = np.matmul(rotation_matrix, yaw_matrix)

    return rotation_matrix
```

We are using 300 frames to generate our simulation. The <b>animate_above</b> method gets called for each frame. The code for the same is given below:

``` sh
def animate_above(frame_number):
    # declaring the row and column list
    pr = []
    pc = []

    # using the global parameters to create the animation
    global tx, ty, tz, tilt, twist, compass, foc

    # flight is landing on the runway
    if frame_number > 260:
        ty += 10
        tz /= 1.05
    # flight is aligning to the runway
    elif frame_number > 250:
        if twist <= 0: 
            twist = 0
        else:
            twist -= 0.01047

        if compass >= 6.28:
            compass = 6.28
        else:
            compass += 0.01047 * 3
        
        if ty < 40:
            ty += 1 
        else:
            ty = 40

    elif frame_number > 210:
        compass += 0.01047 * 3
        if twist <= 0: 
            twist = 0
        else:
            twist -= 0.01047

        tx = 0

    elif frame_number > 180: 
        compass += 0.01047 * 3
        twist -= 0.01047 

        tx -= 2
        ty += 1

    elif frame_number > 150:
        compass += 0.01047 * 3
        twist += 0.01047 
        tx -= 1
        ty -= 2

    # flight going parallel to the runway
    elif frame_number > 120: 
        ty -= 19

    # flight taking right turn in next 60 frames
    elif frame_number > 85:
        compass += 0.01047 * 3
        twist += 0.01047
        tx += 1
        ty -= 1

    elif frame_number > 60:
        compass += 0.01047 * 3
        twist -= 0.01047 
        tx += 1
        ty -= 1

    elif frame_number > 30:
        compass += 0.01047 * 3
        twist += 0.01047 
        tx += 1
        ty += 1

    # flight taking off in 1st 30 frames
    elif frame_number > 15:
        ty += 20

    elif frame_number > 8:
        tz *= 1.2
        ty += 20 
        
    else:
        tz += 3
        ty += 20


    # getting the rotation matrix
    rotation_matrix = get_rotation_matrix(tilt, twist, compass)
    
    # declaring the focal_length_matrix
    focal_length_matrix = np.array([[foc, 0, 0],
                                    [0, foc, 0],
                                    [0, 0, 1]])

    # declaring the camera_position_matrix
    camera_position_matrix = np.array([[1, 0, 0, tx], 
                                       [0, 1, 0, ty],
                                       [0, 0, 1, tz]])

    # calculation of the projection matrix
    projection_matrix = np.matmul(rotation_matrix, camera_position_matrix)
    projection_matrix = np.matmul(focal_length_matrix, projection_matrix) 
    
    # iterating and multiplying the points with the projection matrix
    for p in pts3:
        # creation of homogeneous coordinates
        p = np.array([p[0], p[1], p[2], 1])
      # multiply the homogeneous coordinates with the projection matrix
        mat_ = np.matmul(projection_matrix, p)

        # appending the x and y coordinates in a list to plot the same
        if mat_[2] < 0:
            continue

        pr.append(-mat_[0] / mat_[2])
        pc.append(-mat_[1] / mat_[2])
    
    # setting the limits for the plots
    plt.cla()
    plt.gca().set_xlim([-0.01, 0.01])
    plt.gca().set_ylim([-0.01, 0.01])

    # plotting the plot for the x and y coordinates 
    line, = plt.plot(pr, pc, 'k',  linestyle="", marker=".", markersize=2)
    
    # returning the drawn graph
    return line,
```

In our simulation, we perform the the following operations on the airplane
<ul>
<li> Take the airplane off. 
<li> Turn right. 
<li> Align the plane to the runway.
<li> Fly the plane parallel to the runway.
<li> Turn right.
<li> Align to the runway.
<li> Land the plane. 
</ul>

For each of the frame, we are calculating the final 3d points by taking the matrix multiplication of final rotation matrix, focal length matrix and position matrix (tx, ty, tz) for each of the data point. Some of the resulting datapoints are negative which means that the points is getting projected behind the camera. We need to discard such points. After discarding these points, we are playing with the tx, ty, tz, tilt, twist and yaw angles to generate our final simulation. We then convert the 3d coordinates into 2d coordinates and then project on 2d plane. We chose 300 frames to generate our animation, and the final result which we get is of 1 minute. The result for our animation is given in the results section.

### Result :
The mp4 result of the simulation is given below:

https://media.github.iu.edu/user/20705/files/26166421-cd66-4340-8d50-55e23f5cbfee

### Difficulty Faced : 
<ul>
<li> We faced difficulty to setup the ffmpeg extension for python on our local devices. After exploring, we were able to solve the error regarding the ffmpeg python module.
<li> Initially the airplane was able to take off and turn right successfully. We were facing difficulties in aligning the airplane to the runway. After playing with tx, ty, and tilt, twist and compass angles, we were able to align our airplane to the runway. 
<li> We were getting 2 images of the of the runway. We found out that we need to discard the negative values. After discarding the negative values we got the correct results.
<li> We were gettting an inverted output of the runway. To tackle this, we inverted the values and got the correct orientation of the runway.
</ul>

### References :
Lecture slides to understand the concepts of 3d projections


## Part 2 : Understanding Markov Random Fields
Looping belief propagation algorithm is implemented which is a message passing algorithm where a node passes message to a neighboring node only when it has received all the incoming messages, excluding the message from the destination node to itself. Normalization is performed to avoid zero probabilities as they tend towards zero by continuously multiplying the probabilities.


Algorithm Steps:
1)	Reading the r_bribes and d_bribes text files

2)	Updating messages using update equation for each label R and D: 

 ![image](https://media.github.iu.edu/user/21148/files/dfc0d654-bbeb-41e6-809c-e65b534ff703)
 
 ``` sh
 r_new[i,j,k] = min(r_bribes[i,j] + sum_r(r_messages,i,j,k), d_bribes[i,j] + sum_d(d_messages,i,j,k) + 1000)
 d_new[i,j,k] = min(r_bribes[i,j] + sum_r(r_messages,i,j,k)+1000, d_bribes[i,j] + sum_d(d_messages,i,j,k))
 ```
 In the above code, 1000 represents the fencing cost when the two neighboring labels are different and when they are same the cost is zero which we have ignored(not explicitly added to the code)
 
 
 
3)	Normalization for the R and D messages


``` sh
if r <= 4: #finding the maximum from the
            max_r = r_new.max()
            max_d = d_new.max()
            r_messages=r_new #4 - left right up down
            d_messages=d_new
        else:
          r_new *= max_r/r_new.max()
          d_new *= max_d/d_new.max()
          r_messages=r_new #4 - left right up down
          d_messages=d_new
```

In the above code nprmalization of message was done. At the message aray at fourth iteration is considered to perform the normalization. From the fourth iteration, every message array undergoers min max normalizartion where the elements of the array gets normalized based on the max value of the messages in the fourth iteration.
The fourth iteration was chosen with experimentation as in this case, the messages are getting converged early

4)	Message Initialization

``` sh
 r_inc=np.zeros((n,n,4)) #4 - left right up down
    d_inc=np.zeros((n,n,4))
    for i in range(n):
        for j in range(n):
            for k in range(4):
                #print(k)
                #(0,1) is right
                #(0,-1) is left
                #(1,0) is down
                #(-1,0) is up
                di, dj = [(0,1),(1,0),(0,-1),(-1,0)][k]
                   
                ni , nj = i+di , j+dj
                if  ni < 0 or ni >=n or nj<0 or nj>=n:
                    continue
                r_inc[i,j,k]=r_messages[ni,nj,(k+2)%4]
```
An numpy array of size NxNX4 is created where the last dimension represents message in each direction. This array represents the outgoing messages for each node in the corresponding direction.

Similar initialization is done for D label messages



5)	Updating beliefs using final state determination equation:
 
![image](https://media.github.iu.edu/user/21148/files/540be18b-5fde-4d1d-a301-a69fe09e3fc3)


``` sh
beliefs_r = r_bribes + r_inc.sum(axis=2) #messages are summed over the 3rd dimension
beliefs_d = d_bribes + d_inc.sum(axis=2)

```

The above code finds the beliefs of each party by summing the cost (which is the bribe cost for each party) with the incoming messages to the node for the respective beliefs.

6)  Extracting the incoming messages from outgoing messages array.

``` sh
 r_inc=np.zeros((n,n,4)) #4 - left right up down
    d_inc=np.zeros((n,n,4))
    for i in range(n):
        for j in range(n):
            for k in range(4):
                #print(k)
                di, dj = [(0,1),(1,0),(0,-1),(-1,0)][k]
                   
                ni , nj = i+di , j+dj
                if  ni < 0 or ni >=n or nj<0 or nj>=n:
                    continue
                r_inc[i,j,k]=r_messages[ni,nj,(k+2)%4]
```
THe r_messages represent the outgoinf messages. We need incoming messages for the further calculations. The incoming message to a pixel is the outgoing message of the corresponding pixel in the reverse direction. So, by performing (k+2)%4, the value of k which represnts the direction of the message changes and the corresponding message can be assigned to incoming message array.


Output:

<img width="206" alt="Screen Shot 2023-04-11 at 10 13 08 PM" src="https://media.github.iu.edu/user/21193/files/35f7ab6a-de73-4c93-bc1c-53264d57b1b2">


Experiments performed:  
1)	Tried changing the number of iterations required from a very small number to a large no of iterations 
2)	Experimented with different normalization techniques to get the optimal cost of labeling like min max, log, and exponential

Difficulties Faced: 
1)	We were getting optimal label cost only between the iterations 22 to 55. To solve this issue, we performed normalization.
2)	Defining the messages data structure and updating the messages

What we can do further to make our algorithm better:
1)	We can try to decrease the cost of computation by implementing the algorithm by dynamic programming
2)	We can try implementing the algorithm by choosing better convergence criteria instead of keeping fix no of iterations


Reference: http://nghiaho.com/?page_id=1366 and the lecture slides on Stereo and Markov Random Fields

## Part 3 : Inferring depth from stereo.
This part is divided into 3 parts. 
The first being finding the datacost for the given left and right image. 
Secondly, we find the minimum cost using the naive stereo technique. We compare the disparity map produced by this and the ground truth depth map. 
Lastly, we apply the loopy belief propagation for choosing the disparity labels giving the minimum cost. Here,we find the data cost and update the messages from the neighbouring (left, right ,up, down) pixels for every disparity value and finding the best disparity value for each pixel in the right image when compared to the left image. Similarly, the disparity map and the difference between the disparity map and the ground truth is found. 

The stereo energy function is defined by following formula.
<br>
![image](https://media.github.iu.edu/user/21148/files/b47b00ce-66a4-4607-b5f7-7602ce92f8fb)

D function defines the data cost of the images while V function is to calculate pairwise distance for the pixels of the two images. Alpha is the smoothing factor. 

## Finding the Data Cost.
The data function calculates the similarity among the two views of the images at different disparity levels and returns the sum of squared error values among different disparity values and is defined as below: 

![image](https://media.github.iu.edu/user/21148/files/baa5e742-a914-4c16-8a8b-36fe78f136f7)

The disparity cost is calculated by the above formula. Since the images are in RGB format. This window matching is done for every pixel of every colour channel and an average of the cost of the three colour channels is considered as the cost of that pixel for a disparity value.

W refers to the window size, i and j refer the pixel location and d represents the disparity value. This checks the error values of different disparity values by comapring each pixel in a window around each pixel and sums the squared error for each disparity value for that pixel. 

``` sh
def disp_cost_rgb(img1, img2):
    result = np.zeros((img1.shape[0], img1.shape[1], MAX_DISPARITY))
    w = 2
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            for d in range(MAX_DISPARITY):
              result[i,j,d] = float('inf')

    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            for d in range(MAX_DISPARITY):
                bin_bound_window=0
                sum_each_d=0
                for u in range(-w,w+1):
                    for v in range(-w,w+1):
                        row = i+u
                        col = j+v-d
                        if i+u <0 or i+u >= img1.shape[0] or j+v<0 or j+v >= img1.shape[1] or j+v-d<0 or j+v-d >=img1.shape[1]:
                            continue
                        bin_bound_window+=1
                        sum_each_d += (int(img1[i+u,j+v,0]) - int(img2[i+u,j+v-d,0])) ** 2
                        sum_each_d += (int(img1[i+u,j+v,1]) - int(img2[i+u,j+v-d,1])) ** 2
                        sum_each_d += (int(img1[i+u,j+v,2]) - int(img2[i+u,j+v-d,2])) ** 2  
                
                result[i,j,d] = sum_each_d/3
                
    return result

``` 
The above code calculates the cost for each disparity values for each pixel and returns a numpy array of dimensions H,W and Max_disparity. The function takes in the images as the input parameters. It loops through each pixel of the left image, then for a window size around the pixel for each disparity finds the squred error and returns the result. By experimentation, we have chosen the window size as 2. The disparity values range from 0 to 29, 30 values.
 
## Naive Stereo.
This function just results the min error value at a pixel among all the disparity values. This is just the implementaion of the disparity cost function that has the least error
``` sh
def naive_stereo(img1, img2, disp_cost):
    return np.argmin(disp_cost, axis=2)
``` 
Naive Stereo simply checks the disparity value for which the disparity cost is minimum. 

## Loopy Belief Implementation

The pixels in the image are in two dimension and there will be loop formation in the image. So loopt belief function is used to find th Beliefs for the disparity levels. First the messages that are to be sent from one pixel to other pixel is calculated as depicted in below.
<br>
![image](https://media.github.iu.edu/user/21148/files/3b7c25ac-d35f-4e1e-b91b-136057828b25)
<br>
D is the disparity function and summation m part represents the messages sent to the pixel from its neighbours apart from the one the message is being sent to. This is pretty much similar to the part2. 

The D cost is the squared error pixel by pixel of the left and right image. The V function is used as Pott's metric which we have take V(x) = 1 for x = 0 and V(x) = 0, where x is the difference between the label of a pixel and it's neighbouring pixel. The label being the disparity value.
Alpha the smoothing factor is taken as 10. 
We run this code for iterations = 50

This process is repeated for some iterations by normalizing the messages in each iteration. The messages converges in between the iterations and there will not be much change after that.

Then based on these messages, the beliefs are calculated by finding the minimum value that is defined as 
<br>
![image](https://media.github.iu.edu/user/21148/files/4c8bdeb3-5335-41d5-a55b-707d28e09d29)

The data cost is added to the incoming messages to the pixel for different disparit values and the minimum is calculated form them that has the depth image in loopy implementation.

### Message Updation.
The following code is used to update the messages.

``` sh
def update_message(msg_u,msg_d,msg_l,msg_r,disp_cost,alpha):
    msg_u = np.zeros(disp_cost.shape)
    msg_d = np.zeros(disp_cost.shape)
    msg_l = np.zeros(disp_cost.shape)
    msg_r = np.zeros(disp_cost.shape)

    h,w,MAX_DISPARITY = disp_cost.shape
    
    #incoming messages 
    incoming_U = np.roll(msg_u, 1,axis=0)
    incoming_L = np.roll(msg_l, 1, axis=1)
    incoming_D = np.roll(msg_d, -1, axis=0)
    incoming_R = np.roll(msg_r, -1, axis=1)

    #
    npqu = disp_cost + incoming_L + incoming_D + incoming_R
    npql = disp_cost + incoming_U + incoming_D + incoming_R
    npqd = disp_cost + incoming_L + incoming_U + incoming_R
    npqr = disp_cost + incoming_L + incoming_D + incoming_U

    spqU = np.amin(npqu, axis=2)
    spqL = np.amin(npql, axis=2)
    spqD = np.amin(npqd, axis=2)
    spqR = np.amin(npqr, axis=2)

    for p in range(MAX_DISPARITY):
        msg_u[:,:,p]= np.minimum(npqu[:,:,p], alpha+spqU)
        msg_d[:,:,p]= np.minimum(npqd[:,:,p], alpha+spqD)
        msg_l[:,:,p]= np.minimum(npql[:,:,p], alpha+spqL)
        msg_r[:,:,p]= np.minimum(npqr[:,:,p], alpha+spqR)

    return msg_u, msg_d, msg_l, msg_r
``` 

Initially the messages are initilized with zeroes of dimension if height, width and Max_Disparity for messages in each direction (Up, Down, Left and Right). These represents the outgoing messages. In the next stepts, the incoming messages are calculted in each direction for each pixel by rolling the outgoing messages. Then for each direction, the messages are calculated by the function where the incoming images are added.

### Normalizing Messages
The following function normalizes the message. This makes sure that the messages don't go to infinity with the iterations.
The average is calculted first for each direction and the average is subtracted from the messages in each direction and the resultant messages are returned. 

``` sh
def normalize_message(msg_u,msg_d,msg_l,msg_r):
    #normalizing using the avg. 

    avg = np.mean(msg_u,axis=2)
    msg_u -= avg[:,:,np.newaxis]

    avg = np.mean(msg_d,axis=2)
    msg_d -= avg[:,:,np.newaxis]
    
    avg = np.mean(msg_l,axis=2)
    msg_l -= avg[:,:,np.newaxis]
    
    avg = np.mean(msg_r,axis=2)
    msg_r -= avg[:,:,np.newaxis]

    return msg_u,msg_d,msg_l,msg_r
``` 

### Belief Function
The belieds are summing all the incoming messages for each pixel with the data cost of the pixel for that disparity value. After this then srgmin function is returned from the function as the result which will have the depth map

``` sh
def compute_beliefs(disp_cost,msg_u,msg_d,msg_l,msg_r):

    beliefs= disp_cost.copy()
    incoming_U = np.roll(msg_d,1,axis=0)
    incoming_L = np.roll(msg_r,1,axis=0)
    incoming_D = np.roll(msg_u,-1,axis=0)
    incoming_R = np.roll(msg_l,-1,axis=1)

    #add the incoming messages of all 4 neighbours
    beliefs+= incoming_D + incoming_L + incoming_R + incoming_U

    return np.argmin(beliefs,axis=2)
``` 

### MRF Function
This function just calls all the previuosly mentioned functions in sequence and return diparity as the final outpuy for loopy belief

``` sh
def mrf_stereo(img1, img2, disp_cost):

    h,w = img1.shape
    msg_u = np.zeros((h,w,MAX_DISPARITY))
    msg_d = np.zeros((h,w,MAX_DISPARITY))
    msg_l = np.zeros((h,w,MAX_DISPARITY))
    msg_r = np.zeros((h,w,MAX_DISPARITY))

    for r in range(iter):
        msg_u, msg_d,msg_l,msg_r = update_message(msg_u,msg_d,msg_l,msg_r,disp_cost,alpha)
        msg_u,msg_d,msg_l,msg_r = normalize_message(msg_u,msg_d,msg_l,msg_r)
        disparity = compute_beliefs(disp_cost,msg_u,msg_d,msg_l,msg_r)

    return disparity
``` 
## Results: 
When tested for the Aloe/view1.png and Aloe/view5.png.We get the following results. 
This is the disparity map produced by the naive stereo algorithm.
<br>
![output-naive](https://media.github.iu.edu/user/21540/files/ee17c359-2751-40c3-8039-25f0c04014d3)

This is the disparity map produced by the mrf stereo loopy belief algorithm.
<br>
![output-mrf](https://media.github.iu.edu/user/21540/files/dbf871fd-8391-4006-b067-b6e3cd80cd60)

The disparity map of the two algorithms is compared to the ground truth of the depth map and the mean squared error of is calculated and can be seen below.
<br>
<img width="620" alt="image" src="https://media.github.iu.edu/user/21540/files/1a5d89eb-f58a-411e-8a30-4f37185763ee">

## Difficulties
<ul>
 <li> Deciding a data strucutre to store the messages in four directions, for each disparity value and for each pixel
 <li> Finding disparity cost of the pixels for differnt disparity values.
 <li> Finding a optimal Max_Disparity and optimal window size for each example took some time.
</ul>

## References:
1. https://github.com/aperezlebel/StereoMatching/tree/962de28a278e4f5c365400f1c3b45f57dfadd2b0 (For loopy belief, this code was used for reference.)

## Contributions:
Part1 was implemented by Shrirang and Sakshi. Part2 was implemented by Shruti and Sanjith. Part3 was implemented collectively by everyone. Report part1 was written by Shrirang, part 2 was drafted by Shruti and Sanjith and part3 was jotted by Sanjith and Sakshi.
