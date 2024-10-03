import numpy as np
from math import *
import os

# The following tries to avoid a warning when run on the linux machines via ssh.
if os.environ.get('DISPLAY') is None:
     import matplotlib 
     matplotlib.use('Agg')
       
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# variables to declare the initial camera position
foc = 0.003 
(tx, ty, tz) = (0, 0, 50)
(tilt, twist, compass) = (-pi/2, 0, 0)

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


# load in 3d point cloud
with open("airport.pts", "r") as f:
    pts3 = [ [ float(x) for x in l.split(" ") ] for l in f.readlines() ]


# creating the animation!
fig, ax  = plt.subplots()

frame_count = 300 

# creating animation function
ani = animation.FuncAnimation(fig, animate_above, frames=range(0, frame_count), blit=True)

# saving the animation 
ani.save("movie.mp4")
