# Reference -> http://nghiaho.com/?page_id=1366
import numpy as np
import sys

#No of iterations
iter = 2000 
#function for sum of the incoming messages - Label R
def sum_r(r_mat,i,j,k):
    s = 0   
    for l in range(4):
        di, dj = [(0,1),(1,0),(0,-1),(-1,0)][l]
        #checking the condition for adding incoming messages.            
        ni , nj = i+di , j+dj
        if  l == k or ni < 0 or ni >=n or nj<0 or nj>=n:
            continue
        #the direction for the node and the neighbour will be different.
        s += r_mat[ni,nj,(l+2)%4] #l+2 - to reverse the direction as the messages passed are outgoing
    return s

#function for sum of the incoming messages - Label D
def sum_d(d_mat,i,j,k):
    s = 0
    for l in range(4):
        di, dj = [(0,1),(1,0),(0,-1),(-1,0)][l]
        ni , nj = i+di , j+dj
         #checking the condition for adding incoming messages.
        if l == k or ni < 0 or ni >=n or nj<0 or nj>=n:
            continue
        #the direction for the node and the neighbour will be different.
        s +=d_mat[ni,nj,(l+2)%4] #l+2 - to reverse the direction as the messages passed are outgoing
    return s


def read_files(file_name):
    with open(file_name, 'r') as file:
        lines = file.readlines()
    bribes = np.zeros((len(lines), len(lines[0].split())))
    for i, line in enumerate(lines):
        bribes[i] = np.array([int(x) for x in line.split()])
    return bribes


if __name__ == '__main__':
    n=int(sys.argv[1])
    r_bribe = sys.argv[2]
    d_bribe = sys.argv[3]

    #Read files
    r_bribes = read_files(r_bribe)
    d_bribes = read_files(d_bribe)

    beliefs_r= np.zeros((n,n))
    beliefs_d=np.zeros((n,n))


    #Initialize
    r_messages=np.zeros((n,n,4)) #4 - left right up down
    d_messages=np.zeros((n,n,4))
    r_new=np.zeros((n,n,4)) #4 - left right up down
    d_new=np.zeros((n,n,4))
   
    r_new=np.zeros((n,n,4)) #4 - left right up down
    d_new=np.zeros((n,n,4))

    for r in range(iter):
        for i in range(n):
            for j in range(n):
                for k in range(4):

                    #calculate meassages passed to the neighbor
                    #(0,1) is right
                    #(0,-1) is left
                    #(1,0) is down
                    #(-1,0) is up
                    di, dj = [(0,1),(1,0),(0,-1),(-1,0)][k]
                    ni , nj = i+di , j+dj

                    #out of grid condition
                    if ni < 0 or ni >=n or nj<0 or nj>=n:
                        continue

                    #update messages using update equation when passing messages between nodes for each label R & D
                    r_new[i,j,k] = min(r_bribes[i,j] + sum_r(r_messages,i,j,k), d_bribes[i,j] + sum_d(d_messages,i,j,k) + 1000)
                    d_new[i,j,k] = min(r_bribes[i,j] + sum_r(r_messages,i,j,k)+1000, d_bribes[i,j] + sum_d(d_messages,i,j,k) )


        #normalization
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

    #this code block initializes arrays to store the incoming messages and computes these messages based on the outgoing messages from neighboring nodes in the opposite direction for the labels R and D
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

    for i in range(n):
        for j in range(n):
            for k in range(4):
                di, dj = [(0,1),(1,0),(0,-1),(-1,0)][k]
                   
                ni , nj = i+di , j+dj
                if  ni < 0 or ni >=n or nj<0 or nj>=n:
                    continue
                d_inc[i,j,k]=d_messages[ni,nj,(k+2)%4]
                   
    #update beliefs - product of all incoming messages
    beliefs_r = r_bribes + r_inc.sum(axis=2) #messages are summed over the 3rd dimension
    beliefs_d = d_bribes + d_inc.sum(axis=2)

    final_b= [[0 for i in range(n)] for j in range(n)]
   
    #comparing the beliefs to update the labels
    for j in range(n):    
        for i in range(n):
            if beliefs_r[i,j] < beliefs_d[i,j]:
                final_b[i][j]= 'R'
            else:
                final_b[i][j] ='D'
   
    #convert as per the output required
    print("Computing optimal labeling:")
    for i in final_b:
        for j in i:
            print(j,end=' ')
        print(' ')

    #calculating the cost for the resultant matrix
    cost = 0
    for i in range(n):
        for j in range(n):
            if final_b[i][j]=="R":
                cost+= r_bribes[i][j]
            else:
                cost+= d_bribes[i][j]
            for k in range(0,4):
                di, dj = [(0,1),(1,0),(0,-1),(-1,0)][k]
                ni , nj = i+di , j+dj
                #check out of grid condition.
                if  ni < 0 or ni >=n or nj < 0 or nj>=n:
                    continue
                if final_b[i][j] != final_b[ni][nj]:
                    cost+= 500
    
    print("Total cost = ",cost)