# --------------------------proj 3 ques 1----------------------------
import numpy as np
import math
import scipy as sp
import cv2 as cv
import os

# Image points
x = [[757,213],[758,415],[758,686],[759,966],[1190,172],[329,1041],[1204,850],[340,159]]
X = [[0,0,0],[0,3,0],[0,7,0],[0,11,0],[7,1,0],[0,11,7],[7,9,0],[0,1,7]]
# print(X[0])
A = np.zeros((0,12),np.float32)
#Creating the A-Matrix
for l in range(len(X)):
    ui_point = x[l][0]
    vi_point = x[l][1]
    Xi_point = X[l][0]
    Yi_point = X[l][1]
    Zi_point = X[l][2]    
    
    A_row1 = [0,0,0,0,(-Xi_point),(-Yi_point),(-Zi_point),-1,(vi_point*Xi_point),(vi_point*Yi_point),(vi_point*Zi_point),(vi_point*1)]
    A_row2 = [Xi_point,Yi_point,Zi_point,1,0,0,0,0,(-ui_point*Xi_point),(-ui_point*Yi_point),(-ui_point*Zi_point),(-ui_point*1)]
    A_row3 = [(-vi_point*Xi_point),(-vi_point*Yi_point),(-vi_point*Zi_point),(-vi_point*1),(ui_point*Xi_point),(ui_point*Yi_point),(ui_point*Zi_point),(ui_point*1),0,0,0,0]
    A = np.vstack((A,A_row1,A_row2,A_row3))    


#Computing the P matrix
_,_,V_T = np.linalg.svd(A)
Pmatrix = V_T[-1]
print("P matrix")
print(Pmatrix)
P_matrix = np.reshape(V_T[-1],(3,4))
div = P_matrix[-1,-1]
P_matrix = P_matrix/div
# print(np.shape(P_matrix))
#Computing the C matrix
_,_,v_t = np.linalg.svd(P_matrix)

C_vector = v_t[-1]
div_c = C_vector[-1]
C_vector = C_vector/div_c  
# print(C_vector)
C_vector = C_vector[:-1]
# print(C_vector)
C_vector = np.reshape(C_vector,(3,1))
print("Translation Matrix")
print(C_vector)
#joining the Identity matrix and C-vector
Identity_matrix = [[1,0,0],[0,1,0],[0,0,1]]
new_IdentityMatrix = np.hstack((Identity_matrix,-C_vector)) 
# print(new_IdentityMatrix)
#Computing the M-matrix
inverseIdentityMatrix = np.linalg.pinv(new_IdentityMatrix)
# print(inverseIdentityMatrix)  
M_matrix = np.matmul(P_matrix,inverseIdentityMatrix)
# print(M_matrix)

#Computing the K and R

K,R = sp.linalg.rq(M_matrix)
print("Intrisic Matrix K")
print(K)
print("Rotational Matrix")
print(R)
ones = np.ones((8,1),np.float32)
Worldpoints = np.hstack((X,ones))
# print(Worldpoints)
Worldpoints = Worldpoints.T
#print(Worldpoints)
P_World = P_matrix @ Worldpoints
#print(P_World)
div_pworld = P_World[-1]
P_World = P_World/div_pworld
P_World = P_World[:-1]
P_World = P_World.T
#print("pworld")
# print(P_World)
image_pts = np.array(x)
# print("imagepts")
# print(image_pts)

Reprojection_error = np.linalg.norm(image_pts - P_World,axis=1)
print("Reprojection Errors for every points")
for i in range(0,8):
    print(f"The reprojection error for the point {(i+1)} = {Reprojection_error[i]}")


#---------------------project 3 Ques 2-------------------------------------------------


criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
object_coord = []
image_coords = []
amc= []
arr_map = np.zeros((6*9,3), np.float32)
arr_map[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

folder_path = "/home/srv/perception673/project3"

for filename in os.listdir(folder_path):
    if filename.endswith(".jpg"):
        image_path = os.path.join(folder_path, filename)
        image = cv.imread(image_path)
        image = cv.resize(image,(2000,1512))
        gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
        ret,corner = cv.findChessboardCorners(gray, (9,6),None	) 
        
        if ret:
            object_coord.append(arr_map*21.5)
            acc_corners = cv.cornerSubPix(gray,corner,(11,11),(-1,-1),criteria)
            image_coords.append(acc_corners)
            cv.drawChessboardCorners(image,(9,6),acc_corners,ret)
            cv.imshow("image",image)
            cv.waitKey()
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(object_coord, image_coords, gray.shape[::-1], None, None)

mean_error = 0
for i in range(len(object_coord)):
    image_points,_ = cv.projectPoints(object_coord[i],rvecs[i],tvecs[i],mtx,dist)
    error = cv.norm(image_coords[i],image_points,cv.NORM_L2)/len(image_points)
    print(f"The projection error for image {i+1} = {error}")
    mean_error += error


print("total error: {}".format(mean_error/len(object_coord)))
print("calibration matrix(K)")
print(mtx)    
cv.destroyAllWindows()
