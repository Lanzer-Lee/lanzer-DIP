#Student: 2104 Lanzer
import numpy as np
import cv2

def load_data(filename_A,filename_B,filename_coordinate_A,filename_coordinate_B):
    #load image and coordinates saved by hands before, save them as matrix
    P=np.loadtxt(filename_coordinate_A)
    P=np.concatenate((P.T,np.ones([1,7])),axis=0)
    Q=np.loadtxt(filename_coordinate_B)
    Q=np.concatenate((Q.T,np.ones([1,7])),axis=0)
    img_A=cv2.imread(filename_A)
    img_A=cv2.resize(img_A,([int(img_A.shape[1]/3),int(img_A.shape[0]/3)]),cv2.INTER_AREA)
    img_B=cv2.imread(filename_B)
    img_B=cv2.resize(img_B,([int(img_B.shape[1]/3),int(img_B.shape[0]/3)]),cv2.INTER_AREA)
    return img_A,img_B,P,Q

def get_matrix(P,Q):
    #calculate the transformation matrix
    H=np.matmul(Q,P.T)
    H=np.matmul(H,np.linalg.inv(np.matmul(P,P.T)))
    return H

def get_size(input_img,H):
    #calculate the size and offset of the output image
    length,width=input_img.shape[:2]
    x0,y0=np.matmul(H,np.array([[0],[0],[1]]))[:2]
    x1,y1=np.matmul(H,np.array([[0],[width-1],[1]]))[:2]
    x2,y2=np.matmul(H,np.array([[length-1],[width-1],[1]]))[:2]
    x3,y3=np.matmul(H,np.array([[length-1],[0],[1]]))[:2]
    width_new=max([abs(y0-y1),abs(y0-y2),abs(y0-y3),abs(y1-y2),abs(y1-y3),abs(y2-y3)])
    length_new=max([abs(x0-x1),abs(x0-x2),abs(x0-x3),abs(x1-x2),abs(x1-x3),abs(x2-x3)])
    return [int(length_new)+2,int(width_new)+2,3],int(min([x0,x1,x2,x3])),int(min([y0,y1,y2,y3]))

def registration(input_img,H):
    #image registration by forward transformation
    output_size,x_bias,y_bias=get_size(input_img,H)
    output_img=np.zeros(output_size,dtype=np.uint8)
    for x in range(input_img.shape[0]):
        for y in range(input_img.shape[1]):
            u,v=np.matmul(H,np.array([[x],[y],[1]]))[:2]
            u=int(np.round(u))-x_bias
            v=int(np.round(v))-y_bias
            output_img[u,v,:]=input_img[x,y,:]
    return output_img

def reverse_registration(input_img,H):
    #image registration by reverse transformation
    output_size,x_bias,y_bias=get_size(input_img,H)
    output_img=np.zeros(output_size,dtype=np.uint8)
    H=np.linalg.inv(H)
    for u in range(output_size[0]):
        for v in range(output_size[1]):
            x,y=np.matmul(H,np.array([[u],[v],[1]]))[:2]
            x=int(np.round(x))+x_bias
            y=int(np.round(y))+y_bias
            if(0<=x and x<input_img.shape[0] and 0<=y and y<input_img.shape[1]):
                output_img[u,v,:]=input_img[x,y,:]
    return output_img

def main():
    #run get_coordinate.py to get coordinates saved as txt
    filename_coordinate_A='data/coordinate.txt'
    filename_coordinate_B='data/coordinate2.txt'
    filename_A='Image A.jpg'
    filename_B='Image B.jpg'
    img_A,img_B,P,Q=load_data(filename_A,filename_B,filename_coordinate_A,filename_coordinate_B)
    H_ab=get_matrix(P,Q)
    H_ba=get_matrix(Q,P)
    #print(H_ab)
    #print(H_ba)
    #input A, template B, output A' like B
    outA=reverse_registration(img_A,H_ba)
    #input B, template A, output B' like A
    outB=reverse_registration(img_B,H_ab)
    cv2.imshow('A',outA)
    cv2.imshow('B',outB)
    cv2.imwrite('out_A.jpg',outA)
    cv2.imwrite('out_B.jpg',outB)
    cv2.waitKey(0)
    
if __name__=='__main__':
    main()





