'''
School: XJTU
Course: DIP
Student: Lanzer
Class: 2104
'''
import cv2
import numpy as np

def bit_transform(img,bit):
    return np.uint8(img/2**(8-bit))*2**(8-bit)

def lanzer_mean(img):
    #return np.mean(img)
    sum=0.0
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            sum+=float(img[x,y])
    return sum/(img.shape[0]*img.shape[1])

def lanzer_variance(img):
    #return np.var(img)
    sum=0.0
    img_mean=lanzer_mean(img)
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            sum+=(img[x,y]-img_mean)**2
    return sum/(img.shape[0]*img.shape[1])

def lanzer_resize_nearest(img,size):
    #return cv2.resize(img,size,interpolation=cv2.INTER_NEAREST)
    resized_img=np.zeros(size,dtype=np.uint8)
    x_factor=size[0]/img.shape[0]
    y_factor=size[1]/img.shape[1]
    for x in range(0,size[0],1):
        for y in range(0,size[1],1):
            x_old=int(x/x_factor)
            y_old=int(y/y_factor)
            resized_img[x,y]=img[x_old,y_old]
    return resized_img

def lanzer_resize_linear(img,size):
    #return cv2.resize(img,size,interpolation=cv2.INTER_LINEAR)
    resized_img=np.zeros(size)
    x_factor=size[0]/img.shape[0]
    y_factor=size[1]/img.shape[1]
    padded_img=np.zeros([img.shape[0]+2,img.shape[1]+2],dtype=np.uint8)
    padded_img[1:img.shape[0]+1,1:img.shape[1]+1]=img
    for x in range(0,size[0],1):
        for y in range(0,size[1],1):
            x_old=x/x_factor+1
            y_old=y/y_factor+1
            x0=int(x_old)
            y0=int(y_old)
            u=x_old-x0
            v=y_old-y0
            resized_img[x,y]=(1-u)*(1-v)*padded_img[x0,y0]+(1-u)*v*padded_img[x0,y0+1]+u*(1-v)*padded_img[x0+1,y0]+u*v*padded_img[x0+1,y0+1]
    return np.uint8(resized_img)

def cubic_kernel(x):
    if (np.abs(x)<=1):
        return 1-2*np.power(np.abs(x),2)+np.power(np.abs(x),3)
    elif ((1<np.abs(x))&(np.abs(x)<2)):
        return 4-8*np.abs(x)+5*np.power(np.abs(x),2)-np.power(np.abs(x),3)
    else:
        return 0
    
def lanzer_resize_cubic(img,size):
    #return cv2.resize(img,size,interpolation=cv2.INTER_CUBIC)
    resized_img=np.zeros(size)
    x_factor=size[0]/img.shape[0]
    y_factor=size[1]/img.shape[1]
    padded_img=np.zeros([img.shape[0]+4,img.shape[1]+4],dtype=np.uint8)
    padded_img[2:img.shape[0]+2,2:img.shape[1]+2]=img
    for i in range(0,size[0],1):
        for j in range(0,size[1],1):
            x=i/x_factor+2
            y=j/y_factor+2
            x0=int(x)
            y0=int(y)
            u=x-x0
            v=y-y0
            A=np.array([cubic_kernel(u+1),cubic_kernel(u),cubic_kernel(u-1),cubic_kernel(u-2)])
            C=np.array([cubic_kernel(v+1),cubic_kernel(v),cubic_kernel(v-1),cubic_kernel(v-2)])
            B=padded_img[x0-1:x0+3,y0-1:y0+3]
            resized_img[i,j]=np.matmul(A,np.matmul(B,C.T))
    return resized_img

def lanzer_rotate(img,theta):
    new_shape=[int(img.shape[0]*np.cos(theta))+int(img.shape[1]*np.sin(theta)),int(img.shape[0]*np.cos(theta))+int(img.shape[1]*np.sin(theta))]
    img_new=np.zeros(new_shape,dtype=np.uint8)
    rows,cols=new_shape[:2]
    T=np.matrix([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]])
    T=np.linalg.inv(T)
    for x in range(0,img_new.shape[0],1):
        for y in range(0,img_new.shape[1],1):
            coordinate=np.array([[x-int(rows/2)],[y-int(cols/2)],[1]])
            coordinate_old=np.matmul(T,coordinate)
            x_old=int(coordinate_old[0])+int(img.shape[0]/2)
            y_old=int(coordinate_old[1])+int(img.shape[1]/2)
            if(0<=x_old<img.shape[0] and 0<=y_old<img.shape[1]):
                img_new[x,y]=img[x_old,y_old]
    return img_new

def lanzer_horizontal_shear(img,factor):
    new_shape=[img.shape[0],int((factor+1)*img.shape[1])]
    img_new=np.zeros(new_shape,dtype=np.uint8)
    rows,cols=new_shape[:2]
    T=np.matrix([[1,0,0],[factor,1,0],[0,0,1]])
    T=np.linalg.inv(T)
    for x in range(0,img_new.shape[0],1):
        for y in range(0,img_new.shape[1],1):
            coordinate=np.array([[x-int(rows/2)],[y-int(cols/2)],[1]])
            coordinate_old=np.matmul(T,coordinate)
            x_old=int(coordinate_old[0])+int(img.shape[0]/2)
            y_old=int(coordinate_old[1])+int(img.shape[1]/2)
            if(0<=x_old<img.shape[0] and 0<=y_old<img.shape[1]):
                img_new[x,y]=img[x_old,y_old]
    return img_new

def main():
    #load image
    lena=cv2.imread('lena.bmp',0)
    #mean and variance
    print(lanzer_mean(lena))
    print(lanzer_variance(lena))
    #rotate image
    rotated_lena=lanzer_rotate(lena,np.pi/6)
    #resize the rotated image by INTER_NEAREST
    resized_rotated_lena_nearest=lanzer_resize_nearest(rotated_lena,[2048,2048])
    cv2.imwrite('lena_rotated_near.bmp',resized_rotated_lena_nearest)
    print('lena rotated near has been OK')
    #resize the rotated image by INTER_LINEAR
    resized_rotated_lena_linear=lanzer_resize_linear(rotated_lena,[2048,2048])
    cv2.imwrite('lena_rotated_linear.bmp',resized_rotated_lena_linear)
    print('lena rotated linear has been OK')
    #resize the rotated image by INTER_CUBIC
    resized_rotated_lena_cubic=lanzer_resize_cubic(rotated_lena,[2048,2048])
    cv2.imwrite('lena_rotated_triple.bmp',resized_rotated_lena_cubic)
    print('lena rotated triple has been OK')
    #load image
    elain=cv2.imread('elain1.bmp',0)
    #rotate image
    rotated_elain=lanzer_rotate(elain,1.5)
    #resize the rotated image by INTER_NEAREST
    resized_rotated_elain_nearest=lanzer_resize_nearest(rotated_elain,[2048,2048])
    cv2.imwrite('elain_rotated_near.bmp',resized_rotated_elain_nearest)
    print('elain rotated near has been OK')
    #resize the rotated image by INTER_LINEAR
    resized_rotated_elain_linear=lanzer_resize_linear(rotated_elain,[2048,2048])
    cv2.imwrite('elain_rotated_linear.bmp',resized_rotated_elain_linear)
    print('elain rotated linear has been OK')
    #resize the rotated image by INTER_CUBIC
    resized_rotated_elain_triple=lanzer_resize_cubic(rotated_elain,[2048,2048])
    cv2.imwrite('elain_rotated_triple.bmp',resized_rotated_elain_triple)
    print('elain rotated triple has been OK')
    #shear image
    shear_lena=lanzer_horizontal_shear(lena,1.5)
    #resize the sheared image by INTER_NEAREST
    resized_shear_lena_nearest=lanzer_resize_nearest(shear_lena,[2048,2048])
    cv2.imwrite('lena_shear_near.bmp',resized_shear_lena_nearest)
    print('lena shear near has been OK')
    #resize the sheared image by INTER_LINEAR
    resized_shear_lena_linear=lanzer_resize_linear(shear_lena,[2048,2048])
    cv2.imwrite('lena_shear_linear.bmp',resized_shear_lena_linear)
    print('lena shear linear has been OK')
    #resize the sheared image by INTER_CUBIC
    resized_shear_lena_cubic=lanzer_resize_cubic(shear_lena,[2048,2048])
    cv2.imwrite('lena_shear_triple.bmp',resized_shear_lena_cubic)
    print('lena shear triple has been OK')
    #load image
    elain=cv2.imread('elain1.bmp',0)
    #shear image
    shear_elain=lanzer_horizontal_shear(elain,1.5)
    #resize the sheared image by INTER_NEAREST
    resized_shear_elain_nearest=lanzer_resize_nearest(shear_elain,[2048,2048])
    cv2.imwrite('elain_shear_near.bmp',resized_shear_elain_nearest)
    print('elain shear near has been OK')
    #resize the sheared image by INTER_LINEAR
    resized_shear_elain_linear=lanzer_resize_linear(shear_elain,[2048,2048])
    cv2.imwrite('elain_shear_linear.bmp',resized_shear_elain_linear)
    print('elain shear linear has been OK')
    #resize the sheared image by INTER_CUBIC
    resized_shear_elain_triple=lanzer_resize_cubic(shear_elain,[2048,2048])
    cv2.imwrite('elain_shear_triple.bmp',resized_shear_elain_triple)
    print('elain shear triple has been OK')

if __name__=='__main__':
    main()

