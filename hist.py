import numpy as np
from matplotlib import pyplot as plt
import cv2
import os

def load_img(filename):
    img=cv2.imread(filename,0)
    return img

def plot_img_histogram(img,filename):
    plt.hist(img.ravel(),bins=256,range=[0,255])
    plt.show()
    plt.savefig(filename)

def hist_equal(img):
    equaled_img=np.zeros(img.shape)
    pixel_num=np.zeros([256,1])
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            pixel_num[img[x,y],0]+=1
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            equaled_img[x,y]=np.sum(pixel_num[:img[x,y],0])
    return np.uint8(equaled_img*255/img.shape[0]/img.shape[1])

def generate_template(img):
    template=np.zeros([256,1],dtype=np.float64)
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            template[img[x,y],0]+=1
    return template/img.shape[0]/img.shape[1] 

def hist_match(img,template):
    matched_img=np.zeros(img.shape)
    pixel_num=np.zeros([256,1])
    G=np.zeros([256,1])
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            pixel_num[img[x,y],0]+=1
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            matched_img[x,y]=np.sum(pixel_num[:img[x,y],0])
    matched_img=np.uint8(matched_img*255/img.shape[0]/img.shape[1])
    for q in range(256):
        G[q,0]=np.sum(template[:q,0])
    G=np.uint8(G*255)
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            for q in range(256):
                if(matched_img[x,y]==G[q,0]):
                    matched_img[x,y]=q
    return matched_img

def img_mean_variance(img):
    (img_mean,img_var)=cv2.meanStdDev(img)
    return img_mean,img_var

def img_padding(img,padding_size):
    padded_img=np.zeros([img.shape[0]+2*padding_size,img.shape[1]+2*padding_size],dtype=np.uint8)
    padded_img[padding_size:padding_size+img.shape[0],padding_size:padding_size+img.shape[1]]=img
    return padded_img

def part_hist_hance(img,k0=0.5,k1=0.01,k2=0.5,E=3,masksize=7):
    hanced_img=np.zeros(img.shape)
    mean_global,var_global=img_mean_variance(img)
    padded_img=img_padding(img,3)
    for x in range(hanced_img.shape[0]):
        for y in range(hanced_img.shape[1]):
            mean_part,var_part=img_mean_variance(padded_img[x:x+masksize+1,y:y+masksize+1])
            if(mean_part<=k0*mean_global and k1*np.sqrt(var_global)<=np.sqrt(var_part) and np.sqrt(var_part)<=k2*np.sqrt(var_global)):
                hanced_img[x,y]=E*img[x,y]
            else:
                hanced_img[x,y]=img[x,y]
    return np.uint8(hanced_img)

def img_segment(img,T0=0.1):
    T_new,temp=img_mean_variance(img)
    T_old=0
    segmented_img=np.zeros(img.shape,dtype=np.uint8)
    while(T_new-T_old>T0):
        G1=np.zeros(img.shape)
        pixel_num1=0
        G2=np.zeros(img.shape)
        pixel_num2=0
        for x in range(img.shape[0]):
            for y in range(img.shape[1]):
                if(img[x,y]>T_new):
                    G1[x,y]=img[x,y]
                    pixel_num1+=1
                else:
                    G2[x,y]=img[x,y]
                    pixel_num2+=1
        if(pixel_num1==0):
            mean1=0
        else:
            mean1=np.sum(G1)/pixel_num1
        if(pixel_num2==0):   
            mean2=0
        else:
            mean2=np.sum(G2)/pixel_num2
        T_old=T_new
        T_new=(mean1+mean2)*0.5
        if(T_new-T_old<T0):
            for x in range(G1.shape[0]):
                for y in range(G1.shape[1]):
                    if(G1[x,y]>0):
                        segmented_img[x,y]=255
            return segmented_img
    return segmented_img 

    

def main():
    i=0
    template=generate_template(load_img('figure/citywall.bmp'))
    for root,dirs,files in os.walk('figure'):
        for file in files:
            filename=os.path.join(root,file)
            save_file=os.path.join(root,'output')
            figure=load_img(filename)
            plot_img_histogram(figure,save_file+'/hist'+str(i)+'.png')
            equaled_figure=hist_equal(figure)
            cv2.imwrite(save_file+'/equal'+str(i)+'.bmp',equaled_figure)
            matched_figure=hist_match(figure,template)
            cv2.imwrite(save_file+'/match'+str(i)+'.bmp',matched_figure)
            hanced_figure=part_hist_hance(figure)
            cv2.imwrite(save_file+'/hance'+str(i)+'.bmp',hanced_figure)
            segged_figure=img_segment(figure)
            cv2.imwrite(save_file+'/seg'+str(i)+'.bmp',segged_figure)
            i+=1

if __name__=='__main__':
    main()


