import numpy as np
import cv2
import os

class Filter():
    def __init__(self,kernal_size):
        self.size=np.uint8(kernal_size)
        self.kernal=np.zeros(kernal_size)
        self.padding=int((kernal_size[0]-1)/2)

    def corr2d(self,img):
        output=np.zeros(img.shape)
        padded_img=np.zeros([int(img.shape[0]+2*self.padding),int(img.shape[1]+2*self.padding)])
        padded_img[self.padding:img.shape[0]+self.padding,self.padding:img.shape[1]+self.padding]=img
        for x in range(output.shape[0]):
            for y in range(output.shape[1]):
                output[x,y]=(padded_img[x:x+self.size[0],y:y+self.size[1]]*self.kernal).sum()
        return np.uint8(output)

    def gaussian(self,img,K,sigma):
        self.kernal=np.zeros(self.size)
        x0=(self.size[0]-1)/2
        y0=(self.size[1]-1)/2
        for x in range(self.size[0]):
            for y in range(self.size[1]):
                self.kernal[x,y]=K*np.exp(-((x-x0)**2+(y-y0)**2)/2/sigma**2)
        self.kernal=self.kernal/self.kernal.sum()
        return self.corr2d(img)
    
    def gaussian_plus(self,img,K,sigma,rule):
        if rule=='68':
            self.size[0]=int(2*sigma+1)
            self.size[1]=self.size[0]
        if rule=='95':
            self.size[0]=int(4*sigma+1)
            self.size[1]=self.size[0]
        if rule=='99.7':
            self.size[0]=int(6*sigma+1)
            self.size[1]=self.size[0]
        if(self.size[0]%2==0):
            self.size+=1
        self.padding=int((self.size[0]-1)/2)
        return self.gaussian(img,K,sigma)
    
    def meadian(self,img):
        self.kernal=np.ones(self.size)/(self.size[0]*self.size[1])
        return self.corr2d(img)
    
    def unsharp(self,img,k):
        return np.uint8(img+k*(img-self.gaussian(img,1,1)))
    
class Filter33(Filter):
    def __init__(self):
        super().__init__(np.array([3,3]))

    def laplace_simple(self,img):
        self.kernal=np.array([[0,1,0],[1,-4,1],[0,1,0]])
        return self.corr2d(img)
    
    def laplace_trig(self,img):
        self.kernal=np.array([[1,1,1],[1,-8,1],[1,1,1]])
        return self.corr2d(img)
    
    def laplace_another1(self,img):
        self.kernal=np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
        return self.corr2d(img)
    
    def laplace_another2(self,img):
        self.kernal=np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
        return self.corr2d(img)
    
    def sobel(self,img):
        self.kernal=np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
        img=self.corr2d(img)
        self.kernal=np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
        return self.corr2d(img)
    
def load_img(filename):
    return cv2.imread(filename,0)

def main():
    filter33=Filter(np.array([3,3]))
    filter55=Filter(np.array([5,5]))
    filter77=Filter(np.array([7,7]))
    hign_filter=Filter33()
    for root,dirs,files in os.walk('figure'):
        for file in files:
            filename=os.path.join(root,file)
            image=load_img(filename)  
            gaussian33_image=filter33.gaussian(image,1,1)
            meadian33_image=filter33.meadian(image)
            unsharp_image=filter33.unsharp(image,0.2)
            gaussian55_image=filter55.gaussian(image,1,1)
            meadian55_image=filter55.meadian(image)
            gaussian77_image=filter77.gaussian(image,1,1)
            meadian77_image=filter77.meadian(image)
            gaussian68_image=filter33.gaussian_plus(image,1,1,'68')
            gaussian95_image=filter33.gaussian_plus(image,1,1,'95')
            gaussian68_image=filter33.gaussian_plus(image,1,1,'99.7')
            laplace_image=hign_filter.laplace_simple(image)
            sobel_image=hign_filter.sobel(image)
            cv2.imshow('1',gaussian33_image)
            cv2.imshow('2',meadian33_image)
            cv2.imshow('3',unsharp_image)
            cv2.imshow('4',gaussian55_image)
            cv2.imshow('5',meadian55_image)
            cv2.imshow('6',gaussian77_image)
            cv2.imshow('7',meadian77_image)
            cv2.imshow('8',gaussian68_image)
            cv2.imshow('9',gaussian95_image)
            cv2.imshow('10',laplace_image)
            cv2.imshow('11',sobel_image)
            cv2.waitKey(0)
         


            

    filename='test4 copy.bmp'
    #filter33=Filter(np.array([3,3]))
    filter33=Filter33()
    test1=load_img(filename)
    #print(filter33.padding)
    #print(int(test1.shape[0]+2*filter33.padding))
    #print(test1.shape)
    #zero_test1=filter33.corr2d(test1)
    #print(zero_test1)
    meadian_test1=filter33.sobel(test1)
    cv2.imwrite('temp.bmp',meadian_test1)
    cv2.imshow('before',test1)
    cv2.imshow('after',meadian_test1)
    cv2.waitKey(0)


if __name__=="__main__":
    main()
        
