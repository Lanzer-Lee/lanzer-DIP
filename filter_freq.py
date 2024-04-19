import cv2
import numpy as np
from matplotlib import pyplot as plt

class Filter():
    def __init__(self,img):
        self.shape=2*np.array(img.shape)

    def fft(self,f):
        return np.fft.fftshift(np.fft.fft2(f,self.shape))
    
    def ifft(self,F):
        return np.abs(np.fft.ifft2(np.fft.ifftshift(F)))[:int(self.shape[0]/2),:int(self.shape[1]/2)]
        #return np.abs(np.fft.ifft2(np.fft.ifftshift(F)))
    
    def filter(self,f,method,D0=20,n=None):
        return self.ifft(method(self.fft(f)))

    def ILPF(self,F,D0):
        for x in range(self.shape[0]):
            for y in range(self.shape[1]):
                if((x-self.shape[0]/2)**2+(y-self.shape[1]/2)**2>D0**2):
                    F[x,y]=0
        return F

    def BLPF(self,F,D0,n):
        H=np.zeros(self.shape)
        u0=int(H.shape[0]/2)
        v0=int(H.shape[1]/2)
        for u in range(H.shape[0]):
            for v in range(H.shape[1]):
                H[u,v]=1/(1+((u-u0)**2+(v-v0)**2)**n/D0**(2*n))
        return F*H
    
    def GLPF(self,F,D0):
        H=np.zeros(self.shape)
        u0=int(H.shape[0]/2)
        v0=int(H.shape[1]/2)
        for u in range(H.shape[0]):
            for v in range(H.shape[1]):
                H[u,v]=np.exp(-((u-u0)**2+(v-v0)**2)/2/D0**2)
        return F*H
    
    def IHPL(self,F,D0):
        for x in range(self.shape[0]):
            for y in range(self.shape[1]):
                if((x-self.shape[0]/2)**2+(y-self.shape[1]/2)**2<=D0**2):
                    F[x,y]=0
        return F
    
    def BHPL(self,F,D0,n):
        H=np.zeros(self.shape)
        u0=int(H.shape[0]/2)
        v0=int(H.shape[1]/2)
        for u in range(H.shape[0]):
            for v in range(H.shape[1]):
                if(u==u0 and v==v0):
                    H[u,v]=1
                else:
                    H[u,v]=1/(1+D0**(2*n)/((u-u0)**2+(v-v0)**2)**n)
        return F*H
    
    def GHPL(self,F,D0):
        H=np.zeros(self.shape)
        u0=int(H.shape[0]/2)
        v0=int(H.shape[1]/2)
        for u in range(H.shape[0]):
            for v in range(H.shape[1]):
                H[u,v]=1-np.exp(-((u-u0)**2+(v-v0)**2)/2/D0**2)
        return F*H
    
    def Laplacian(self,F):
        H=np.zeros(self.shape)
        u0=int(H.shape[0]/2)
        v0=int(H.shape[1]/2)
        for u in range(H.shape[0]):
            for v in range(H.shape[1]):
                H[u,v]=-4*(np.pi)**2*(((u-u0)/self.shape[0])**2+((v-v0)/self.shape[1])**2)
        return(1-H)*F
    
    def unmask(self,F,k1=1,k2=1,D0=10):
        H=np.zeros(self.shape)
        u0=int(H.shape[0]/2)
        v0=int(H.shape[1]/2)
        for u in range(H.shape[0]):
            for v in range(H.shape[1]):
                H[u,v]=1-np.exp(-((u-u0)**2+(v-v0)**2)/2/D0**2)
        return (k1+k2*H)*F
    
    def power(self,F):
        p=np.sum(np.abs(F)*np.abs(F))
        return p
    
    def plot_freq(self,F):
        plt.imshow(np.log(np.abs(F)),cmap='gray')
        plt.show()

    def plot_img(self,img):
        plt.imshow(img,cmap='gray')
        plt.show()

def main():
    #filename='test3_corrupt.pgm'
    filename1='test1.pgm'
    filename2='test2.tif'
    filename3='test3_corrupt.pgm'
    filename4='test4 copy.bmp'
    img1=cv2.imread(filename1,0)
    img2=cv2.imread(filename2,0)
    img3=cv2.imread(filename3,0)
    img4=cv2.imread(filename4,0)
    cv2.imwrite('output/test1.bmp',img1)
    cv2.imwrite('output/test2.bmp',img2)
    cv2.imwrite('output/test3.bmp',img3)
    cv2.imwrite('output/test4.bmp',img4)
    f1=Filter(img1)
    f2=Filter(img2)
    f3=Filter(img3)
    f4=Filter(img4)
    #butterworth
    F1=f1.fft(img1)
    F1_B=f1.BLPF(F1,20,1)
    print('power1=',f1.power(F1_B)/f1.power(F1),'\n')
    g1_b=f1.ifft(F1_B)
    cv2.imwrite('output/test1_btw_low.bmp',g1_b)
    F2=f2.fft(img2)
    F2_B=f2.BLPF(F2,20,1)
    print('power2=',f2.power(F2_B)/f2.power(F2),'\n')
    g2_b=f2.ifft(F2_B)
    cv2.imwrite('output/test2_btw_low.bmp',g2_b)
    #gaussian
    F1=f1.fft(img1)
    F1_G=f1.GLPF(F1,20)
    print('power3=',f1.power(F1_G)/f1.power(F1),'\n')
    g1_g=f1.ifft(F1_G)
    cv2.imwrite('output/test1_gaussian_low.bmp',g1_g)
    F2=f2.fft(img2)
    F2_G=f2.GLPF(F2,20)
    print('power4=',f2.power(F2_G)/f2.power(F2),'\n')
    g2_g=f2.ifft(F2_G)
    cv2.imwrite('output/test2_gaussian_low.bmp',g2_g)
    #butterworth
    F3=f3.fft(img3)
    F3_B=f3.BHPL(F3,20,1)
    print('power5=',f3.power(F3_B)/f3.power(F3),'\n')
    g3_b=f3.ifft(F3_B)
    cv2.imwrite('output/test3_btw_high.bmp',g3_b)
    F4=f4.fft(img4)
    F4_B=f4.BHPL(F4,20,1)
    print('power6=',f4.power(F4_B)/f4.power(F4),'\n')
    g4_b=f4.ifft(F4_B)
    cv2.imwrite('output/test4_btw_high.bmp',g4_b)
    #gaussian
    F3=f3.fft(img3)
    F3_G=f3.GHPL(F3,20)
    print('power7=',f3.power(F3_G)/f3.power(F3),'\n')
    g3_g=f3.ifft(F3_G)
    cv2.imwrite('output/test3_gaussian_high.bmp',g3_g)
    F4=f4.fft(img4)
    F4_G=f4.GHPL(F4,20)
    print('power8=',f4.power(F4_G)/f4.power(F4),'\n')
    g4_g=f4.ifft(F4_G)
    cv2.imwrite('output/test4_gaussian_high.bmp',g4_g)
    #laplace
    F3=f3.fft(img3)
    F3_L=f3.Laplacian(F3)
    print('power9=',f3.power(F3_L)/f3.power(F3),'\n')
    g3_l=f3.ifft(F3_L)
    cv2.imwrite('output/test3_laplace_high.bmp',g3_l)
    F4=f4.fft(img4)
    F4_L=f4.Laplacian(F4)
    print('power10=',f4.power(F4_L)/f4.power(F4),'\n')
    g4_l=f4.ifft(F4_L)
    cv2.imwrite('output/test4_laplace_high.bmp',g4_l)
    #unmask
    F3=f3.fft(img3)
    F3_L=f3.unmask(F3)
    print('power11=',f3.power(F3_L)/f3.power(F3),'\n')
    g3_l=f3.ifft(F3_L)
    cv2.imwrite('output/test3_unmask_high.bmp',g3_l)
    F4=f4.fft(img4)
    F4_L=f4.unmask(F4)
    print('power12=',f4.power(F4_L)/f4.power(F4),'\n')
    g4_l=f4.ifft(F4_L)
    cv2.imwrite('output/test4_unmask_high.bmp',g4_l)
    #IMG=filter.fft(img)
    #IMG=filter.GHPL(IMG,0.01)
    #img2=filter.ifft(IMG)
    #filter.plot_img(img2)
    #filter.plot_freq(IMG)

    

    
if __name__=='__main__':
    main()
    
    
    