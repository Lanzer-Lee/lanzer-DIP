import numpy as np
import cv2
from matplotlib import pyplot as plt

class Noise():
    def __init__(self):
        pass

    def gaussian(self,img,mean,sigma):
        noise=np.random.normal(mean,sigma,img.shape)
        return img+noise
    
    def saltpepper(self,img,ps,pp):
        #salt=np.random.rand(img.shape)
        salt_pepper=np.random.rand(img.shape[0],img.shape[1])
        output=np.zeros(img.shape,dtype=np.uint8)
        for x in range(output.shape[0]):
            for y in range(output.shape[1]):
                if(salt_pepper[x,y]<=ps):
                    output[x,y]=255
                elif(salt_pepper[x,y]>=1-pp):
                    output[x,y]=0
                else:
                    output[x,y]=img[x,y]
        return output
    
    def move(self,img,a,b,T):
        F=np.fft.fftshift(np.fft.fft2(img))
        H=np.zeros(F.shape,dtype=F.dtype)
        G=np.zeros(F.shape,dtype=F.dtype)
        for u in range(0,F.shape[0]):
            for v in range(0,F.shape[1]):
                if(u>0 or v>0):
                    H[u,v]=(T/(np.pi*(a*u+b*v)))*np.sin(np.pi*(a*u+b*v))*np.exp(-np.pi*(u*a+v*b)*complex(0,1))
                else:
                    H[u,v]=T
                G[u,v]=H[u,v]*F[u,v]
        return np.fft.ifft2(np.fft.ifftshift(G))

class RestoreFilter():
    def __init__(self) -> None:
        pass

    def arithmetic_mean_filter(self,img,kernal_size):
        padded_img=self.padding(img,np.uint((np.array(kernal_size)-1)/2),0)
        output=np.zeros(img.shape)
        for x in range(img.shape[0]):
            for y in range(img.shape[1]):
                output[x,y]=(padded_img[x:x+kernal_size[0],y:y+kernal_size[1]]).sum()
        return output/(kernal_size[0]*kernal_size[1])
    
    def geometric_mean_filter(self,img,kernal_size):
        padded_img=self.padding(img,np.uint((np.array(kernal_size)-1)/2),1)
        output=np.zeros(img.shape)
        for x in range(output.shape[0]):
            for y in range(output.shape[1]):
                output[x,y]=((padded_img[x:x+kernal_size[0],y:y+kernal_size[1]]**(1/(kernal_size[0]*kernal_size[1]))).prod())
        return np.uint8(output)
    
    def harmonic_averaging_filter(self,img,kernal_size):
        padded_img=self.padding(img,np.uint((np.array(kernal_size)-1)/2),1)
        padded_img=1/padded_img
        output=np.zeros(img.shape)
        for x in range(output.shape[0]):
            for y in range(output.shape[1]):
                output[x,y]=(kernal_size[0]*kernal_size[1])/((padded_img[x:x+kernal_size[0],y:y+kernal_size[1]]).sum())
        return np.uint8(output)
    
    def anti_harmonic_averaging_filter(self,img,kernal_size,Q):
        padded_img=self.padding(img,np.uint((np.array(kernal_size)-1)/2),1)
        output=np.zeros(img.shape)
        for x in range(output.shape[0]):
            for y in range(output.shape[1]):
                output[x,y]=((padded_img[x:x+kernal_size[0],y:y+kernal_size[1]]**(Q+1)).sum())/((padded_img[x:x+kernal_size[0],y:y+kernal_size[1]]**Q).sum())
        return np.uint8(output)
    
    def median_filter(self,img,kernal_size):
        padded_img=self.padding(img,np.uint((np.array(kernal_size)-1)/2),1)
        output=np.zeros(img.shape)
        for x in range(output.shape[0]):
            for y in range(output.shape[1]):
                flatten_kernal=np.sort(padded_img[x:x+kernal_size[0],y:y+kernal_size[1]])
                output[x,y]=flatten_kernal[1,1]
        return output
    
    def max_filter(self,img,kernal_size):
        padded_img=self.padding(img,np.uint((np.array(kernal_size)-1)/2),1)
        output=np.zeros(img.shape)
        for x in range(output.shape[0]):
            for y in range(output.shape[1]):
                output[x,y]=np.max(padded_img[x:x+kernal_size[0],y:y+kernal_size[1]])
        return output

    def min_filter(self,img,kernal_size):
        padded_img=self.padding(img,np.uint((np.array(kernal_size)-1)/2),1)
        output=np.zeros(img.shape)
        for x in range(output.shape[0]):
            for y in range(output.shape[1]):
                output[x,y]=np.min(padded_img[x:x+kernal_size[0],y:y+kernal_size[1]])
        return output   
    
    def winner_filter(self,img,K,a,b,T):
        G=np.fft.fftshift(np.fft.fft2(img))
        H=np.zeros(G.shape,dtype=G.dtype)
        F=np.zeros(G.shape,dtype=G.dtype)
        for u in range(H.shape[0]):
            for v in range(H.shape[1]):
                if(u>0 or v>0):
                    H[u,v]=(T/(np.pi*(a*u+b*v)))*np.sin(np.pi*(a*u+b*v))*np.exp(-np.pi*(u*a+v*b)*complex(0,1))
                else:
                    H[u,v]=T
        F=(np.abs(H)**2/(np.abs(H)**2+K))*G/H
        return np.fft.ifft2(np.fft.ifftshift(F))
    
    def padding(self,img,padding_size,padding_mode):
        if(padding_mode==0):
            padded_img=np.zeros([img.shape[0]+2*padding_size[0],img.shape[1]+2*padding_size[1]])
        elif(padding_mode==1):
            padded_img=np.ones([img.shape[0]+2*padding_size[0],img.shape[1]+2*padding_size[1]])
        padded_img[padding_size[0]:padding_size[0]+img.shape[0],padding_size[1]:padding_size[1]+img.shape[1]]=img
        return padded_img
    
class Image():
    def __init__(self) -> None:
        pass

    def load_image(self,filename):
        return cv2.imread(filename,0)
    
    def plot_image(self,img):
        plt.imshow(np.clip(img/255.0,0.0,1.0),cmap='gray')
        plt.show()

    def plot_image_complex(self,img):
        plt.imshow(np.clip(np.abs(img)/255.0,0.0,1.0),cmap='gray')
        plt.show()

    def plot_image_cv(self,img):
        cv2.imshow('image',img)
        

def test1():
    noise=Noise()
    image=Image()
    restore=RestoreFilter()
    filename='lena.bmp'
    lena=image.load_image(filename)
    gaussian_lena=noise.gaussian(lena,0,10)
    p_lena=restore.anti_harmonic_averaging_filter(gaussian_lena,[3,3],1)
    n_lena=restore.anti_harmonic_averaging_filter(gaussian_lena,[3,3],-1)
    a_lena=restore.arithmetic_mean_filter(gaussian_lena,[3,3])
    h_lena=restore.harmonic_averaging_filter(gaussian_lena,[3,3])
    g_lena=restore.geometric_mean_filter(gaussian_lena,[3,3])
    max_lena=restore.max_filter(gaussian_lena,[3,3])
    min_lena=restore.min_filter(gaussian_lena,[3,3])
    median_lena=restore.median_filter(gaussian_lena,[3,3])
    image.plot_image(gaussian_lena)
    cv2.imwrite('output/gaussian_lena.bmp',gaussian_lena)
    image.plot_image(p_lena)
    cv2.imwrite('output/p_harmonic_lena.bmp',p_lena)
    image.plot_image(n_lena)
    cv2.imwrite('output/n_harmonic_lena.bmp',n_lena)
    image.plot_image(a_lena)
    cv2.imwrite('output/arithmetic_mean_lena.bmp',a_lena)
    image.plot_image(h_lena)
    cv2.imwrite('output/harmonic_mean_lena.bmp',h_lena)
    image.plot_image(g_lena)
    cv2.imwrite('output/geometric_mean_lena.bmp',g_lena)
    image.plot_image(max_lena)
    cv2.imwrite('output/max_lena.bmp',max_lena)
    image.plot_image(min_lena)
    cv2.imwrite('output/min_lena.bmp',min_lena)
    image.plot_image(median_lena)
    cv2.imwrite('output/median_lena.bmp',median_lena)

def test2():
    noise=Noise()
    image=Image()
    restore=RestoreFilter()
    filename='lena.bmp'
    lena=image.load_image(filename)
    salt_pepper_lena=noise.saltpepper(lena,0.1,0.1)
    p_lena=restore.anti_harmonic_averaging_filter(salt_pepper_lena,[3,3],1)
    n_lena=restore.anti_harmonic_averaging_filter(salt_pepper_lena,[3,3],-1)
    a_lena=restore.arithmetic_mean_filter(salt_pepper_lena,[3,3])
    h_lena=restore.harmonic_averaging_filter(salt_pepper_lena,[3,3])
    g_lena=restore.geometric_mean_filter(salt_pepper_lena,[3,3])
    max_lena=restore.max_filter(salt_pepper_lena,[3,3])
    min_lena=restore.min_filter(salt_pepper_lena,[3,3])
    median_lena=restore.median_filter(salt_pepper_lena,[3,3])
    image.plot_image(salt_pepper_lena)
    cv2.imwrite('output2/salt_pepper_lena.bmp',salt_pepper_lena)
    image.plot_image(p_lena)
    cv2.imwrite('output2/p_harmonic_lena.bmp',p_lena)
    image.plot_image(n_lena)
    cv2.imwrite('output2/n_harmonic_lena.bmp',n_lena)
    image.plot_image(a_lena)
    cv2.imwrite('output2/arithmetic_mean_lena.bmp',a_lena)
    image.plot_image(h_lena)
    cv2.imwrite('output2/harmonic_mean_lena.bmp',h_lena)
    image.plot_image(g_lena)
    cv2.imwrite('output2/geometric_mean_lena.bmp',g_lena)
    image.plot_image(max_lena)
    cv2.imwrite('output2/max_lena.bmp',max_lena)
    image.plot_image(min_lena)
    cv2.imwrite('output2/min_lena.bmp',min_lena)
    image.plot_image(median_lena)
    cv2.imwrite('output2/median_lena.bmp',median_lena)

def test3():
    noise=Noise()
    image=Image()
    restore=RestoreFilter()
    filename='lena.bmp'
    lena=image.load_image(filename)
    salt_lena=noise.saltpepper(lena,0.1,0)
    pepper_lena=noise.saltpepper(lena,0,0.1)
    p_salt_lena=restore.anti_harmonic_averaging_filter(salt_lena,[3,3],1)
    n_salt_lena=restore.anti_harmonic_averaging_filter(salt_lena,[3,3],-1)
    p_pepper_lena=restore.anti_harmonic_averaging_filter(pepper_lena,[3,3],1)
    n_pepper_lena=restore.anti_harmonic_averaging_filter(pepper_lena,[3,3],-1)
    image.plot_image(salt_lena)
    cv2.imwrite('output3/salt_lena.bmp',salt_lena)
    image.plot_image(pepper_lena)
    cv2.imwrite('output3/pepper_lena.bmp',pepper_lena)
    image.plot_image(p_salt_lena)
    cv2.imwrite('output3/p_salt_lena.bmp',p_salt_lena)
    image.plot_image(n_salt_lena)
    cv2.imwrite('output3/n_salt_lena.bmp',n_salt_lena)
    image.plot_image(p_pepper_lena)
    cv2.imwrite('output3/p_pepper_lena.bmp',p_pepper_lena)
    image.plot_image(n_pepper_lena)
    cv2.imwrite('output3/n_pepper_lena.bmp',n_pepper_lena)

def test4():
    noise=Noise()
    image=Image()
    restore=RestoreFilter()
    filename='lena.bmp'
    lena=image.load_image(filename)
    moved_lena=noise.move(lena,0.1,0.1,1)
    gaussian_moved_lena=noise.gaussian(moved_lena,0,0.1)
    restored_lena=restore.winner_filter(gaussian_moved_lena,1e-13,0.1,0.1,1)
    image.plot_image_complex(moved_lena)
    cv2.imwrite('output4/moved_lena.bmp',np.uint8(np.abs(moved_lena)))
    image.plot_image_complex(gaussian_moved_lena)
    cv2.imwrite('output4/gaussian_moved_lena.bmp',np.uint8(np.abs(gaussian_moved_lena)))
    image.plot_image_complex(restored_lena)
    cv2.imwrite('output4/restored_lena.bmp',np.uint8(np.abs(restored_lena)))

def test5():
    noise=Noise()
    image=Image()
    restore=RestoreFilter()
    filename='lena.bmp'
    lena=image.load_image(filename)
    gaussina_lena1=noise.gaussian(lena,0,1)
    gaussina_lena2=noise.gaussian(lena,0,10)
    gaussina_lena3=noise.gaussian(lena,0,30)
    image.plot_image(gaussina_lena1)
    cv2.imwrite('output5/gaussina_lena1.bmp',gaussina_lena1)
    image.plot_image(gaussina_lena2)
    cv2.imwrite('output5/gaussina_lena2.bmp',gaussina_lena2)
    image.plot_image(gaussina_lena3)
    cv2.imwrite('output5/gaussina_lena3.bmp',gaussina_lena3)

def temp():
    noise=Noise()
    image=Image()
    restore=RestoreFilter()
    filename='temp.png'
    lena=image.load_image(filename)
    restored_lena=restore.min_filter(lena,[3,3])
    #restored_lena=restore.min_filter(restored_lena,[3,3])
    
    restored_lena=restore.anti_harmonic_averaging_filter(restored_lena,[3,3],-3)
    #restored_lena=restore.median_filter(restored_lena,[3,3])
    restored_lena=restore.max_filter(restored_lena,[3,3])
    #restored_lena=restore.max_filter(restored_lena,[3,3])

    restored_lena=np.uint8(restored_lena)
    cv2.imshow('before',lena)
    cv2.imshow('after',restored_lena)
    cv2.waitKey(0)

def main():
    temp()

if __name__=='__main__':
    main()