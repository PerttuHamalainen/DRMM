from PIL import Image
import numpy as np

def crop(imgName,bgColor=255.0,epsilon=2):
    img = np.array(Image.open(imgName))
    bw=np.mean(img,axis=2)
    if bgColor is None:
        bgColor=np.median(bw)
        epsilon=np.std(bw)*0.01

    rowMeans=np.mean(bw,axis=1)
    #print("rowMeans",rowMeans)
    rowContents=np.clip(np.sign(np.abs(bgColor-rowMeans)-epsilon),0,1)
    colMeans=np.mean(bw,axis=0)
    #print("colMeans",colMeans)
    colContents=np.clip(np.sign(np.abs(bgColor-colMeans)-epsilon),0,1)

    #crop left
    for i in range(img.shape[1]):
        if colContents[i]!=0:
            img=img[:,i:,:]
            colContents=colContents[i:]
            break

    #crop right
    for i in reversed(range(img.shape[1])):
        if colContents[i]!=0:
            img=img[:,:i,:]
            break

    #crop top
    for i in range(img.shape[0]):
        if rowContents[i]!=0:
            img=img[i:,:,:]
            rowContents=rowContents[i:]
            break

    #crop bottom
    for i in reversed(range(img.shape[0])):
        if rowContents[i]!=0:
            img=img[:i,:,:].copy()
            break
    Image.fromarray(img).save(imgName)


