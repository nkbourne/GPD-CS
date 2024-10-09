import os
import albumentations as A
import cv2
import os, glob
# import torch.utils.data as data


def generate(traindatapath,traintxt):
    files = os.listdir(traindatapath)
    files.sort()
    listText = open(traintxt,'w')
    for file in files:
        filename = os.path.join(traindatapath, file) + '\n'
        listText.write(filename)
    listText.close()

Imagepath="datasets/BSD400"
Trainpath = "datasets/train/"
Traintxt = "datasets/train.txt"

transform = A.Compose([
    A.RandomScale(scale_limit=(0.9,1.5),interpolation=1, always_apply=False, p=0.5),
    A.RandomRotate90(always_apply=False,p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomCrop(width=64,height=64)
])

if __name__ == '__main__':
    oriimages = glob.glob(os.path.join(Imagepath, "*"))
    os.makedirs(Trainpath, exist_ok=True)
    for i in range(len(oriimages)):
        img = cv2.imread(oriimages[i])
        if (img.shape[0]>=64)and(img.shape[1]>=64):
            for j in range(400):
                trans = transform(image=img)
                timg = trans["image"]
                cv2.imwrite(Trainpath+str(i)+"_"+str(j)+".png",timg)
    generate(Trainpath,Traintxt)

