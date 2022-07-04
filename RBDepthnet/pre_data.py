#裁剪图片、路径写入txt
import random
import os
from PIL import Image

def img_proc(srcpath, destpath, list, mode):
    if not os.path.exists(destpath):
        os.makedirs(destpath)
    if mode=='train':
        txt = open(os.path.join(destPath,'train.txt'),'w')
    else:
        txt = open(os.path.join(destPath,'test.txt'),'w')
    for i in range(len(list)):
        prename = os.path.join(srcpath, str(list[i]))
        # srcImg = Image.open(prename + '.jpg')
        # depthImg = Image.open(prename + '.png').convert('L')
        # srcImg = srcImg.transpose(Image.FLIP_TOP_BOTTOM)
        # depthImg = depthImg.transpose(Image.FLIP_TOP_BOTTOM)
        # # srcImg = srcImg.resize((640, 360), Image.ANTIALIAS)
        # # depthImg = depthImg.resize((640, 360), Image.ANTIALIAS)
        # prename = os.path.join(path, str(list[i]))
        # srcImg.save(prename + '.jpg')
        # depthImg.save(prename + '.png')
        txt.write(prename + '.jpg')
        txt.write(' ')
        txt.write(prename + '.png')
        txt.write(' ')
        txt.write('800')
        txt.write('\n')
    txt.close()
    pass

list = [i for i in range(1,6405)]
random.shuffle(list)
srcPath = '/disk1/lcx/fifa'
destPath = '/disk1/lcx/cusdepth_640_360'
trainset = list[:int(6405*0.8)]
testset = list[int(6405*0.8):]
img_proc(srcPath,destPath,trainset,'train')
img_proc(srcPath,destPath,testset,'test')