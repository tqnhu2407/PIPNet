import os
import shutil
import scipy.io
import numpy as np
import time
from PIL import Image
import xml.etree.ElementTree as ET

def get_images():
    file_list = scipy.io.loadmat('./data/stanford_dogs/file_list.mat')
    id = 0
    images = []
    for i in range(len(file_list['file_list'])):
        name = file_list['file_list'][i][0][0]
        id += 1
        images.append([str(id) + ' ' + name])
    return images

def get_split():
    file_list = scipy.io.loadmat('./data/stanford_dogs/file_list.mat')
    train_list = scipy.io.loadmat('./data/stanford_dogs/train_list.mat')
    test_list = scipy.io.loadmat('./data/stanford_dogs/test_list.mat')

    name2id = {}
    id = 0
    for i in range(len(file_list['file_list'])):
        name = file_list['file_list'][i][0][0]
        id += 1
        name2id[name] = str(id)
    
    split = []
    for e in train_list['file_list']:
        name = e[0][0]
        id = name2id[name]
        split.append([id + ' 1'])
    for e in test_list['file_list']:
        name = e[0][0]
        id = name2id[name]
        split.append([id + ' 0'])
    split = sorted(split, key=lambda x: int(x[0].split(' ')[0]))

    return split

def get_bboxes():
    file_list = scipy.io.loadmat('./data/stanford_dogs/file_list.mat')
    name2id = {}
    id = 0
    for i in range(len(file_list['file_list'])):
        name = file_list['file_list'][i][0][0]
        id += 1
        name2id[name] = str(id)

    file_list = scipy.io.loadmat('./data/stanford_dogs/file_list.mat')
    annotation_list = file_list['annotation_list']
    annotation_paths = [_[0][0] for _ in annotation_list]
    bboxes = dict()
    for annotation_file in annotation_paths:
        filepath = os.path.join('./data/stanford_dogs/Annotation/', annotation_file)
        with open(filepath, 'r') as f:
            annotation = f.read()
        root = ET.fromstring(annotation)
        for bndbox in root.findall("./object/bndbox"):
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmin').text)
            ymax = float(bndbox.find('ymin').text)      
            width = xmax - xmin
            height = ymax - ymin      
        id = name2id[annotation_file + '.jpg']
        bboxes[int(id)] = (xmin, ymin, width, height)
    return bboxes

def preprocess_dog():
    path = './data/stanford_dogs/'

    time_start = time.time()

    train_save_path = os.path.join(path,'dataset/train_crop/')
    test_save_path = os.path.join(path,'dataset/test_crop/')

    images = get_images()
    split = get_split()
    bboxes = get_bboxes()

    num = len(images)
    for k in range(num):
        id, fn = images[k][0].split(' ')
        id = int(id)
        file_name = fn.split('/')[0]
        if int(split[k][0][-1]) == 1:
            
            if not os.path.isdir(train_save_path + file_name):
                os.makedirs(os.path.join(train_save_path, file_name))
            img = Image.open(os.path.join(os.path.join(path, 'Images'),images[k][0].split(' ')[1])).convert('RGB')
            x, y, w, h = bboxes[id]
            cropped_img = img.crop((x, y, x+w, y+h))
            cropped_img.save(os.path.join(os.path.join(train_save_path,file_name),images[k][0].split(' ')[1].split('/')[1]))
            print('%s' % images[k][0].split(' ')[1].split('/')[1])
        else:
            if not os.path.isdir(test_save_path + file_name):
                os.makedirs(os.path.join(test_save_path,file_name))
            img = Image.open(os.path.join(os.path.join(path, 'Images'),images[k][0].split(' ')[1])).convert('RGB')
            x, y, w, h = bboxes[id]
            cropped_img = img.crop((x, y, x+w, y+h))
            cropped_img.save(os.path.join(os.path.join(test_save_path,file_name),images[k][0].split(' ')[1].split('/')[1]))
            print('%s' % images[k][0].split(' ')[1].split('/')[1])

    train_save_path = os.path.join(path,'dataset/train/')
    test_save_path = os.path.join(path,'dataset/test_full/')

    num = len(images)
    for k in range(num):
        id, fn = images[k][0].split(' ')
        id = int(id)
        file_name = fn.split('/')[0]
        if int(split[k][0][-1]) == 1:
            
            if not os.path.isdir(train_save_path + file_name):
                os.makedirs(os.path.join(train_save_path, file_name))
            img = Image.open(os.path.join(os.path.join(path, 'images'),images[k][0].split(' ')[1])).convert('RGB')
            width, height = img.size
        
            img.save(os.path.join(os.path.join(train_save_path,file_name),images[k][0].split(' ')[1].split('/')[1]))
            
            print('%s' % images[k][0].split(' ')[1].split('/')[1])
        else:
            if not os.path.isdir(test_save_path + file_name):
                os.makedirs(os.path.join(test_save_path,file_name))
            shutil.copy(path + 'images/' + images[k][0].split(' ')[1], os.path.join(os.path.join(test_save_path,file_name),images[k][0].split(' ')[1].split('/')[1]))
            print('%s' % images[k][0].split(' ')[1].split('/')[1])
    time_end = time.time()
    print('DOGS, %s!' % (time_end - time_start))
    return

def explore_dogs():
    return

if __name__ == "__main__":
    preprocess_dog()