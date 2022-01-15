import os
import sys
# sys.path.append('..')
import random
import math
import cv2
import numpy as np
from torch.utils import data

try:
    from .dataLAPA106 import transformerr, category_ids, square_box
except:
    from dataLAPA106 import transformerr, category_ids, square_box

class FaceLMHorizontalFlip(object):
    def __init__(self, p, points_flip):
        super().__init__()
        self.p = p
        self.points_flip = points_flip
    
    def __call__(self, image, target):
        if random.random() > self.p:
            image = cv2.flip(image, 1)
            # image = image.transpose(Image.FLIP_LEFT_RIGHT)
            target = target[self.points_flip, :]
            target[:,0] = 1-target[:,0]
            return image, target
        else:
            return image, target

points_flip = [17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 28, 29, 30, 31, 36, 35, 34, 33, 32, 46, 45, 44, 43, 48, 47, 40, 39, 38, 37, 42, 41, 55, 54, 53, 52, 51, 50, 49, 60, 59, 58, 57, 56, 65, 64, 63, 62, 61, 68, 67, 66]
points_flip = (np.array(points_flip)-1).tolist()

class F300WDataset(data.Dataset):
    TARGET_IMAGE_SIZE = (256, 256)
    
    def __init__(self, data_dir, split, augment=False, transforms=None):
        self.data_dir = data_dir
        
        self.augment = augment
        self.transforms = transforms
        self.flip_aug = FaceLMHorizontalFlip(p=0.5, points_flip=points_flip)

        split_filename = os.path.join(self.data_dir, split + ".txt")
        with open(split_filename, "r") as f:
            self.img_path_list = [x.strip() for x in f.readlines()]
    
    def __len__(self):
        return len(self.img_path_list)

    def _get_68_landmark(self, path):
        landmark = None
        with open(path, "r") as ff:
            landmark = [line.strip() for line in ff.readlines()[3:-1]]
            landmark = [[float(x) for x in pt.split()] for pt in landmark]
        landmark = np.array(landmark)
        assert len(landmark)==68, "There should be 68 landmarks. Get {len(lm)}"
        return landmark
    
    def lmks2box(self, lmks, img, expand_forehead=0.2):
        xy = np.min(lmks, axis=0).astype(np.int32) 
        zz = np.max(lmks, axis=0).astype(np.int32)

        # Expand forehead
        expand = (zz[1] - xy[1]) * expand_forehead
        xy[1] -= expand
        xy[1] = max(xy[1], 0) 

        wh = zz - xy + 1

        center = (xy + wh/2).astype(np.int32)
        boxsize = int(np.max(wh)*1.1)
        xy = center - boxsize//2
        x1, y1 = xy
        x2, y2 = xy + boxsize
        height, width, _ = img.shape
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)
        return [x1, y1, x2, y2]

    def __get_default_item(self):
        return self.__getitem__(0)

    def __getitem__(self, index):
        image_path = self.img_path_list[index]
        image_path = os.path.join(self.data_dir, "images", image_path)

        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Get 68 landmarks
        anno_path = image_path[:-4] + ".pts"
        landmark = self._get_68_landmark(anno_path)
        expand_random = random.uniform(0.1, 0.17)
        box = self.lmks2box(landmark, img, expand_forehead=expand_random)


        # If fail then get the default item
        # if np.min(landmark[:,0]) < 0 or \
        #    np.min(landmark[:,1]) < 0 or \
        #    np.max(landmark[:,0]) >= img.shape[1]  or \
        #    np.max(landmark[:,1]) >= img.shape[0] :
        #    print("Get default itemmmmmmmmmmmmmmmmm!")
        #    return self.__get_default_item()
        # landmark[landmark < 0] = 0

        # Round box in case box out of range
        x1, y1, x2, y2 = box
        x1 = max(x1, 0)
        y1 = max(y1, 0)
        x2 = min(x2, img.shape[1]-1)
        y2 = min(y2, img.shape[0]-1)
        box = [x1, y1, x2, y2]

        if self.augment:
            # Flip augmentation
            img, landmark = self.flip_aug(img, landmark)

            # Mask for invalid landmark
            mask_landmark = np.zeros_like(landmark)
            mask_landmark[landmark < 0] = 1.0
            mask_landmark[:, 0][landmark[:, 0] >= img.shape[1]] = 1.0
            mask_landmark[:, 1][landmark[:, 1] >= img.shape[0]] = 1.0

            # Clean landmark before do augmentation
            clean_landmark = landmark.copy()
            clean_landmark[clean_landmark < 0] = 0
            clean_landmark[:, 0][clean_landmark[:, 0] >= img.shape[1]] = img.shape[1] - 1
            clean_landmark[:, 1][clean_landmark[:, 1] >= img.shape[0]] = img.shape[0] - 1

            # Do augmentation
            transformed = transformerr(image=img, bboxes=[box], category_ids=category_ids, keypoints=clean_landmark)
            imgT = np.array(transformed["image"])
            boxes = np.array(transformed["bboxes"])
            lmks = np.array(transformed["keypoints"])

            lmks_ok = (lmks.shape == landmark.shape)
            box_ok = len(boxes) >0
            augment_sucess = lmks_ok and box_ok
            if (augment_sucess):
                imgT = imgT
                box = boxes[0]
                # Update invalid landmark
                lmks[mask_landmark > 0] = landmark[mask_landmark > 0]
                lmks = lmks
            else:
                # print("Augment not success!!!!!!!!")
                imgT = img
                box = box
                lmks = landmark
        else:
            imgT = img
            box = box
            lmks = landmark

        assert  (lmks.shape == landmark.shape), f'Lmks Should have shape {landmark.shape}'

        expand_random = random.uniform(1.0, 1.1)
        # print("Expand: ", expand_random)
        box = square_box(box, imgT.shape, lmks, expand=expand_random)
        x1, y1, x2, y2 = list(map(math.ceil, box))
        imgT = imgT[y1:y2, x1:x2]

        lmks[:,0], lmks[:,1] = (lmks[: ,0] - x1)/imgT.shape[1] ,\
                               (lmks[:, 1] - y1)/imgT.shape[0]

        imgT = cv2.resize(imgT, self.TARGET_IMAGE_SIZE)
        augment_sucess = (lmks.shape == landmark.shape) and ((lmks >= 0).all())\
                             and ((lmks < 1).all())

        if self.transforms is not None:
            imgT = self.transforms(imgT)  # Normalize, et
        
        return imgT, lmks

if __name__ == "__main__":
    from torchvision import  transforms

    # transform = transforms.Compose([transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])

    f300w = F300WDataset(data_dir='/home/ubuntu/data/300W', split='train', augment=True, transforms=None)

    print(len(f300w))
    count = 0
    for img, landmarks in f300w:
        print(img.shape, landmarks.shape)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # import pdb; pdb.set_trace()
        landmarks = np.reshape(landmarks, (-1,2))
        for p in landmarks:
            p = p*256.0
            p = p.astype(int)

            img = cv2.circle(img, tuple(p), 1, (255, 0, 0), 1)
        
        cv2.imwrite("../save/image_{}.png".format(count), img)

        count += 1

        if count==20:
            break