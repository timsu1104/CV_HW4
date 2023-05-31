# camera-ready

import numpy as np
import cv2
import os, glob
from time import time
from cityscapesscripts.helpers.csHelpers import ensurePath
import multiprocessing as mp

def accum(label_img_path):
    count = np.zeros(19)
    label_img = cv2.imread(label_img_path, -1)
    for trainId in range(19):
        count[trainId] += np.sum(label_img == trainId)
    return count

if __name__ == '__main__':
    start = time()
    cityscapes_data_path = "../data"
    ensurePath(cityscapes_data_path)
    
    mp.set_start_method("forkserver", True)

    print ("computing class weights")
    train_label_img_paths = sorted(glob.glob(os.path.join(cityscapes_data_path, "gtFine/train/*/*_gtFine_labelTrainIds.png")))

    with mp.Pool(mp.cpu_count()) as p:
        result = p.map(accum, train_label_img_paths)
    
    trainId_count = sum(result)

    # the ENet paper:
    class_weights = 1/np.log(1.02 + trainId_count / trainId_count.sum())

    print (class_weights)
    np.save(os.path.join(cityscapes_data_path, "class_weights.npy"), np.array(class_weights))
    print(f"Elapsed {time() - start} seconds. ")