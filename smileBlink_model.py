#from service_streamer import ManagedModel
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)

import tensorflow as tf
import argparse
import numpy as np
import sys
#sys.path.append('..')
from solu_base import Solu
from blink_net import BlinkLRCN
from solver import Solver
import cv2
from py_utils import x_utils as ulib
import time
import traceback
tf.compat.v1.disable_eager_execution()


class BlinkSmile():
    def __init__(self):
        self.net = BlinkLRCN(is_train=False)
    
        self.net.build()
        sess = tf.compat.v1.Session()
        # Init solver
        self.solver = Solver(sess=sess,
                        net=self.net,
                        mode='lrcn')
        self.solver.init()
        print("model build success")

    def predict(self,input_vid_path):

        input_vid_path = input_vid_path[0]
        print("Here the input is ",input_vid_path)
        solution = Solu(input_vid_path)
        
        stride = 10
        batch_size = np.arange(0, solution.frame_num, stride)
        
        count =0
        for i in batch_size:
            try:
                eye1_list, eye2_list = [], []
                eye1_index = []
                eye2_index = []

                for j in range(i, np.minimum(i + stride, solution.frame_num)):
                    eye1, eye2 = solution.get_eye_by_fid(j)
                    
                    if eye1 is not None:
                        eye1_index.append(j)
                        ank1 = cv2.resize(eye1, (self.net.img_size[0], self.net.img_size[1]))
                        eye1_list.append(ank1)
                    else:
                        continue    

                    if eye2 is not None:
                        eye2_index.append(j)
                        ank = cv2.resize(eye2, (self.net.img_size[0], self.net.img_size[1]))
                        eye2_list.append(ank)
                    else:
                        continue    
                eye2_list = [np.array(i) for i in eye2_list]
                eye1_list = [np.array(i) for i in eye1_list]
                
                try:
                    eye1_full = ulib.pad_to_max_len(eye1_list, self.net.max_time,pad=np.zeros(eye1_list[0].shape, dtype=np.int32))
                    eye2_full = ulib.pad_to_max_len(eye2_list, self.net.max_time,pad=np.zeros(eye2_list[0].shape, dtype=np.int32))
                    eye1_probs, = self.solver.test([eye1_full], [len(eye1_list)])
                    #print(eye1_probs)
                    eye2_probs, = self.solver.test([eye2_full], [len(eye2_list)])
                except:
                    continue
                for j in range(i, np.minimum(i + stride, solution.frame_num)):
                    try:
                        if j in eye1_index:
                            eye1_prob = eye1_probs[0][eye1_index.index(j), 1]
                        else:
                            eye1_prob = 0.5

                        if j in eye2_index:
                            eye2_prob = eye2_probs[0][eye2_index.index(j), 1]
                        else:
                            eye2_prob = 0.5
                        #print(eye1_prob,eye2_prob)

                        if eye1_prob>0.5 and eye2_prob>0.5:
                            count+=1
                    except Exception as e:
                        traceback.print_exc()
                        #print("in for loop",e)
                        pass
            except Exception as e:
                        import traceback
                        traceback.print_exc() 
                        print(e)
                        pass

        return [count]
 