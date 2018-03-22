#!/home/zhujun/python27/bin/python

import numpy as np
import random
from skimage import transform as tf
from skimage import io
from os import walk
from matplotlib import pyplot as plt
import scipy.io as scio
import multiprocessing as mp
import traceback
amount=15000
basic = '/data2/swfccnn/frcn/data/output_make3d/'
# path ='/data2/swfccnn/frcn/data/make3d/Test134/'
# depth_path = '/data2/swfccnn/frcn/data/make3d/Test134Depth/Gridlaserdata/'
# output_path = basic + 'aug_output/test_output_path/'
# output_depth_path = basic + 'aug_output/test_output_depth_path/'
path ='/data2/swfccnn/frcn/data/make3d/Train400Img/'
depth_path = '/data2/swfccnn/frcn/data/make3d/Train400Depth/'
output_path = basic + 'aug_output/train_output_path/'
output_depth_path = basic + 'aug_output/train_output_depth_path/'
cnt = 0

class dataTrans():

    def __init__(self, input, target):
        # self.net_size = (172, 230)
        self.net_size = (360, 480)
        self.input = input
        self.target = target
        self.output = tf.resize(self.input, self.net_size)
        self.target_out = tf.resize(self.target, self.net_size)

        self.scale_to_percent = (1, 1.5)
        self.rotation_deg = (-5, 5)
        self.color_rate = (0.8, 1.2)

    def crop(self, image1, image2):
        assert(image1.shape[:-1] == image2.shape)
        width, height = image2.shape
        x_start_limit = width - self.net_size[0]
        y_start_limit = height - self.net_size[1]
        x_start = random.randint(0, x_start_limit-1)
        y_start = random.randint(0, y_start_limit-1)

        return (image1[x_start:x_start+self.net_size[0]-1, y_start:y_start+self.net_size[1]-1],
                image2[x_start:x_start+self.net_size[0]-1, y_start:y_start+self.net_size[1]-1].astype(int))

    def flip(self):
         prob = random.uniform(0, 1)
         if prob > 0.5:
             self.output = tf.rotate(self.output, 180)
             self.target_out = tf.rotate(self.target_out, 180)

    def rotation(self):
           deg = random.uniform(self.rotation_deg[0], self.rotation_deg[1])
           self.output =  tf.rotate(self.output, deg)
           self.target_out = tf.rotate(self.target_out, deg)

    def scale(self):
         scale = random.uniform(self.scale_to_percent[0], self.scale_to_percent[1])
         self.output = tf.rescale(self.output, scale)
         self.target_out = tf.rescale(self.target_out, scale)
         self.target_out = (self.target_out/scale).astype(int)

         self.output, self.target_out =self.crop(self.output, self.target_out)

    def colorTrans(self):
         color_rate = random.uniform(self.color_rate[0], self.color_rate[1])
         self.output = (color_rate*self.output).astype(int)
         self.target_out = self.target_out


    def translation(self):
        zoomin_img = tf.resize(self.input, [445, 560])
        zoomin_dep = tf.resize(self.target, [445, 560])
        self.output, self.target_out =self.crop(zoomin_img, zoomin_dep)

    def getNewData(self, name, depth_name):
        io.imsave(name, self.output)
        io.imsave(depth_name, self.target_out)
#        print self.target_out

    def epcho(self, name, depth_name):
         ops = [self.flip(), self.rotation(), self.scale(), self.colorTrans(), self.translation()]
         for i in range(0, len(ops)):
             lucky_number = random.uniform(0, 1)
             if lucky_number > 0.5:
                 ops[i]

         self.getNewData(name, depth_name)

    def testTrans(self):
        self.epcho("", "")

        io.imshow(self.input)
        plt.show()
        io.imshow(self.output)
        plt.show()
        io.imshow(self.target)
        plt.show()
        io.imshow(self.target_out)
        plt.show()



def augData(path, depth_path, amount, output_path, output_depth_path):
	cnt = 0
	while cnt < amount:
         for dirpath, dirnames, filenames in walk(path):
			##to do: prefix img-10.21op2-p-139t000
               for filename in filenames:
                    try:
                  	 prefix = '-'.join('.'.join(filename.split('.')[0:-1]).split('-')[1:])
                  	 name = path+'img-'+prefix+'.jpg'
                  	 depth_name = depth_path+'depth_sph_corr-'+prefix+'.mat'
                  	 #print name
                  	 #print depth_name
                 	 name_matrix = io.imread(name)
                 	 depth_name_matrix = scio.loadmat(depth_name)
                 	 depth_name_matrix = depth_name_matrix['Position3DGrid'][:,:,3]
                  	 depth_name_matrix[depth_name_matrix > 70] = 70
                 	 trans = dataTrans(name_matrix, depth_name_matrix)
                 	 out_prefix = "img-%s.png" % str(cnt)
                 	 out_depth_prefix = "depth_sph_corr-%s.png" % str(cnt)
                 	 try:
                  	    trans.epcho(output_path+out_prefix, output_depth_path+out_depth_prefix)
                    	    cnt += 1
                    	    print cnt
                         except:
                  	    continue
	            except:
			 continue

def single_agu(filename):
    try:
        global cnt
        prefix = '-'.join('.'.join(filename.split('.')[0:-1]).split('-')[1:])
        name = path+'img-'+prefix+'.jpg'
        depth_name = depth_path+'depth_sph_corr-'+prefix+'.mat'
#print name
#print depth_name
        name_matrix = io.imread(name)
        depth_name_matrix = scio.loadmat(depth_name)
        depth_name_matrix = depth_name_matrix['Position3DGrid'][:,:,3]
        depth_name_matrix[depth_name_matrix > 70] = 70
        trans = dataTrans(name_matrix, depth_name_matrix)
        out_prefix = "img-%s.png" % str(cnt)
        out_depth_prefix = "depth_sph_corr-%s.png" % str(cnt)
        try:
            trans.epcho(output_path+out_prefix, output_depth_path+out_depth_prefix)
            cnt += 1
            print cnt
        # except:
        except Exception as e:
            print e
            return
    except Exception as e:
        print e

        return
def mf_wrap(args):
    single_agu(*args)



def multiple_augData(path):
    p = mp.Pool(4)
    # cnt = 0
    while cnt < amount:
         for dirpath, dirnames, filenames in walk(path):
             # f1 = [(f,c) for f in filenames for c in range(cnt, cnt+len(filenames))]
             p.map(single_agu, filenames)
if __name__ == '__main__':
    multiple_augData(path)

#    basic_path = 'C:/Users/jszhu/Documents/didi/depth/FCRN-DepthPrediction-master/matlab/Make3D/data/'
#    input_path = basic_path + 'Test3/img-10.21op2-p-139t000.jpg'
#    target_path = basic_path + 'Gridlaserdata3/depth_sph_corr-10.21op2-p-139t000.mat'
#
#    input = io.imread(input_path)
#    target = scio.loadmat(target_path)
#    target = target['Position3DGrid'][:,:,3]
#    trans = dataTrans(input, target)
#    trans.testTrans()


        # basic = '/data2/swfccnn/frcn/data/output_make3d/'
        # path ='/data2/swfccnn/frcn/data/make3d/Test134/'
        # depth_path = '/data2/swfccnn/frcn/data/make3d/Test134Depth/Gridlaserdata/'
        # output_path = basic + 'aug_output/test_output_path/'
        # output_depth_path = basic + 'aug_output/test_output_depth_path/'
        # augData(path, depth_path, 15000, output_path, output_depth_path)










