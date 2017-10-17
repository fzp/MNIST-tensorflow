import tensorflow.examples.tutorials.mnist.input_data as input_data
import numpy as np
import matplotlib.pyplot as plt
import pylab

mnist = input_data.read_data_sets("input_data/", one_hot=True)    

batch_xs, batch_ys = mnist.train.next_batch(100)        
for one_pic_vic in batch_xs:
    one_pic_arr = np.reshape(one_pic_vic,(28,28))                           
    plt.imshow(one_pic_arr)
    pylab.show()