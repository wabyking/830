# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import os
import numpy as np 
import pickle
import tensorflow as tf
from Discrimiator import Dis
from Generator import Gen
from tqdm import tqdm
from config import Singleton
from dataHelper import DataHelper
import itertools
from multiprocessing import Pool
from multiprocessing import cpu_count

FLAGS=Singleton().get_andy_flag()
helper=DataHelper(FLAGS)
helper.test_user_count=helper.test.groupby("uid").apply(lambda group: len(group[group.rating>0.5])).to_dict()

os.environ['CUDA_VISIBLE_DEVICES'] = ''


if os.path.exists(FLAGS.pretrained_model) and FLAGS.pretrained:
	print("Fineutune the discrimiator with pretrained MF named " + FLAGS.pretrained_model)
	paras= pickle.load(open(FLAGS.pretrained_model,"rb"))
else:
	print("Fail to load pretrained MF model ")
	paras=None



g1 = tf.Graph()
g2 = tf.Graph()
sess1 = tf.InteractiveSession(graph=g1)   
sess2 = tf.InteractiveSession(graph=g2)


with g1.as_default():
    gen = Gen(itm_cnt = helper.i_cnt, 
             usr_cnt = helper.u_cnt, 
             dim_hidden = FLAGS.rnn_embedding_dim, 
             n_time_step = FLAGS.item_windows_size, 
             learning_rate =  FLAGS.learning_rate, 
             grad_clip = 0.2,
             emb_dim = FLAGS.mf_embedding_dim,
             lamda = FLAGS.lamda,
             initdelta = 0.05,
             MF_paras=paras,
             model_type=FLAGS.model_type,
             update_rule = 'sgd',
             use_sparse_tensor=FLAGS.sparse_tensor
             )
    gen.build_pretrain()
    init1=tf.global_variables_initializer()
    saver1 = tf.train.Saver(max_to_keep=50)
    #    saver1.restore(sess1,checkpoint_filepath)
    checkpoint_filepath= "model/netflix_three_month/Dis/0831/joint-25-0.19800-0.22200.ckpt"
        #joint-25-0.16839-0.18968.ckpt"
        # checkpoint_filepath= "model/Dis/joint-25-0.26067-0.29000.ckpt"
    checkpoint_filepath= "model/netflix_3_month/GAN_Gen/joint-25-0.21800-0.23600.ckpt"
    saver1.restore(sess1,checkpoint_filepath)
    
with g2.as_default():
    dis = Dis(itm_cnt = helper.i_cnt, 
             usr_cnt = helper.u_cnt, 
             dim_hidden = FLAGS.rnn_embedding_dim, 
             n_time_step = FLAGS.item_windows_size, 
             learning_rate = 0.005, #0.0005
             grad_clip = 0.2,
             emb_dim = FLAGS.mf_embedding_dim,
             lamda = FLAGS.lamda,
             initdelta = 0.05,
             MF_paras=paras,
             model_type='joint',
             update_rule = 'sgd',
             use_sparse_tensor=FLAGS.sparse_tensor,
             pairwise=helper.conf.pairwise
             )
    dis.build_pretrain()
    saver2 = tf.train.Saver(max_to_keep=50)
    init2=tf.global_variables_initializer()
    checkpoint_filepath= "model/netflix_3_month/Dis/joint-25-0.22200-0.21400.ckpt"
    saver2.restore(sess2,checkpoint_filepath)

sess,model=sess1,gen
def evaluateMultiProcess(sess,model,mp=False,users_set=None):
    if users_set is None:
        users_set=helper.test_users
#        users_set=helper.test[helper.test.rating>4.99]["uid"].unique()
    print("evaluate %d users" %len(users_set))
    # users_set=helper.users
    results=None
    if mp:
        # try:
        pool=Pool(cpu_count())
        results= pool.map(helper.getScore,zip(list(users_set), itertools.repeat(sess),itertools.repeat(model) ))
        # except:
            # pool.close()
    else:
        results= [ i for i in map(helper.getScore,zip(users_set, itertools.repeat(sess),itertools.repeat(model) ))]
    return list(np.mean(np.array(results),0))

#users_set=[ k for k,v in helper.test_user_count.items() if v > 100]

def testModel(sess,model):
    
    users_set=[ k for k,v in helper.test_user_count.items() if v > 100]
    print (evaluateMultiProcess(sess,model,users_set=users_set))
  

scores=[]
x=[i for i in range(10,100,10)] + [i for i in range(100,17000,100)]
x= [25,16]
for rerank_size in x:
    helper.conf.re_rank_list_length=rerank_size
    score= testModel(sess,model)
    scores.append(score)
    print(scores)

# build y
x = np.linspace(0, 2 * np.pi, 10)
y1, y2 = np.sin(x), np.cos(x)
 
plt.plot(x, y1, marker='o', mec='r', mfc='w')
plt.plot(x, y2, marker='*', ms=10)
plt.show()