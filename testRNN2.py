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
    print(results)
    return list(np.mean(np.array(results),0))

#users_set=[ k for k,v in helper.test_user_count.items() if v > 100]

def testModel(sess,model):
    
    users_set=[ k for k,v in helper.test_user_count.items() if v > 100]
    print (evaluateMultiProcess(sess,model,users_set=users_set))
    

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
    init2=tf.global_variables_initializer()
    saver2 = tf.train.Saver(max_to_keep=50)


for sess_,model,init,saver in zip([sess1,sess2],[gen,dis],[init1,init2],[saver1,saver2]):
    sess_.run(init)
    

sess = sess2
model = dis    
best_p5 = 0
print ("training " +("Dis" if isinstance(model ,Dis) else "Gen") +" model")

for epoch in range(10):

    rnn_losses_g, mf_losses_g, joint_losses_g = [],[],[]

    for i,(u_seqs,i_seqs,rating,uid,itemid) in enumerate(helper.getBatch_with_multi_pickle(dns=FLAGS.dns,sess=sess,model=None,fresh=False)):
   
        _,loss_mf_g,loss_rnn_g,joint_loss_g ,mf,rnn= model.pretrain_step(sess, rating, uid, itemid, u_seqs, i_seqs)            

        rnn_losses_g.append(loss_rnn_g)
        mf_losses_g.append(loss_mf_g)
        joint_losses_g.append(joint_loss_g)

    print(" rnn loss : %.5f mf loss : %.5f  : joint loss %.5f" %
          (np.mean(np.array(rnn_losses_g)),np.mean(np.array(mf_losses_g)),np.mean(np.array(joint_losses_g))) )
    
    scores = (helper.evaluateMultiProcess(sess, model))
    print(scores)
   
    testModel(sess,model)
    if FLAGS.model_type == "mf":
        curentt_p5_score = scores[1]
    else:
        curentt_p5_score = scores[1][1]
    if curentt_p5_score > best_p5:        	
        best_p5 = curentt_p5_score
        print("best p5 score %5.f"% best_p5)
        checkpoint_dir="model/"+ helper.conf.dataset +("/Dis" if isinstance(model ,Dis) else "/Gen") +"/"
        helper.create_dirs(checkpoint_dir)        
        saver.save(sess, checkpoint_dir + '%s-%d-%.5f-%.5f.ckpt'% (FLAGS.model_type,FLAGS.re_rank_list_length,scores[0][1], scores[1][1]))

#            helper.create_dirs("model/mf")
#            mf_model = 'model/mf/%s-%d-%.5f.pkl'% (FLAGS.model_type,FLAGS.re_rank_list_length,best_p5)
#            dis.saveMFModel(sess,mf_model)
#            print(best_p5)

def  pairtrain():
    print (helper.evaluateMultiProcess(sess2, dis))
    joint_losses=[]

    for epoch in range(500):
        for i,((user,u_seqs,item,i_seqs,item_neg,i_seqs_neg)) in enumerate(helper.getBatch(dns=FLAGS.dns,sess=sess2,model=dis,fresh=False)):

            _,joint_loss = dis.pretrain_step_pair(sess2, user,u_seqs,item,i_seqs,item_neg,i_seqs_neg)     

            joint_losses.append(joint_loss) 
        print("mean loss = %.5f"% np.mean(joint_loss))
        scores = (helper.evaluateMultiProcess(sess2, dis))
            # print(helper.evaluateRMSE(sess,model))
        print(scores)

        
def analysisData():
	datas=[]
	for i,(uids,itemids,ratings) in enumerate(helper.getBatch4MF()):
		for uid,itemid,rating in zip(uids,itemids,ratings):
			line="%d\t%d\t%d" %(uid,itemid,rating)
			datas.append(line)
	with open("a.txt","w") as f:
		f.write("\n".join(datas))
	datas=[]

	for i,(_,_,ratings,uids,itemids) in enumerate(helper.prepare()):
		for uid,itemid,rating in zip(uids,itemids,ratings):
			line="%d\t%d\t%d" %(uid,itemid,rating)
			datas.append(line)

	with open("b.txt","w") as f:
		f.write("\n".join(datas))
             

