import os
import pandas as pd
import numpy as np  
import datetime


data="netflix_six_month"

netflix_month={"start":"2005-06-00",
                "split":"2005-12-00",
                "end"  :"2005-13-00"
        }
netflix_year={"start":"2004-06-00",
                "split":"2005-06-00",
                "end"  :"2005-07-00"
        }
netflix_full={"start":"1999-12-00",
                "split":"2005-12-00",
                "end"  :"2005-13-00"
        }
movieslen100k={"start":"1000-12-00",
                        "split":"1998-03-08",
                        "end"  :"3005-13-00"
                }
date_dict={"netflix_six_month":netflix_month,"netflix_year":netflix_year,"netflix_full":netflix_full,"movieslen100k":movieslen100k}




def split_data(data):
    splited_date = date_dict[data]
    if not data.startswith("netflix"):
        filename="data/"+data+"/ratings.csv"
    else:
        filename="data/netflix/ratings.csv"
    df=pd.read_csv(filename,names=["uid","itemid","rating","timestamp"], sep="\t")
#    df =df[ (df.timestamp > "2005-08-31") & (df.timestamp < "2005-13") ]

    if  not data.startswith("netflix"):
        stamp2date = lambda stamp :datetime.datetime.fromtimestamp(stamp)
        df["timestamp"]=  df["timestamp"].apply(stamp2date).dt.strftime(date_format="%Y-%m-%d") 
#    pd.to_datetime(df['c'],format='%Y-%m-%d %H:%M:%S')#

    test =df[ (df.timestamp > splited_date["split"]) & (df.timestamp < splited_date["end"]) ]
    train =df[ (df.timestamp > splited_date["start"]) & (df.timestamp < splited_date["split"])]
    train_user_count=train.groupby("uid").apply(lambda group: len(group[group.rating>4.99])).to_dict()
    test_user_count=test.groupby("uid").apply(lambda group: len(group[group.rating>4.99])).to_dict()
    #print(len(df[df.rating>3.99]))
    
    if False:
        index=np.random.random(len(df))<0.8
        train=df[index]
        test=df[~index]
    else:
    
    
        train_users = {user for user,cnt in train_user_count.items() if cnt>20}   
    
        test_users = {user for user,cnt in test_user_count.items() if cnt>40}  &  train_users 
        whole_users=(test_users & train_users)
        test1=test[test.uid.isin(whole_users)]
        train1=train[train.uid.isin(train_users)]
        
    
        whole=pd.concat([train1,test1])
        whole['u_original'] = whole['uid'].astype('category')
        whole['i_original'] = whole['itemid'].astype('category')
        whole['uid'] = whole['u_original'].cat.codes
        whole['itemid'] = whole['i_original'].cat.codes
        whole = whole.drop('u_original', 1)
        whole = whole.drop('i_original', 1)
       
        # print (len(users))
        # print (len(items))
        print (len(whole.uid.unique()))
        print (len(whole.itemid.unique()))
#        test1 =whole[ (whole.timestamp > "2005-11-31") & (whole.timestamp < "2005-13") ]
#        train1 =whole[ (whole.timestamp > "2005-08-31") & (whole.timestamp < "2005-12")]
#        train1.to_csv("netflix_dir/train.csv",index=False,header=None,sep="\t")
#        test1.to_csv("netflix_dir/test.csv",index=False,header=None,sep="\t") 
        path_dir="data/"+data
        if not os.path.exists(path_dir):
            os.makedirs(path_dir)
        whole.to_csv(path_dir+"/ratings_subset.csv",index=False,header=None,sep="\t") 

def processNetflix():
	root="training_set"

	with open("ratings.csv","w") as out:
		for i in os.listdir(root):
			if os.path.isfile(os.path.join(root,i)):
				with open(os.path.join(root,i)) as f:
					lines=f.readlines()
					itemid= (lines[0].strip()[:-1])				
					print (itemid)
					for line in lines[1:]:
						line=line.strip()
						tokens=line.split(",")
						tokens.append(itemid)
						out.write(",".join(tokens)+"\n")


	df=pd.read_csv("ratings.csv",names=["uid","rating","timestamp","itemid"])
	df[["uid","itemid","rating","timestamp"]].to_csv("ratings.csv",index=False,header=None,sep="\t")


if __name__=="__main__":
    split_data(data)