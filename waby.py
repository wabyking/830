# -*- coding: utf-8 -*-
import pandas as pd
from datetime import datetime
filename ="data/netflix/ratings.csv"
filename ="data/movieslen100k/ratings.csv"
df=pd.read_csv(filename,names=["uid","itemid","rating","timestamp"], sep="\t")

if "movies" in filename:
     stamp2date = lambda stamp :datetime.fromtimestamp(stamp)
     df["timestamp"]=  df["timestamp"].apply(stamp2date).dt.strftime(date_format="%Y-%m-%d")

start=df.timestamp.min()
end=df.timestamp.max()
split= "1998-03-08"
start_data= datetime.strptime(start, "%Y-%m-%d")
end_data= datetime.strptime(end, "%Y-%m-%d")
days =end_data-start_data
density = len(df)/days.days /len(df.uid.unique())



if True:
    df=df[ (df.timestamp >"2005-09-00") & (df.timestamp < "2005-13-00") ]
    test =df[ (df.timestamp >"2005-12-00") & (df.timestamp < "2005-13-00") ]
    split_start="1999-12-00"
    train =df[ (df.timestamp > split_start) & (df.timestamp < "2005-12-00")]

if  "movies" in filename:
    split="1998-03-00"
    test =df[ (df.timestamp >=split) & (df.timestamp <end) ]
    
    train =df[ (df.timestamp >start) & (df.timestamp < split)]
get_density(df)
get_density(train)
get_density(test)


def get_density(df):
    print (df)
    start=df["timestamp"].min()
    end=df["timestamp"].max()
    print(start,end)
    start_data= datetime.strptime(start, "%Y-%m-%d")
    end_data= datetime.strptime(end, "%Y-%m-%d")
    days =end_data-start_data
    density = len(df)/days.days /len(df.uid.unique())
    return density