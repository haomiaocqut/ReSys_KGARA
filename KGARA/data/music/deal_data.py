import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
df = pd.read_table('ratings_final.txt',header=None)
#划分train，test，保证用户两边都有
train,test,_,_ = train_test_split(df,df[0],test_size=0.3)
train.sort_values(0,inplace=True)
test.sort_values(0,inplace=True)
#test用户
uid = set(test[0].unique())
#总样本
sid = set(df[1].unique())
#结果文档
result = pd.DataFrame(columns=range(101))
#设置随机数种子,每一个种子下的结果是一样的
np.random.seed(0)
test_result = pd.DataFrame(columns=range(3))
#循环每一个用户
for u in uid:
     #test该用户的正样本
     P = set(test.loc[(test[0]==u)&(test[2]==1),1])
     if len(P)>0:
          #该用户的正样本
          P_ = set(df.loc[(df[0]==u)&(df[2]==1),1])
          #从test正样本里抽
          p = random.sample(P,1)
          #从总样本里减去该用户的正样本再抽100作为负样本
          n = random.sample(sid-P_,100)
          #保存结果
          if len(p) > 0:
               result.loc[u] = ['('+str(u)+','+str(p[0])+')']+n
               test_result.loc[u] = [u,p[0],1]
               
     
#保存数据
f = open("test_negative.txt",'w')
for u in result.index:
     f.write(' '.join(result.loc[u].astype(str).values))
     f.write("\n")
f.close()

train.to_csv('train.txt',index=0,header=None,sep=' ')
test.to_csv('test11.txt',index=0,header=None,sep=' ')
test_result.to_csv('test.txt',index=0,header=None,sep=' ')










