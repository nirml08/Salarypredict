import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import mse as ms
import update as ud

df1=pd.read_csv("C://Users//root//Desktop//trainset.csv")
df2=pd.read_csv("C://Users//root//Desktop//testset.csv")

#creating train array
X=df1.iloc[:,0]
Y=np.array(df1.iloc[:,-1])
#mean normalization
Y=(Y-np.average(Y))/(np.max(Y)-np.min(Y))
#X=np.array(X)
X_train=np.array(X)
Y_train=np.array(Y)

#creating test array
X=df2.iloc[:,0]
Y=np.array(df2.iloc[:,-1])
#mean normalization
Y=(Y-np.average(Y))/(np.max(Y)-np.min(Y))
#X=np.array(X)
X_test=np.array(X)
Y_test=np.array(Y)
#creating random index in range
#rand_index=random.sample(range(0,30),22)

#plt.scatter(X_train,Y_train)
#y=mx+c
w=0.0354#random.random()#-0.0254
b=-0.0629#random.random()#0.386
alpha=0.00001

pred_y=[]
cost_y=[]
w_lst=np.array([])
b_lst=np.array([])
print(b_lst)
#loop for trainning
#loop for trainning
for itr in range(0,13000):
    #resettting the array for storing new predictions
    pred_y=[]
    for i in range(0,len(X_train)):
        y=(w*X_train[i])+b
        pred_y=np.append(pred_y,y)
    #calculating mse
    
    cost_y=np.append(cost_y,ms.cost(pred_y,Y_train))
    #updating parameters
    tempo1=w-(alpha*(ud.rule1(pred_y,Y_train,X_train)))
    w_lst=np.append(w_lst,tempo1)
    tempo2=b-(alpha*(ud.rule2(pred_y,Y_train)))
    b_lst=np.append(b_lst,tempo2)
    w=tempo1
    b=tempo2
    if (itr % 1000)==0:
        if(cost_y[itr]<cost_y[itr-1]):
            print("\ndecreasing:",cost_y[itr])
        else:
            print("\nincreasing:",cost_y[itr])
            

pred_test_y=np.array([])
for i in range(0,len(X_test)):
    h=(w*X_test[i])+b
    pred_test_y=np.append(pred_test_y,h)
        
#rmse of train set    
rmse=ms.rmse(pred_y,Y_train)
print("RMSE of Train set",rmse)

#rmse of tst set
rmse=ms.rmse(pred_test_y,Y_test)
print("RMSE of Test set",rmse)




#plt.show()
 

#optimize teh cost function




#(y"-Y)2/2

    



