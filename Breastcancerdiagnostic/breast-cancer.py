# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 22:55:21 2022

@author: asus
"""
import pandas as pd
import numpy as np

data=pd.read_csv('breast-cancer.csv')
data
data.diagnosis.replace(['M','B'],[0,1],inplace=True)
x=data.iloc[:,2:].values
y=data.iloc[:,1:2].values


#kayip,unique veriler
import re 
kayip_veriler=[]
sayisal_olmayan_veriler=[]

for oznitelik in data:
    essizdeger=data[oznitelik].unique()
    print("'{}' ozniteliğine sahip unique deger {}".format(oznitelik,essizdeger.size))
    
    if(True in pd.isnull(essizdeger)):
        s="{} ozniteliğe ait kayip veriler {}".format(oznitelik,pd.isnull(data[oznitelik]).sum())
        kayip_veriler.append(s)
    for i in range(1,np.prod(essizdeger.shape)):
       if(re.match('nan',str(essizdeger[i]))): 
           break
       if not(re.search('(^/d+\d*$)|(^\d*\.?\d+$)',str(essizdeger[i]))):
           sayisal_olmayan_veriler.append(oznitelik)
           break 
       
print("Kayıp veriye sahip oznitelikler:\n{}\n\n".format(kayip_veriler))
print("Sayısal olmayan veriye sahip oznitelikler:\n{}".format(sayisal_olmayan_veriler))


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.15,random_state=0)

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()

x_train=scaler.fit_transform(x_train)
x_test=scaler.fit_transform(x_test)
y_train=scaler.fit_transform(y_train.reshape(-1,1))
y_test=scaler.fit_transform(y_test.reshape(-1,1))

from sklearn.neighbors import KNeighborsClassifier 
classifier=KNeighborsClassifier(n_neighbors=3,metric='minkowski',p=2)
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)
y_pred

from sklearn.metrics import confusion_matrix,classification_report
hm=confusion_matrix(y_test, y_pred)
rapor=classification_report(y_test, y_pred)

from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt

ypo,dpo,esikDeger=roc_curve(y_test,y_pred)
aucDegeri=auc(ypo,dpo)
plt.figure()
plt.plot(ypo,dpo,label='AUC %0.2f'%aucDegeri)
plt.plot([0,1],[0,1],'k--')
plt.legend(loc="best")
plt.show()



