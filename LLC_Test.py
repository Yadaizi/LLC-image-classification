# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 11:31:55 2016

@author: YadaiziMeng
"""
import numpy as np
import scipy.io as sio
from sklearn import svm
import os.path
import os
import time

start = time.time()

def LLC_coding_appr(B, X, knn):
    
    knn=5
    beta = 1e-4
    nframe=X.shape[0]   #get value N  1392
    nbase=B.shape[0]    #get value M  1024
    
    
    #find k nearest neighbors
    XX=np.sum(np.square(X),axis=1)    #1392*1
    BB=np.sum(np.square(B),axis=1)    #1024*1
    D=np.tile(XX,(nbase,1)).T-np.dot(2*X,(B.T))+np.tile(BB,(nframe,1))
    IDX=np.argsort(D, axis=1)[:,0:5]
    II=np.eye(knn)
    Coeff=np.zeros((nframe,nbase))
    
    for i in range(0,nframe):
        idx=IDX[i,:]-1
        z=B[idx,:]-np.tile(X[i,:],(knn,1))
        C=np.dot(z,z.T)
        C=C+beta*np.trace(C)*II
        w=np.linalg.solve(C, np.ones((knn,1)))
        w= w / np.sum(w)
        Coeff[i,idx]=w.T
        
    return Coeff


def LLC_pooling(feaSet, B, pyramid, knn):
    dSize=B.shape[1]
    nSmp=feaArr.shape[1]
    idxBin =  np.zeros((nSmp, 1))
    llc_codes = LLC_coding_appr(B.T, feaArr.T, knn)
    llc_codes = llc_codes.T    
    pLevels = len(pyramid)
    pBins = np.power(pyramid,2)
    tBins = sum(pBins)
    beta = np.zeros((dSize, tBins))
    bId = 0   
    for iter1 in range(0,pLevels):
        nBins=pBins[iter1]
        wUnit = img_width / pyramid[iter1]
        hUnit = img_height / pyramid[iter1]
        xBin = np.ceil(x / wUnit)
        yBin = np.ceil(y / hUnit)
        idxBin = (yBin - 1)*(pyramid[iter1]) + xBin
        for iter2 in range(0,nBins):
            bId=bId+1
            sidxBin=np.where(idxBin == iter2+1)[0]
           
            
            print sidxBin        
            if len(sidxBin) == 0:
                continue
            beta[:, bId-1] = np.amax(llc_codes[:, sidxBin-1],axis=1)
            
        
    if bId != tBins:
        raise Exception('Index number error!')
    beta = np.reshape(beta,beta.size,order='F').T
    beta = beta/np.power((sum(np.power(beta,2))),0.5)
    print sum(beta)
    return beta
    

def retr_database_dir(rt_data_dir):
     print("dir to the subfolders...")
     database = {}
     database['path'] = []
     database['cname'] = []
     database['nclass'] = 0
     database['imnum'] = 0
     database['label'] = []
     
     for path, subdirs, files in os.walk(rt_data_dir):
         for name in files:
             if name[0] != '.':
                 database['path'].append(os.path.join(path, name))
                 
         if len(subdirs) != 0:
             database['cname'] = subdirs
     database['nclass'] = len(database['cname'])
     database['imnum'] = len(database['path'])
     
     label_num = 1
     for names in database['cname']:
         folder_path = rt_data_dir + '/' + names
         label_len = len([name for name in os.listdir(folder_path) if os.path.isfile(folder_path + '/' + name)])
         for k in range(0, label_len):
             database['label'].append(label_num)
         label_num += 1
     return database


#parameter setting
pyramid = np.array([1, 2, 4])
knn=5
nRounds=10
tr_num  = 30


#set path
img_dir = 'image/Caltech101'       # directory for the image database                             
data_dir = 'data/Caltech101'       # directory for saving SIFT descriptors
fea_dir = 'features/Caltech101'    # directory for saving final image features

#retrieve the directory of the database and load the codebook
database = retr_database_dir(data_dir)
Bpath = 'dictionary/Caltech101_SIFT_Kmeans_1024.mat'
B=sio.loadmat(Bpath)['B']
nCodebook=B.shape[1]

#extract image features
dFea = np.sum(nCodebook*np.square(pyramid))
nFea = len(database['path'])
fdatabase=retr_database_dir(fea_dir)
fdatabase['path'] = []
fdatabase['label'] = []
for iter1 in range(0,nFea):
    print 'iteration:', iter1
    if not((iter1+1)%5):
        print(".")
    if not ((iter1+1)%100):
        print("%d images processed." % (iter1+1))
    fpath = database['path'][iter1]
    flabel = database['label'][iter1]
    
    
    
    (rtpath, fname)  = os.path.split(fpath)
    feaSet=sio.loadmat(fpath)['feaSet']
    feaArr=feaSet['feaArr'][0][0]
    x=feaSet['x'][0][0]
    y=feaSet['y'][0][0]
    img_width=feaSet['width'][0][0][0][0]
    img_height=feaSet['height'][0][0][0][0]
    
    feaPath = os.path.join(fea_dir, str(flabel),fname)
    fea = LLC_pooling(feaSet, B, pyramid, knn)
    label = database['label'][iter1]
    
    if os.path.isdir(os.path.join(fea_dir, str(flabel))) ==0:
        os.mkdir(os.path.join(fea_dir, str(flabel)))
    sio.savemat(feaPath,{'fea':fea,'label':label})
    
    
    
    
    fdatabase['label'].append(flabel)
    
    fdatabase['path'].append(feaPath)
#    
#    
    
#evaluate the performance of the image feature using SVM
print '\n Testing...\n'
clabel=np.unique(fdatabase['label'])  # array of unique labels
nclass=len(clabel)         #number of unique labels
accuracy=np.zeros((nRounds,1))
for ii in range(0,nRounds):
    print 'Round: ', ii+1
    tr_idx = np.array([])  #train index
    ts_idx = np.array([])  #test index
    for jj in range(0,nclass):
        idx_label = np.where(fdatabase['label']==clabel[jj])[0]
        num=len(idx_label)  
        idx_rand = np.random.permutation(num)
        tr_idx=np.concatenate((tr_idx, idx_label[(idx_rand[0:tr_num])]),axis=0)
        ts_idx=np.concatenate((ts_idx, idx_label[(idx_rand[tr_num:num])]),axis=0)
    print 'Traning number: ', len(tr_idx)
    print 'Testing number: ', len(ts_idx)
    #load the training features
    tr_fea = np.zeros((len(tr_idx), dFea))
    tr_label=np.zeros((len(tr_idx), 1))
    for jj in range(0,len(tr_idx)):
        fpath=fdatabase['path'][int(tr_idx[jj])]
        
        tr_fea[jj,:]=sio.loadmat(fpath)['fea'].T.reshape(dFea,)
        tr_label[jj]=sio.loadmat(fpath)['label']
    
    lin_clf = svm.LinearSVC()
    lin_clf.fit(tr_fea, tr_label) 
    
    #load the testing features
    ts_num = len(ts_idx)
    ts_label = np.array([])
    ts_fea = np.zeros((len(ts_idx), dFea))
    ts_label = np.zeros((len(ts_idx), 1))
    for jj in range(0,len(ts_idx)):
        fpath=fdatabase['path'][int(ts_idx[jj])]
        ts_fea[jj, :]=sio.loadmat(fpath)['fea'].T.reshape(dFea,)
        ts_label[jj]=sio.loadmat(fpath)['label']
    C = lin_clf.predict(ts_fea)
    #normalize the classification accuracy by averaging over different
    acc = np.zeros((nclass, 1))
    
    for jj in range(0,nclass):
        c=clabel[jj]
        idx=np.where(ts_label==c)[0]
        curr_pred_label = C[idx]
        curr_gnd_label = ts_label[idx].reshape(len(idx),)
        print 'length of correct lables: ',len(np.where(curr_pred_label == curr_gnd_label)[0])
        print 'length of total idx: ',len(idx)
        acc[jj] = float(len(np.where(curr_pred_label == curr_gnd_label)[0]))/float(len(idx))
    
    accuracy[ii] = np.mean(acc)
    print 'Classification accuracy for round: ', ii, "{0:.2%}".format(float(accuracy[ii]))

Ravg = np.mean(accuracy)
Rstd = np.std(accuracy)

print'==============================================='
print'Average classification accuracy: ', "{0:.2%}".format(float(Ravg))
print'Standard deviation: ', "{0:.2%}".format(float(Rstd))
print'==============================================='



print 'running time: ', time.time()-start
    
