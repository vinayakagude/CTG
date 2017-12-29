#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 11:13:43 2017

@author: gudeharikishan
"""

import csv 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from peakdetect import peakdetect
from numpy import matrix
from scipy import stats

def feature_extraction(file):
    global cont
    global pk
    global lw
    global pa
    global dl
    global ds
    global ln
    global dp
    global var
    global skew
    global tend
    global cstarts
    global cends
    global mstarts
    global mends
    global ends
    global starts   
    global minh
    global maxh
    global meanh
    global medianh
    global modeh
    global peaks
    global widh
    
    data = []
    print(file)
    ifile  = open(file, "r")
    read = csv.reader(ifile)
    for row in read :
        data.append(row)
    
    fhr=[]
    uc=[]
    
    arraydata = data
    
    x = 0
    while(x < (len(arraydata)-2)):
        if arraydata[x][1] == 'HR2' and arraydata[x+1][1] == 'UA':
            x = x+2
            continue
        else:
            arraydata.pop(x)
    
    x = 1        
    while(x < (len(arraydata)-2)):
        if arraydata[x][1] == 'UA' and arraydata[x-1][1] == 'HR2':
            x = x+2
            continue
        else:
            arraydata.pop(x)        
        
    fhr=[]
    uc=[]
    
    for x in range(len(arraydata)):
        if arraydata[x][1] == 'HR2':
            fhr.append(arraydata[x])
        else:
            uc.append(arraydata[x])
    for x in range(len(fhr)):
        fhr[x].pop(0)
        fhr[x].pop(0)
        fhr[x].pop(0)
    
    for x in range(len(uc)):
        uc[x].pop(0)
        uc[x].pop(0) 
        uc[x].pop(0)
            
    
    def flatten(l):
        flatList = []
        for elem in l:
            # if an element of a list is a list
            # iterate over this list and add elements to flatList 
            if type(elem) == list:
                for e in elem:
                    flatList.append(e)
            else:
                flatList.append(elem)
        return flatList
    
    fhr = flatten(fhr)
    uc = flatten(uc)
    list(fhr)
    list(uc)
    fhr = list(map(int, fhr))
    uc = list(map(int, uc))
          
    fhr = np.array(fhr)
    ucz = np.array(uc)
            
    fhr = fhr[fhr > 100]
    
    
    fhr1 = np.array_split(fhr, 1)
    ucz = np.array_split(ucz, 1)
    
  
    N = 40
    fhr = pd.rolling_mean(fhr1[0], N)[N-1:]
    plt.plot(fhr)
    mn = np.mean(fhr)
    mdn = np.median(fhr)
    peaks = peakdetect(fhr, lookahead=300)
    peaks = np.array(peaks)
    maxs = np.array(peaks[0])
    mins = np.array(peaks[1])
 
    if len(maxs) == 0 or len(mins) == 0:
        output = []
        output.append('NA')
        output.append('NA')
        
    else:
        mdn = np.mean(fhr)
        mn = np.median(fhr)
        j=0
        count = 0
        
        while(abs(mdn-j) > 0.15):    
            peaks = peakdetect(fhr, lookahead=300)
            peaks = np.array(peaks)
            maxs = np.array(peaks[0])
            mins = np.array(peaks[1])
            maxs = matrix(maxs).transpose()[0].getA()[0]
            mins = matrix(mins).transpose()[0].getA()[0]
            #print(maxs) #Maximums
            #print(mins) #Minimums
            
            ends = []
            starts = []
            
            pk = []
            m15 = mdn+10
            for k in range(len(maxs)):
                if fhr[int(maxs[k])] > m15:
                    pk.append(maxs[k])
            
            # Removing values above 170        
            pk = list(filter(lambda x : (int(fhr[int(x)])) < 170 , pk))
            #print(pk)
            
            ec = 0
            sc = 0
            #Finding starting and ending points
            for x in range (len(pk)):
                t =int(len(fhr)-1)
                s = int(pk[x])
                n = sc
                p = ec
                for y in range (s,t):
                    if fhr[y]-fhr[y+1]<0:
                        ends.append(y)
                        ec = ec+1
                        break
                    if fhr[y] == mdn:
                        ends.append(y)
                        ec = ec+1
                        break
                for k in range (0,s-1):
                    if fhr[s-k]-fhr[s-k-1]<0:
                        starts.append(s-k)
                        sc = sc +1
                        break
                    if fhr[s-k] == mdn:
                        starts.append(s-k)
                        sc = sc+1
                        break
                if sc == n:
                    starts.append('NA')
                if ec == p:
                    ends.append('NA')             
                    
            # add statement: 
                
            ev = 0
            sv = 0        
            mends = []
            mstarts = []       
            lw = []
            n15 = mdn-9     
            for k in range(len(mins)):
                if fhr[int(mins[k])] < n15:
                    lw.append(mins[k])
                    
            for x in range (len(lw)):
                t =int(len(fhr)-1)
                s = int(lw[x])
                o = sv
                r = ev
                for y in range (s,t):
                    if fhr[y]-fhr[y+1]>0:
                        mends.append(y)
                        ev = ev+1
                        break
                    if fhr[y] == mdn:
                        mends.append(y)
                        ev = ev+1
                        break
                for k in range (0,s-1):
                    if fhr[s-k]-fhr[s-k-1]>0:
                        mstarts.append(s-k)
                        sv = sv+1
                        break
                    if fhr[s-k] == mdn:
                        mstarts.append(s-k)
                        sv = sv+1
                        break
                if sv == o:
                    mstarts.append('NA')
                if ev == r:
                    mends.append('NA')
             
            np.array(starts)
            np.array(mstarts)
            np.array(ends)
            np.array(mends)
            np.array(pk)
            np.array(lw)
            
            count = 0
            check = []
            #check duration
            for x in range(len(starts)):
                if starts[x] ==  'NA':
                    np.delete(starts, x)
                    np.delete(ends, x)
                    np.delete(pk, x)
                    break
            for x in range(len(ends)):       
                if ends[x] =='NA':
                    np.delete(starts, x)
                    np.delete(ends, x)
                    np.delete(pk, x)
                    break
                    
                check.append(starts[x] - ends[x])
                if ends[x] - starts[x] > 40: 
                    count = count+1
                else:
                    np.delete(starts, x)
                    np.delete(ends, x)
                    np.delete(pk, x)
                    
            for x in range(len(mstarts)):
                if mstarts[x] ==  'NA':
                    np.delete(mstarts, x)
                    np.delete(mends, x)
                    np.delete(lw, x)
                    break
            for x in range(len(mends)):      
                if mends[x] =='NA':
                    np.delete(mstarts, x)
                    np.delete(mends, x)
                    np.delete(lw, x)
                    break
                    
                check.append(mstarts[x] - mends[x])
                if mends[x] - mstarts[x] > 40: 
                    count = count+1
                else:
                    np.delete(mstarts, x)
                    np.delete(mends, x)
                    np.delete(lw, x)        
                    
            
                
            #check value for baseline:
            index = []
            for x in range(len(starts)):
                if starts[x] ==  'NA':
                    break
                if ends[x] =='NA':
                    break
                else:
                    for y in range(int(starts[x]),int(ends[x])):
                        index.append(y)
                        
            for x in range(len(mstarts)):
                if mstarts[x] ==  'NA':
                    break
                if mends[x] =='NA':
                    break
                else:
                    for y in range(mstarts[x],mends[x]):
                        index.append(y)        
                
            new_x = np.delete(fhr, index)
            j = mdn
            mdn = np.mean(new_x)
            count = count+1
            
        # add set intersection
        pa = []
        for k in range(len(pk)):
            if int(starts[k])-int(ends[k]) > 480:
                pa.append(pk[k])
                
        
        
        lw = np.array(lw)
        lw = lw.astype(np.int64)
        pk = np.array(pk)
        pk = pk.astype(np.int64)
          
        plt.plot(lw,fhr[lw],'bo')
        plt.plot(pk,fhr[pk],'ro')
    
       
        
    ucz = np.asarray(ucz)
    ucz = ucz[0]
    N = 50
    ucz = pd.rolling_mean(ucz, N)[N-1:]
    z1 = peakdetect(ucz, lookahead=300)
    z = np.array(z1[0])
    if len(z) == 0:
        cn = []
        cn.append('NA')
        cont = []
        print('dfsfs')
    else:
        cont = matrix(z).transpose()[0].getA()[0]
        mdn =np.median(ucz)
        #plt.plot(x, y, '.-')
        #plt.plot(x, ys)
        #plt.show()
        
        cends = []
        cstarts = []
        ec = 0
        sc = 0
        
        
        cn = []
        for k in range(len(cont)):
            if ucz[int(cont[k])] > 20:
                cn.append(cont[k])
            
        cont = np.delete(cont,cn)
        
        #Finding starting and ending points
        for x in range (len(cont)):
            t =int(len(ucz)-1)
            s = int(cont[x])
            n = sc
            p = ec
            for y in range (s,t):
                if ucz[y]-ucz[y+1]<0:
                    cends.append(y)
                    ec = ec+1
                    break
            for k in range (0,s-1):
                if ucz[s-k]-ucz[s-k-1]<0:
                    cstarts.append(s-k)
                    sc = sc +1
                    break
            if sc == n:
                cstarts.append('NA')
            if ec == p:
                cends.append('NA')
        de = []        
        for x in range(len(cont)):
            if cstarts[x] ==  'NA':
                de.append(x)
                
            if cends[x] =='NA':
                de.append(x)
                
        cont = np.delete(cont,de)
        cstarts = np.delete(cstarts,de)
        cends = np.delete(cends,de)
                 
        
        # Late Decelerations
        ln = []
        for x in range(len(cstarts)):
            for y in range(len(mstarts)):
                if int(mstarts[y]) > int(cstarts[x]) and int(mstarts[y]) < int(cends[x]):
                    if (int(mstarts[y]) - int(cstarts[x])) > 20:
                        ln.append(lw[y])
    dl = []
    ds = []
    dp = []
    for x in range(len(lw)):
        if (mstarts[x] - mends[x]) < 480:
            dl.append(lw[x])
    for x in range(len(lw)):
        if (mstarts[x] - mends[x]) > 480 and (mstarts[x] - mends[x]) < 1200:
            dp.append(lw[x])
    for x in range(len(lw)):
        if (mstarts[x] - mends[x]) > 1200:
            ds.append(lw[x])
            
    real = fhr
    acde = []
    # Variability
    for x in range(len(pk)):
        for y in range(len(starts)):
            f = starts[y]
            for f in range(starts[y],ends[y]):
                acde.append(f)
                f = f+1
            
    for x in range(len(lw)):
        for y in range(len(mstarts)):
            f = mstarts[y]
            for f in range(mstarts[y],mends[y]):
                acde.append(f)
                f = f + 1
    
    real = np.delete(real,acde)
    if len(real)==0:
        var = 0
    else:        
        maxr = max(real)
        minr = min(real)
        var = maxr - minr
    
    fhr = fhr.astype(int)
    counts = np.bincount(fhr)
    peaks = []
    for x in range(len(counts)):
        if counts[x] > 0.05*len(fhr):
            peaks.append(x)
    minh = np.amin(fhr)
    maxh = np.amax(fhr)
    widh = maxh - minh
    medianh = np.median(fhr)
    meanh = np.mean(fhr)
    modeh =stats.mode(fhr,axis=0).mode[0]   
    output = np.array([len(pk),len(lw),len(pa),len(ln),len(cont),len(dl),len(dp),len(ds),float(var),widh,minh,maxh,len(peaks),modeh,meanh,medianh])
    print(output)
    return output

    '''
    ind = []  
    for x in range(len(peaks)):
        ind.append(peaks[x][0])    
    print(starts)
    print(ends)    
    plt.plot(ind,fhr1[0][ind],'bo')
'''   