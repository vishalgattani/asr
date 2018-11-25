#!/usr/bin/env python







'''



    NAME    : LDC TIMIT Dataset



    URL     : https://catalog.ldc.upenn.edu/ldc93s1



    HOURS   : 5



    TYPE    : Read - English



    AUTHORS : Garofolo, John, et al.



    TYPE    : LDC Membership



    LICENCE : LDC User Agreement



'''







import errno



import os



from sklearn.externals import joblib



from sklearn.preprocessing import StandardScaler







from sklearn.mixture import GaussianMixture



from os import path



import sys



import tarfile



import fnmatch



import pandas as pd



import subprocess



import argparse



from mapping import phone_maps



import python_speech_features as psf



import scipy.io.wavfile as wav



import numpy as np



from pandas import ExcelWriter











timit_phone_map = phone_maps(mapping_file="kaldi_60_48_39.map")







def underscore(n):
    s=0
    for i in range(len(n)):
        if(n[i]=="_"):
            s=s+1
        if(s==2):
            break
    return(int(n[i+1]))

def underscore1(n):
    s=0
    s1=0
    for i in range(len(n)):
        if(n[i]=="_"):
            s=s+1
        if(n[i]=="/"):
            s1=i
        if(s==2):
            break
    pred=n[s1+1:i]
    return(pred)






def clean(word):



    # LC ALL & strip punctuation which are not required



    new = word.lower().replace('.', '')



    new = new.replace(',', '')



    new = new.replace(';', '')



    new = new.replace('"', '')



    new = new.replace('!', '')



    new = new.replace('?', '')



    new = new.replace(':', '')



    new = new.replace('-', '')



    return new







def compute_mfcc(wav_file, n_delta=0):



    mfcc_feat = psf.mfcc(wav_file)



    if(n_delta == 0):



        return(mfcc_feat)



    elif(n_delta == 1):



        return(np.hstack((mfcc_feat, psf.delta(mfcc_feat,1))))



    elif(n_delta == 2):



        #print(np.hstack((mfcc_feat, psf.delta(mfcc_feat,1), psf.delta(mfcc_feat, 2))).shape)



        return(np.hstack((mfcc_feat, psf.delta(mfcc_feat,1), psf.delta(mfcc_feat, 2))))



    else:



        return 0



def per(mfcc_feat,n_delta):

        #print(mfcc_feat.shape)

        dft=pd.DataFrame(mfcc_feat)

        ll=[]

        ff=open("model_path","r")

        for line in ff:
            ll.append(line[:-1])
            #print(line)
        ans=[]

        mn=[]

        la1=[]

        i=0

        while(i<len(ll)):
            #for i in range(len(ll)-1):
            if(underscore(ll[i])==n_delta):
                #print("tatti")
                model=joblib.load(ll[i])
                #print(ll[i])
                #print(ll[i+1])

                scale=joblib.load(ll[i+1])

                mn.append(ll[i])

                ndef=scale.transform(dft)

                sco=model.score_samples(ndef)

                ans.append(sco.tolist())

            i=i+2

        #print(len(ans[0]))

        #print(len(ans))

        #print(len(mn))

        for i in range(len(ans[0])):

            maxx=-10000

            for j in range(len(ans)):

                try:

                    if(ans[j][i]>maxx):

                        maxx=ans[j][i]

                        mm=mn[j]

                except:

                    print(i,j)

            la1.append(underscore1(mm))

        st=""

        for i in la1:

            st+=i+" "

        return(st)







def read_transcript(full_wav):



    trans_file = full_wav[:-8] + ".PHN"



    with open(trans_file, "r") as file:



        trans = file.readlines()



    durations = [ele.strip().split(" ")[:-1] for ele in trans]



    durations_int = []



    for duration in durations:



        durations_int.append([int(duration[0]), int(duration[1])])



    trans = [ele.strip().split(" ")[-1] for ele in trans]



    trans = [timit_phone_map.map_symbol_reduced(symbol=phoneme) for phoneme in trans]



    # trans = " ".join(trans)



    return trans, durations_int



#def per(filename):





def _preprocess_data(args):

    for n_d in range(3):
        avgg=0
        fcount=0

        datapath = args.timit

        target = path.join(datapath, "TIMIT")

        #print("Preprocessing data")

        #print("Preprocessing Complete")

        #print("Building CSVs")
        mfcc_features = []
        mfcc_labels = []
        d={}
        with open("test_wavs", "r") as file:
            full_wavs = file.readlines()

        full_wavs = [ele.strip() for ele in full_wavs]
        #print(full_wavs)
        for full_wav in full_wavs:
            fcount+=1
            #print("tatti")
            rf = open("reference.txt","w")
            hp = open("hypothesis.txt", "w")
            #print("tatti")
            #print("Computing for file: ", full_wav)
            #orig=full_wav[:-8]+ ".WAV"
            #print(orig)
            #os.system('sox orig full_wav')
            trans, durations = read_transcript(full_wav = full_wav)
            n_delta = n_d
            labels = []
            (sample_rate,wav_file) = wav.read(full_wav)
            mfcc_feats = compute_mfcc(wav_file[durations[0][0]:durations[0][1]], n_delta=n_delta)
            if(n_delta == 0):
                mfcc_feats=np.delete(mfcc_feats, [0] , 1)
            elif(n_delta == 1):
                mfcc_feats=np.delete(mfcc_feats, [0,13] , 1)
            elif(n_delta == 2):
                mfcc_feats=np.delete(mfcc_feats, [0,13,26] , 1)
            for i in range(len(mfcc_feats)):
                    labels.append(trans[0])
            for index, chunk in enumerate(durations[1:]):
                mfcc_feat = compute_mfcc(wav_file[chunk[0]:chunk[1]], n_delta=n_delta)
                #print(mfcc_feat.shape)
                #rf = open("reference.txt","w")
                #hp = open("hypothesis.txt", "w")
                st = ""

                if(n_delta == 0):

                    mfcc_feat=np.delete(mfcc_feat, [0] , 1)

                elif(n_delta == 1):

                    mfcc_feat=np.delete(mfcc_feat, [0,13] , 1)

                elif(n_delta == 2):

                    mfcc_feat=np.delete(mfcc_feat, [0,13,26] , 1)
                #print(mfcc_feat.shape)
                mfcc_feats = np.vstack((mfcc_feats, mfcc_feat))
                for i in range(len(mfcc_feat)):
                    st = st + trans[index] + " "
                    #print(st)
                    labels.append(trans[index])
                #print(st[:-1])
                rf.write(st[:-1] + "\n")
                st1 = per(mfcc_feat,n_delta)
                #print(st1)
                hp.write(st1[:-1] + "\n")
                #print(labels)
            hp.close()
            rf.close()
            os.system("wer reference.txt hypothesis.txt > put.txt")
            pt = open("put.txt","r")
            k =0
            for line in pt:
                linel = line.split(" ")
                if(k == 1):
                    for ii in range(len(linel)):
                        try:
                            
                            avgg+=float(linel[ii][:-1])
                            print(linel[ii])
                            break
                        except:
                            #print(linel[ii])
                            continue
                    #print(linel)
                    break
                k+=1
            
            #print(mfcc_feats.shape) 

            

            mfcc_features.extend(mfcc_feats)

            mfcc_labels.extend(labels)

            #print(mfcc_feats.shape)

        timit_df = pd.DataFrame()

        timit_df["features"] = mfcc_features

        timit_df["labels"] = mfcc_labels

        d={}

        c1=timit_df["features"]

        c2=timit_df["labels"]

        c1=c1.tolist()

        c2=c2.tolist()
        print("=============================")
        print(n_delta)
        print(avgg/fcount)
        print("=============================")


        '''for i in range(len(c2)):

            #print(c2[i])

            #print(c1[i].shape)

            if c2[i] not in d:

                d[c2[i]]=c1[i]

            else:

                d[c2[i]]=np.vstack((d[c2[i]],c1[i]))

        fe=[]

        la=[]

        for i in d:

            sh=d[i].shape

            #print(i)

            #print(sh)

            temp=[i]*sh[0]

            la=la+temp

            ft=d[i].tolist()

            print(len(ft))

            fe=fe+ft

        dft=pd.DataFrame(fe)

        print(dft.shape)

        print(len(la))

        ll=[]

        ff=open("model_path","r")

        for line in ff:

            ll.append(line[:-1])

        ans=[]

        mn=[]

        la1=[]

        i=0

        while(i<len(ll)):

            #for i in range(len(ll)-1):

            if(underscore(ll[i])==n_delta):

                model=joblib.load(ll[i])

                #print(ll[i])

                #print(ll[i+1])

                scale=joblib.load(ll[i+1])

                mn.append(ll[i])

                ndef=scale.transform(dft)

                sco=model.score_samples(ndef)

                ans.append(sco.tolist())

            i=i+2

        print(len(ans[0]))

        print(len(ans))

        print(len(mn))

        for i in range(len(ans[0])):

            maxx=-10000

            for j in range(len(ans)):

                try:

                    if(ans[j][i]>maxx):

                        maxx=ans[j][i]

                        mm=mn[j]

                except:

                    print(i,j)

            la1.append(underscore1(mm))

        print("----------------------------")

        corr=0

        for i in range(len(la)):

            #print(la[i],la1[i])

            if(la[i]==la1[i]):

                corr+=1

        print("==============================")

        print(corr/len(la))

        print(list(d.keys()))

        print(len(list(d.keys())))'''

    



if __name__ == "__main__":







    parser = argparse.ArgumentParser()







    parser.add_argument('--timit', type=str, default="./",



                       help='TIMIT root directory')



    parser.add_argument('--n_delta', type=str, default="2",



                       help='Number of delta features to compute')







    args = parser.parse_args()



    print(args)



    print("TIMIT path is: ", args.timit)



    _preprocess_data(args)



    print("Completed")



