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

def _preprocess_data(args):
    for n_d in range(3):
        datapath = args.timit
        target = path.join(datapath, "TIMIT")

        print("Preprocessing data")

        print("Preprocessing Complete")
        print("Building CSVs")
        
        mfcc_features = []
        mfcc_labels = []
        d={}
        with open("train_wavs", "r") as file:
            full_wavs = file.readlines()
        full_wavs = [ele.strip() for ele in full_wavs]
        for full_wav in full_wavs:
            print("Computing for file: ", full_wav)
            #orig=full_wav[:-8]+ ".WAV"
            #print(orig)
            #os.system('sox orig full_wav')

            trans, durations = read_transcript(full_wav = full_wav)
            n_delta = n_d
            labels = []

            (sample_rate,wav_file) = wav.read(full_wav)
            mfcc_feats = compute_mfcc(wav_file[durations[0][0]:durations[0][1]], n_delta=n_delta)

            for i in range(len(mfcc_feats)):
                    labels.append(trans[0])
            for index, chunk in enumerate(durations[1:]):
                mfcc_feat = compute_mfcc(wav_file[chunk[0]:chunk[1]], n_delta=n_delta)
                mfcc_feats = np.vstack((mfcc_feats, mfcc_feat))
                for i in range(len(mfcc_feat)):
                    labels.append(trans[index])
            mfcc_features.extend(mfcc_feats)
            mfcc_labels.extend(labels)
        timit_df = pd.DataFrame()
        timit_df["features"] = mfcc_features
        timit_df["labels"] = mfcc_labels
        d={}
        c1=timit_df["features"]
        c2=timit_df["labels"]
        c1=c1.tolist()
        c2=c2.tolist()
        for i in range(len(c2)):
            print(c2[i])
            if c2[i] not in d:
                d[c2[i]]=c1[i]
            else:
                d[c2[i]]=np.vstack((d[c2[i]],c1[i]))
        '''for i in d:
            sa=i
            df=d[i]
            break
        for i in d:
            if(sa!=i):
                df=np.vstack((df,d[i]))'''
        


    
    

        print(list(d.keys()))
        print(len(list(d.keys())))
        #writer = ExcelWriter('mfcc.xlsx')
        #timit_df.to_excel(writer,index=False,columns=list(timit_df.columns))
        #writer.save()
        #timit_df.to_hdf("timit.hdf", "timit")
        #mod=GaussianMixture(n_components=2,covariance_type='diag')
        #k = 2
        print(n_delta)
        for i in d:
            if(n_delta == 0):
                k=2
                while(k<=256):
                    mod=GaussianMixture(n_components=k,covariance_type='diag')
                    std=StandardScaler()
                    sc_mod=std.fit(d[i])
                    d1=sc_mod.transform(d[i])
                    joblib.dump(sc_mod, str(i) + "_scale" + str(n_delta) + "_" + str(k)+".pkl")
                    mod.fit(d1)
                    joblib.dump(mod, str(i) + "_" + str(n_delta) + "_" + str(k)+".pkl")
                    k = k*2
            else:
                    mod=GaussianMixture(n_components=64,covariance_type='diag')
                    std=StandardScaler()
                    sc_mod=std.fit(d[i])
                    d1=sc_mod.transform(d[i])
                    joblib.dump(sc_mod, str(i) + "_scale" + str(n_delta) + "_64" +".pkl")
                    mod.fit(d1)
                    joblib.dump(mod, str(i) + "_" + str(n_delta) + "_64" + ".pkl")




    
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
