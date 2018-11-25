import os
import subprocess
#os.system('sox /Users/adityaankush/Downloads/data/lisa/data/timit/raw/TIMIT/TRAIN/DR4/FKDW0/SX217.WAV /Users/adityaankush/Downloads/data/lisa/data/timit/raw/TIMIT/TRAIN/DR4/FKDW0/SX217_rif.WAV')
file=open('/Users/adityaankush/Downloads/ASR_p1/test_wavs','r')
data = file.readlines() 
for line in data: 
	word = line.split() 
	ori=word[0][:-8]+".wav"
	print(ori)
	print(word[0])
	st="sox "+ ori + " "+word[0]
	os.system(st)
	#subprocess.run(["sox", ori,word[0]])
	#print(word) 
