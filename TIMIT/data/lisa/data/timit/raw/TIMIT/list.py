import os
 
path = '/Users/adityaankush/Downloads/data/lisa/data/timit/raw/TIMIT/TEST'
new1=[]
files = os.listdir(path)
for name in files:
	if(name[0]=="D"):
		new1.append(os.path.join(path,name))
    	#print(os.path.join(path,name))
	
new2=[]
for i in new1:
	f1=os.listdir(i)
	for j in f1:
		if(j[0]!='.'):
			new2.append(os.path.join(i,j))
new3=[]
for i in new2:
	f2=os.listdir(i)
	for j in f2:
		#print(j)

		if(j[-1]=='V' and j[-3]=='W'):
			j=j[:-4]+"_rif"+".WAV"
			print(j)
			new3.append(os.path.join(i,j))
ff=open("test_wavs","w")
for i in new3:
	ff.write(i+'\n')
ff.close()