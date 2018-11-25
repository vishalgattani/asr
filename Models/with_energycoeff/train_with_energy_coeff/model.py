import os
def underscore(n):
	for i in range(len(n)):
		if(n[i]=='_'):
			break
	if(i<len(n)-1):
		if(n[i+1]=='s'):
			return(0,i)
		else:
			return(1,i)
	else:
		return(0,0)

path="/Users/adityaankush/Downloads/ASR_p1"
file=os.listdir(path)
lis=[]
for i in file:
	if(i[len(i)-3:]=='pkl' and underscore(i)[0]==1):
		lis.append(os.path.join(path,i))
		nnmame=i[0:underscore(i)[1]+1]+"scale"+i[underscore(i)[1]+1:]
		lis.append(os.path.join(path,nnmame))
ff=open("model_path","w")
for i in lis:
	ff.write(i+"\n")
ff.close()