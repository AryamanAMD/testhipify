import os
from fnmatch import fnmatch
import argparse
def getListOfFiles(dirName):
    listOfFile=os.listdir(dirName)
    allFiles=list()
    for entry in listOfFile:
        fullPath=os.path.join(dirName, entry)
        if os.path.isdir(fullPath):
            allFiles=allFiles+getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
                
    return allFiles





def ftale(x):
	x=x.replace('"', '')
	p=os.path.dirname(x)
	q=os.path.basename(x)
	p=p.replace("\\","/")
	os.system("cd "+p)
	with open(p+"/"+q, 'r') as fp:
		lines = fp.readlines()
		for row in lines:
			word = '#include <GL/glu.h>'
			if row.find(word) == 0:
				flag=1
			else:
				flag=0	

	if flag==1:
		print("GL Headers found")	
	else:
		#$ sed 's/checkCudaErrors/HIPCHECK/g' asyncAPI.cu.hip
		command="/opt/rocm/bin/hipify-perl "+p+"/"+q+" > "+p+"/"+q+".hip"
		os.system("echo "+command)
		os.system(command)
		os.system("cd "+p)
		os.system("echo cd "+p)
		s='\"s/checkCudaErrors/HIPCHECK/g\"'
		os.system("echo "+s)
		os.system('echo sed -i.bak \"s/checkCudaErrors/HIPCHECK/g\" '+q)
		#os.system('sed -i.bak ""{}"" '.format(s)+q)
		
		command="/opt/rocm/bin/hipcc -I /home/taccuser/testhipify/src/samples/Common -I /home/taccuser/testhipify/src/samples/Common/GL -I /home/taccuser/testhipify/src/samples/Common/UtilNPP -I /home/taccuser/testhipify/src/samples/Common/data -I /home/taccuser/testhipify/src/samples/Common/lib/x64 "+p+"/"+os.path.basename(x)+".hip"
		os.system("echo "+command)
		os.system(command)
					
		
				
							
				
	

def fall(y):
	y=y.replace('"', '')
	listOfFiles=getListOfFiles(y)
	for elem in listOfFiles:
		if elem.endswith('.cu'):  ##or elem.endswith('.cpp') 
			ftale(elem) 
				
		
        	
            



parser=argparse.ArgumentParser()
parser.add_argument("-t", "--tale", help="Hipify Single Sample")
parser.add_argument("-a", "--all", help="Hipify all Samples")
args=parser.parse_args()
if args.tale:
	x=args.tale
	##print(x)
	ftale(x)
if args.all:
	y=args.all
	##print(y)
	fall(y)

	
