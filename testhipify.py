import os
from fnmatch import fnmatch
import argparse
import sys
import fileinput
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


def prepend_line(file_name, line):
	#line='#include "HIPCHECK.h"'
    dummy_file = file_name + '.bak'
    with open(file_name, 'r') as read_obj, open(dummy_file, 'w') as write_obj:
        write_obj.write(line + '\n')
        for line in read_obj:
            write_obj.write(line)
    os.remove(file_name)
    os.rename(dummy_file, file_name)
	
    

	


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
		command="/opt/rocm-5.4.0-10890/bin/hipify-perl "+p+"/"+q+" > "+p+"/"+q+".hip"
		os.system("echo "+command)
		os.system(command)
		prepend_line(q,'#include "HIPCHECK.h"')
		textToSearch="checkCudaErrors"
		textToReplace="HIPCHECK"
		fileToSearch=x+".hip"
		tempFile=open(fileToSearch,'r+')
		for line in fileinput.input(fileToSearch):
			tempFile.write(line.replace(textToSearch,textToReplace))
		tempFile.close()	

		#os.system("cd "+p)
		#os.system("echo cd "+p)
		#os.system(r'echo sed -i.bak "s/checkCudaErrors/HIPCHECK/g" '+q)
		#os.system('sed -i.bak ""{}"" '.format(s)+q)

        #/data/driver/testhipify/src/samples/Samples/
		
		#command="/opt/rocm/bin/hipcc -I /home/taccuser/testhipify/src/samples/Common -I /home/taccuser/testhipify/src/samples/Common/GL -I /home/taccuser/testhipify/src/samples/Common/UtilNPP -I /home/taccuser/testhipify/src/samples/Common/data -I /home/taccuser/testhipify/src/samples/Common/lib/x64 "+p+"/"+os.path.basename(x)+".hip"
		#/opt/rocm-5.4.0-10890/bin
		command="/opt/rocm-5.4.0-10890/bin/hipcc -I /data/driver/testhipify/src/samples/Common -I /data/driver/testhipify/src/samples/Common/GL -I /data/driver/testhipify/src/samples/Common/UtilNPP -I /data/driver/testhipify/src/samples/Common/data -I /data/driver/testhipify/src/samples/Common/lib/x64 "+p+"/"+os.path.basename(x)+".hip"
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

	
