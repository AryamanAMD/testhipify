from genericpath import isfile
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
	"""
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
		"""
	#$ sed 's/checkCudaErrors/HIPCHECK/g' asyncAPI.cu.hip
	command="/opt/rocm-5.4.0-10890/bin/hipify-perl "+p+"/"+q+" > "+p+"/"+q+".hip"
	os.system("echo "+command)
	os.system(command)
	prepend_line(p+"/"+q+".hip",'#include "HIPCHECK.h"')
	textToSearch="checkCudaErrors"
	textToReplace="HIPCHECK"
	fileToSearch=x+".hip"
	textToSearch1="hipProfilerStart"
	textToReplace1="roctracer_start"
	textToSearch2="hipProfilerStop"
	textToReplace2="roctracer_stop"
	tempFile=open(fileToSearch,'r+')
	for line in fileinput.input(fileToSearch):
		tempFile.write(line.replace(textToSearch,textToReplace))
	tempFile.close()	
	tempFile=open(fileToSearch,'r+')
	for line in fileinput.input(fileToSearch):
		tempFile.write(line.replace(textToSearch1,textToReplace1))
	tempFile.close()
	tempFile=open(fileToSearch,'r+')
	for line in fileinput.input(fileToSearch):
		tempFile.write(line.replace(textToSearch2,textToReplace2))	
	tempFile.close()
	

		#os.system("cd "+p)
		#os.system("echo cd "+p)
		#os.system(r'echo sed -i.bak "s/checkCudaErrors/HIPCHECK/g" '+q)
		#os.system('sed -i.bak ""{}"" '.format(s)+q)

        #/data/driver/testhipify/src/samples/Samples/
		
		#command="/opt/rocm/bin/hipcc -I /home/taccuser/testhipify/src/samples/Common -I /home/taccuser/testhipify/src/samples/Common/GL -I /home/taccuser/testhipify/src/samples/Common/UtilNPP -I /home/taccuser/testhipify/src/samples/Common/data -I /home/taccuser/testhipify/src/samples/Common/lib/x64 "+p+"/"+os.path.basename(x)+".hip"
		#/opt/rocm-5.4.0-10890/bin
	command='/opt/rocm-5.4.0-10890/bin/hipcc -I/data/driver/testhipify/src/samples/Common -I/data/driver/testhipify/src/samples/Common/GL -I/data/driver/testhipify/src/samples/Common/UtilNPP -I/data/driver/testhipify/src/samples/Common/data -I/data/driver/testhipify/src/samples/Common/lib/x64 '+p+'/'+q+'.hip -o '+os.path.splitext(x)[0]+'.out'
	os.system("echo "+command)
	os.system(command)

    #hipcc  square.cpp -o square.out(done)
	command='hipcc '+p+'/'+os.path.basename(x)+'.hip -o '+p+'/'+os.path.splitext(x)[0]+'.out'
	os.system(command)
	#/home/user/hip/bin/hipcc -use-staticlib  square.cpp -o square.out.static
	command='/opt/rocm-5.4.0-10890/bin/hipcc -use-staticlib '+p+'/'+q+' -o '+p+'/'+os.path.splitext(x)[0]+'.out.static'
	os.system(command)
	#./square.out
	command='./'+os.path.splitext(x)[0]+'.out'
	os.system(command)

					
		
				
							
				
	

def fall(y):
	y=y.replace('"', '')
	listOfFiles=getListOfFiles(y)
	for elem in listOfFiles:
		if elem.endswith('.cu'):  ##or elem.endswith('.cpp') 
			with open('final_ignored_samples.txt','r') as f:
				if elem in f.read():
					print("Ignoring this sample "+elem)
				else:
					ftale(elem)

			


def rem(z):
	
	a=open("samples_to_be_ignored.txt","r+")
	a.truncate(0)

		
	b=open("final_ignored_samples.txt", 'w')
	b.close()	
	z=z.replace('"','')
	#ignore_list = ['<GL/', '<screen', '<drm.h>','FDTD3dGPU.h','<d312',' <GLES3/']
	ignore_list = ['<GL/','<screen/screen.h>', '<drm.h>','"FDTD3dGPU.h"','<d3d12.h>',' <GLES3/gl31.h>','<EGL/egl.h>','<GLFW/glfw3.h>','"cudla.h"']
	listofFiles=getListOfFiles(z)
	for elem in listofFiles:
		if elem.endswith('.cu'):
			with open(elem) as f:
				for line in f:
					if any(word in line for word in ignore_list):
						a.write(elem+"\n")
	
	a.close()
	lines_seen = set()
	outfile = open('final_ignored_samples.txt', "w")
	infile = open('samples_to_be_ignored.txt', "r")
	for line in infile:
		if line not in lines_seen: # not a duplicate
			outfile.write(line)
			lines_seen.add(line)
	outfile.close()	
		
        
    

    		


						



				
		
        	
            



parser=argparse.ArgumentParser(description ='HIPIFY Cuda Samples.Please avoid and ignore samples with graphical operations')
parser.add_argument("-x", "--remove", help='Remove any sample relating to graphical operations e.g.DirectX,Vulcan,OpenGL,OpenCL and so on.')
parser.add_argument("-t", "--tale", help='To run hipify-perl for single sample:python testhipify.py -t "[PATH TO SAMPLE]"')
parser.add_argument("-a", "--all", help='To run hipify-perl for all sample:python testhipify.py --all "[PATH TO SAMPLE FOLDER]"')
args=parser.parse_args()
if args.tale:
	x=args.tale
	##print(x)
	ftale(x)
if args.all:
	y=args.all
	##print(y)
	fall(y)
if args.remove:
	z=args.remove
	rem(z)





	
