import os
import fileinput
import os.path
from sys import platform
def generate(x):
	x=x.replace('"', '')
	p=os.path.dirname(x)
	q=os.path.basename(x)
	p=p.replace("\\","/")
	os.system("cd "+p)
	command="hipify-perl "+x+" > "+x+".hip"
	print(command)
	os.system(command)
	textToSearch="checkCudaErrors"
	textToReplace="HIPCHECK"
	fileToSearch=p+"/"+q+".hip"
	
	textToSearch1="#include <helper_cuda.h>\n"
	textToReplace1='#include "helper_cuda_hipified.h"\n'
	textToSearch2="#include <helper_functions.h>\n"
	textToReplace2='#include "helper_functions.h"\n'
	
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
	
def apply_patches():
	command='git apply --reject --whitespace=fix src/patches/*.patch'
	print(command)
	os.system(command)

	
def compilation_1(x):
	cpp=[]
	x=x.replace('"', '')
	p=os.path.dirname(x)
	q=os.path.basename(x)
	p=p.replace("\\","/")
	if x=='src/samples/Samples/0_Introduction/simpleMPI/simpleMPI.cu':
		command='hipcc -I src/samples/Common src/samples/Samples/0_Introduction/simpleMPI/simpleMPI.cu.hip src/samples/Samples/0_Introduction/simpleMPI/simpleMPI_hipified.cpp -lmpi -o src/samples/Samples/0_Introduction/simpleMPI/simpleMPI.out'
		print(command)
		os.system(command)
	elif x=='src/samples/Samples/0_Introduction/simpleSeparateCompilation/simpleDeviceLibrary.cu' or x=='/src/samples/Samples/0_Introduction/simpleSeparateCompilation/simpleSeparateCompilation.cu':
		command='hipcc -I src/samples/Common -fgpu-rdc simpleDeviceLibrary.cu.hip simpleSeparateCompilation.cu.hip -o simpleSeparateCompilation.out'
		print(command)
		os.system(command)	
	elif x=='src/samples/Samples/0_Introduction/cudaOpenMP/cudaOpenMP.cu':
		command=' hipcc -I../../../Common -fopenmp cudaOpenMP.cu.hip -o cudaOpenMP.out'
		print(command)
		os.system(command)	
	else:
		for file in os.listdir(p):
			if file.endswith("_hipified.cpp") or file.endswith(".cu.hip"):
				cpp.append(file)
			
		

		cpp = [p+'/'+x for x in cpp]
		command='hipcc -I src/samples/Common -I /usr/local/cuda-12.0/targets/x86_64-linux/include '+' '.join(cpp)+' -o '+p+'/'+os.path.basename(os.path.dirname(x))+'.out'
		print(command)

		os.system(command)	

def runsample(x):	
	print('Processing Sample:'+x)
	command='./'+os.path.dirname(x)+'/'+os.path.basename(os.path.dirname(x))+'.out'
	print(command)
	os.system(command)	

file1 = open('final_ignored_samples.txt', 'r')
Lines = file1.readlines()
for line in Lines:
	line = line.strip('\n')
	generate(line)
	#compilation_1(line)
	#runsample(line)

