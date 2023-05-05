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
	global cuda_path
	global user_platform
	cpp=[]
	print(user_platform)
	x=x.replace('"', '')
	p=os.path.dirname(x)
	p=p.replace("\\","/")
	for file in os.listdir(p):
		if file.endswith(".out") or file.endswith(".o"):
			os.remove(os.path.join(p,file))
	try:		
		for file in os.listdir(p):		
			if os.path.getsize(os.path.join(p,file))==0 and file in os.listdir(p.replace("src/","src-original/")):
				x_original=x.replace("src/","src-original/")
				alternate_file=x_original
				os.remove(os.path.join(p,file))
				os.rename(alternate_file,os.path.join(p,file))
	except FileNotFoundError as e:
		print(f'Error:{e}.Skipping replacement of empty files.')						
	if user_platform.lower()=='nvidia':
		for file in os.listdir(p):
			if file.endswith("_hipified.cpp") or file.endswith(".cu.cpp"):
				cpp.append(file)	
	elif user_platform.lower()=='amd':	
		for file in os.listdir(p):
			if file.endswith("_hipified.cpp") or file.endswith(".cu.hip"):
				cpp.append(file)
	cpp = [p+'/'+y for y in cpp]
	file4=open('multithreaded_samples.txt', 'r')
	threaded_samples=file4.read()
	#print(threaded_samples)
	if x in threaded_samples:
		command='hipcc -I /opt/rocm/include -fopenmp -fgpu-rdc -I src/samples/Common -I '+cuda_path+' '+' '.join(cpp)+' -lamdhip64 -o '+p+'/'+os.path.basename(os.path.dirname(x))+'.out '
	else:
		command='hipcc -I /opt/rocm/include -I src/samples/Common -I '+cuda_path+' '+' '.join(cpp)+' -lamdhip64 -o '+p+'/'+os.path.basename(os.path.dirname(x))+'.out'
	file4.close()	
	print(command)
	os.system(command)

def runsample(x):	
	print('Processing Sample:'+x)
	command='./'+os.path.dirname(x)+'/'+os.path.basename(os.path.dirname(x))+'.out'
	print(command)
	os.system(command)	

file1 = open('run_samples_here.txt', 'r')
Lines = file1.readlines()
for line in Lines:
	line = line.strip('\n')
	generate(line)
	compilation_1(line)
	runsample(line)

