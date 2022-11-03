import os
import argparse
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


def parenthesis_check(file_name):
	string=''
	p=os.path.dirname(file_name)
	#f = open("a.cu.hip", "w")
	file=open(file_name,'r')
	file2=open(file_name,'r')
	while 1:
		char=file.read(1)
		if not char:
			break
		if char=='{' or char=='}':
			string+=char
	#print(string)
	result=check(string)
	#print(result)
	if result==1:
		#for line in file2:
			#print(line)
		lines = file2.readlines()
		lines = lines[:-1]	
		for elem in reversed(lines):
			if '}\n' not in elem:
				lines = lines[:-1]	
			else:
				break	
				
		#print(lines)	
		with open(p+'/a.cu.hip','w') as fp:
			for item in lines:
				fp.write(item)
		file.close()
		file2.close()
		os.remove(file_name)
		os.rename(p+'/a.cu.hip', file_name)

			
def parenthesis_check_all(y):
	y=y.replace('"', '')
	listOfFiles=getListOfFiles(y)
	for elem in listOfFiles:
		if elem.endswith('.cu.hip'):  ##or elem.endswith('.cpp') 
			parenthesis_check(elem)			

		


		


        		


open_list = ["{"]
close_list = ["}"]
def check(myStr):
    stack = []
    for i in myStr:
        if i in open_list:
            stack.append(i)
        elif i in close_list:
            pos = close_list.index(i)
            if ((len(stack) > 0) and
                (open_list[pos] == stack[len(stack)-1])):
                stack.pop()
            else:
                return 1
    if len(stack) == 0:
        return 0
    else:
        return 1





	
    
def ftale(x):
	generate(x)
	compilation_1(x)
	compilation_2(x)
	runsample(x)
	
def generate_all(y):
	y=y.replace('"', '')
	listOfFiles=getListOfFiles(y)
	for elem in listOfFiles:
		if elem.endswith('.cu'):  ##or elem.endswith('.cpp') 
			with open('final_ignored_samples.txt','r') as f:
				if elem in f.read():
					print("Ignoring this sample "+elem)
				else:
					generate(elem)

def compilation_1_all(y):
	y=y.replace('"', '')
	listOfFiles=getListOfFiles(y)
	for elem in listOfFiles:
		if elem.endswith('.cu'):  ##or elem.endswith('.cpp') 
			with open('final_ignored_samples.txt','r') as f:
				if elem in f.read():
					print("Ignoring this sample "+elem)
				else:
					compilation_1(elem)	

def compilation_2_all(y):
	y=y.replace('"', '')
	listOfFiles=getListOfFiles(y)
	for elem in listOfFiles:
		if elem.endswith('.cu'):  ##or elem.endswith('.cpp') 
			with open('final_ignored_samples.txt','r') as f:
				if elem in f.read():
					print("Ignoring this sample "+elem)
				else:
					compilation_2(elem)		

def runsample_all(y):
	y=y.replace('"', '')
	listOfFiles=getListOfFiles(y)
	for elem in listOfFiles:
		if elem.endswith('.cu'):  ##or elem.endswith('.cpp') 
			with open('final_ignored_samples.txt','r') as f:
				if elem in f.read():
					print("Ignoring this sample "+elem)
				else:
					runsample(elem)														

def generate(x):
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
	command="hipify-clang "+x
	print(command)
	os.system(command)
	prepend_line(p+"/"+q+".hip",'#include "HIPCHECK.h"')
	prepend_line(p+"/"+q+".hip",'#include "rocprofiler.h"')
	textToSearch="checkCudaErrors"
	textToReplace="HIPCHECK"
	fileToSearch=p+"/"+q+".hip"
	textToSearch1="hipProfilerStart"
	textToReplace1="rocprofiler_start"
	textToSearch2="hipProfilerStop"
	textToReplace2="rocprofiler_stop"
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
	parenthesis_check(x+".hip")

def apply_patches():
	command='git apply --reject --whitespace=fix src/patches/*.patch'
	print(command)
	os.system(command)

	
def compilation_1(x):
	x=x.replace('"', '')
	p=os.path.dirname(x)
	q=os.path.basename(x)
	p=p.replace("\\","/")
	command='hipcc -I src/samples/Common '+p+'/'+q+'.hip -o '+os.path.splitext(x)[0]+'.out'
	print(command)
	os.system(command)

def compilation_2(x):
	x=x.replace('"', '')
	p=os.path.dirname(x)
	q=os.path.basename(x)
	p=p.replace("\\","/")
	command='hipcc -I src/samples/Common -use-staticlib '+p+'/'+q+'.hip -o '+os.path.splitext(x)[0]+'.out.static'
	print(command)
	os.system(command)

def runsample(x):	
	command='./'+os.path.splitext(x)[0]+'.out'
	print(command)
	os.system(command)
	

	

		#os.system("cd "+p)
		#os.system("echo cd "+p)
		#os.system(r'echo sed -i.bak "s/checkCudaErrors/HIPCHECK/g" '+q)
		#os.system('sed -i.bak ""{}"" '.format(s)+q)

        #/data/driver/testhipify/src/samples/Samples/
		
		#command="/opt/rocm/bin/hipcc -I /home/taccuser/testhipify/src/samples/Common -I /home/taccuser/testhipify/src/samples/Common/GL -I /home/taccuser/testhipify/src/samples/Common/UtilNPP -I /home/taccuser/testhipify/src/samples/Common/data -I /home/taccuser/testhipify/src/samples/Common/lib/x64 "+p+"/"+os.path.basename(x)+".hip"
		#/opt/rocm-5.4.0-10890/bin
	#command='/opt/rocm-5.4.0-10890/bin/hipcc -I /long_pathname_so_that_rpms_can_package_the_debug_info/data/driver/testhipify/src/samples/Common -I /long_pathname_so_that_rpms_can_package_the_debug_info/data/driver/testhipify/src/samples/Common/UtilNPP -I /long_pathname_so_that_rpms_can_package_the_debug_info/data/driver/testhipify/src/samples/Common/data -I /long_pathname_so_that_rpms_can_package_the_debug_info/data/driver/testhipify/src/samples/Common/lib/x64 '+p+'/'+q+'.hip -o '+p+'/'+os.path.splitext(x)[0]+'.out'
	#print(command)
	#os.system(command)

    #hipcc  square.cpp -o square.out(done)
	#command='hipcc '+p+'/'+os.path.basename(x)+'.hip -o '+p+'/'+os.path.splitext(x)[0]+'.out'
	#print(command)
	#os.system(command)
	#/home/user/hip/bin/hipcc -use-staticlib  square.cpp -o square.out.static
	#command='/opt/rocm-5.4.0-10890/bin/hipcc -use-staticlib '+p+'/'+q+'.hip -o '+p+'/'+os.path.splitext(x)[0]+'.out.static'
	#print(command)
	#os.system(command)
	#./square.out
	#command='./'+os.path.splitext(x)[0]+'.out'
	#print(command)
	#os.system(command)
	#command2='hipcc -I /long_pathname_so_that_rpms_can_package_the_debug_info/data/driver/testhipify/src/samples/Common asyncAPI.cu.hip -o asynAPI.out'
    
	
					
		
				
							
				
	

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
	print("This script automates sample exclusion.Please backup any paths provided by you to avoid loss or overwriting.")
	input("Press Enter to continue...")
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
parser.add_argument("-a", "--all", help='To run hipify-perl for all sample:python testhipify.py --all "[PATH TO SAMPLE FOLDER]"')
parser.add_argument("-b", "--generate", help='Generate .hip files')
parser.add_argument("-c", "--compile1", help='Compile .hip files')
parser.add_argument("-d", "--compile2", help='Compile .hip files with static libraries')
parser.add_argument("-e", "--execute", help='Execute .out files')
parser.add_argument("-f", "--generate_all", help='Generate all .hip files')
parser.add_argument("-g", "--compile1_all", help='Compile all .hip files')
parser.add_argument("-i", "--compile2_all", help='Compile all .hip files with static libraries')
parser.add_argument("-j", "--execute_all", help='Execute all .out files')
parser.add_argument("-k", "--parenthesis_check", help='Remove last parts from cu.hip files which are out of bounds.')
parser.add_argument("-l", "--parenthesis_check_all", help='Remove all last parts from cu.hip files which are out of bounds.')
parser.add_argument("-p", "--patch", help='Apply all patches in src/patches',action='store_true')
parser.add_argument("-t", "--tale", help='To run hipify-perl for single sample:python testhipify.py -t "[PATH TO SAMPLE]"')
parser.add_argument("-x", "--remove", help='Remove any sample relating to graphical operations e.g.DirectX,Vulcan,OpenGL,OpenCL and so on.')




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
if args.generate:
	a=args.generate
	generate(a)
if args.patch:
	apply_patches()	
if args.compile1:
	b=args.compile1
	compilation_1(b)
if args.compile2:
	c=args.compile2
	compilation_2(c)
if args.execute:
	d=args.execute
	runsample(d)	
if args.generate_all:
	a=args.generate_all
	generate_all(a)
if args.compile1_all:
	b=args.compile1_all
	compilation_1_all(b)
if args.compile2_all:
	c=args.compile2_all
	compilation_2_all(c)
if args.execute_all:
	d=args.execute_all
	runsample_all(d)
if args.parenthesis_check:
	e=args.parenthesis_check
	parenthesis_check(e)
if args.parenthesis_check_all:
	f=args.parenthesis_check_all
	parenthesis_check_all(f)					







	
