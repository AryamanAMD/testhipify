import os
def getListOfFiles(dirName):
    listOfFile = os.listdir(dirName)
    allFiles = list()
    for entry in listOfFile:
        fullPath = os.path.join(dirName, entry)
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
                
    return allFiles





def ftale(x):
	x=x.replace('"', '')
	command="perl hipify-perl "+os.path.basename(x)+" > "+os.path.basename(x)+".hip"
	os.system(command)
	##command="hipcc "+os.path.basename(x)+" -o "+ square.out"
	##print(command)
	

def fall(y):
	"""
	dir_list = os.listdir(y)
	for i in dir_list:
		if i.lower().endswith('.cu') or i.lower().endswith('.cpp'):
			command="perl hipify-perl "+i+" > "+i+".hip"
			os.system(command)
			
	pattern = ["*.cu","*.cpp"]
	for path, subdirs, files in os.walk(y):
		for name in files:
			if fnmatch(name, pattern):
				##print(os.path.join(path, name))
				f=os.path.join(path, name)
				command="perl hipify-perl "+f+" > "+f+".hip"
				##os.system(command)
				print(command)
				"""
	y=y.replace('"', '')
	listOfFiles = getListOfFiles(y)
	for elem in listOfFiles:
		if elem.endswith('.cu'):  ##or elem.endswith('.cpp') 
			ftale(elem) 
				
		
        	
            

from fnmatch import fnmatch
import argparse

parser=argparse.ArgumentParser()
parser.add_argument("-t", "--tale", help = "Hipify Single Sample")
parser.add_argument("-a", "--all", help = "Hipify all Samples")
args=parser.parse_args()
if args.tale:
	x=args.tale
	##print(x)
	ftale(x)
if args.all:
	y=args.all
	##print(y)
	fall(y)

	
