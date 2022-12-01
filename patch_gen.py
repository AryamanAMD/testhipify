import os
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


def generate(x):
    x=x.replace('"', '')
    x=x.replace("\\","/")
    #print(os.path.splitext("/path/to/some/file.txt")[0])
    p=os.path.dirname(x)
    p=p.replace("\\","/")
    q=os.path.basename(x)
    r=os.path.splitext(q)[0]
    command="hipify-perl "+x+" > "+p+'/'+r+"_hipified.cpp"
    print(command)
    os.system(command)
    replace_words(x,'#include <helper_cuda.h>','#include "helper_cuda_hipified.h"')
    replace_words(x,"#include <helper_functions.h>",'#include "helper_functions.h"')
	
	  
def generate_all(y):
	y=y.replace('"', '')
	listOfFiles=getListOfFiles(y)
	for elem in listOfFiles:
		if elem.endswith('_gold.cpp') or elem.endswith('main.cpp'):  ##or elem.endswith('.cpp') 
                    generate(elem)


def replace_words(x,search_text,replace_text):
    p=os.path.dirname(x)
    q=os.path.basename(x)
    r=os.path.splitext(q)[0]
    with open(p+'/'+r+"_hipified.cpp", 'r') as file:
        data = file.read()
        data = data.replace(search_text, replace_text)
    with open(p+'/'+r+"_hipified.cpp", 'w') as file:
        file.write(data)
					    
generate_all('src/samples/Samples')                        
