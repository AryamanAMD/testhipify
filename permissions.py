#os.path.abspath(__file__)
import os
import stat
#os.chmod("testhipify.py", stat.S_IRWXG )
pythonfile = 'testhipify.py'
p1=os.path.abspath(pythonfile)
os.chmod(p1,stat.S_IRWXG)
#incase this doestn work:chmod -R 777 ./
