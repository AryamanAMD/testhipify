##os.path.abspath(__file__)
import os
import sys
import stat
##os.chmod("C:\Users\aryamish\OneDrive - Advanced Micro Devices Inc\Documents\GitHub\AMD_Projects\testhipify.py", stat.S_IRWXG )
pythonfile = 'testhipify.py'
p1=os.path.abspath(pythonfile)
os.chmod(p1,stat.S_IRWXG)
