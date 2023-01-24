usage: testhipify.py [-h] [-a ALL] [-b GENERATE] [-c COMPILE1] [-d COMPILE2] [-e EXECUTE] [-f GENERATE_ALL] [-g COMPILE1_ALL]
                     [-i COMPILE2_ALL] [-j EXECUTE_ALL] [-k PARENTHESIS_CHECK] [-l PARENTHESIS_CHECK_ALL] [-p] [-t TALE] [-x REMOVE] [-s]

HIPIFY Cuda Samples.Please avoid and ignore samples with graphical operations

options:
  -h, --help            show this help message and exit
  -a ALL, --all ALL     To run hipify-perl for all sample:python testhipify.py --all "[PATH TO SAMPLE FOLDER]"
  -b GENERATE, --generate GENERATE
                        Generate .hip files
  -c COMPILE1, --compile1 COMPILE1
                        Compile .hip files
  -d COMPILE2, --compile2 COMPILE2
                        Compile .hip files with static libraries
  -e EXECUTE, --execute EXECUTE
                        Execute .out files
  -f GENERATE_ALL, --generate_all GENERATE_ALL
                        Generate all .hip files
  -g COMPILE1_ALL, --compile1_all COMPILE1_ALL
                        Compile all .hip files
  -i COMPILE2_ALL, --compile2_all COMPILE2_ALL
                        Compile all .hip files with static libraries
  -j EXECUTE_ALL, --execute_all EXECUTE_ALL
                        Execute all .out files
  -k PARENTHESIS_CHECK, --parenthesis_check PARENTHESIS_CHECK
                        Remove last parts from cu.hip files which are out of bounds.
  -l PARENTHESIS_CHECK_ALL, --parenthesis_check_all PARENTHESIS_CHECK_ALL
                        Remove all last parts from cu.hip files which are out of bounds.
  -p, --patch           Apply all patches in src/patches
  -t TALE, --tale TALE  To run hipify-perl for single sample:python testhipify.py -t "[PATH TO SAMPLE]"
  -x REMOVE, --remove REMOVE
                        Remove any sample relating to graphical operations e.g.DirectX,Vulcan,OpenGL,OpenCL and so on.
  -s, --setup           Configure CUDA Installation
