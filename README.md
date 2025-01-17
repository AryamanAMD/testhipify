# TestHIPIFY
Refer to Documentation [HERE](https://github.com/AryamanAMD/testhipify/blob/main/Testhipify%20Report-PDF.pdf)
## Installation
### Setting up dependancies
```bash
git clone https://github.com/AryamanAMD/testhipify.git
```
```bash
cd testhipify
```
```bash
python testhipify.py -s
```
This command would ensure installation of CUDA,GCC,OpenMP and OpenMPI[Installation may take upto 10 minutes].
### How to Process Samples
For a single sample:
```bash
python testhipify.py -t '[Path to that Sample]'
```
Example:
```bash
python testhipify.py -t 'src/samples/Samples/0_Introduction/Clock/clock.cu'
```
For all samples:
```bash
python testhipify.py -a 'src/samples/Samples'
```
### Usage
-a or --all: To run HIPIFY-perl for all samples

-b or --generate: Generate .hip files

-c or --compile1: Compile .hip files

-d or --compile2: Compile .hip files with static libraries

-e or --execute: Execute .out files

-f or --generate_all: Generate all .hip files

-g or --compile1_all: Compile all .hip files

-i or --compile2_all: Compile all .hip files with static libraries[DEPRECATED as of May 2023]

-j or --execute_all: Execute all .out files

-k or --parenthesis_check: Remove last parts from cu.hip files which are out of bounds.

-l or --parenthesis_check_all: Remove all last parts from cu.hip files which are out of bounds.

-p or --patch: Apply all patches in src/patches

-t or --tale: To run HIPIFY-perl for a single sample

-x or --remove: Remove any sample relating to graphical operations

-s or --setup: Configure dependencies.

### Screenshots

![image](https://user-images.githubusercontent.com/115460120/215019805-efe0a5eb-5520-4b90-8bb5-81636a79afd1.png)
