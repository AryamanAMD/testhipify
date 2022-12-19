"""
x=open('final_ignored_samples.txt').read();
y=open('custom_samples_path.txt').read();
print (y in x)
"""
lines_seen = set()
outfile = open('final_ignored_samples.txt', "w")
infile = open('final_ignored_samples2.txt', "r")
for line in infile:
	if line not in lines_seen: # not a duplicate
			outfile.write(line)
			lines_seen.add(line)
outfile.close()
with open('final_ignored_samples.txt') as fp:
	data1 = fp.read()	
with open('final_ignored_samples2.txt') as fp:
	data2 = fp.read()
data1 += data2
with open ('final_ignored_samples2.txt', 'a') as fp:
	fp.write(data1)