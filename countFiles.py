import os, os.path

#print sum((len(f) for _, _, f in os.walk('C:\\Users\\ankitdeora2856\\Desktop\\project 2015\\vot challenge\\vot-toolkit\\vot-workspace\\sequences\\woman\\')))

##tifCounter = 0
##for root, dirs, files in os.walk('C:\\Users\\ankitdeora2856\\Desktop\\project 2015\\vot challenge\\vot-toolkit\\vot-workspace\\sequences\\ball\\'):
##    for file in files:
##        if file.endswith('.jpg'):
##            tifCounter += 1
##
##print tifCounter

tifCounter = 0
for root, dirs, files in os.walk('C:\\Users\\ankitdeora2856\\Desktop\\project 2015\\vot challenge\\vot2014\\'):
    for dir in dirs:
        #if file.endswith('.jpg'):
            tifCounter += 1

print tifCounter