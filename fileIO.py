f.close()
f=open('C:\\Users\\ankitdeora2856\\Desktop\\myfile1.txt','a')
#for i in range(65,91):
    #f.write('\n'+ chr(i))
x1,y1,x2,y2 = 20,30,40,50


f.write(str(x1)+','+ str(y1)+','+ str(x2)+','+ str(y2)+',')
##f.write('deora\n')
f.seek(0,0)
print f.tell()

#f.close()

##f=open('C:\\Users\\ankitdeora2856\\Desktop\\myfile.txt','r')
##for line in f:
##    print line
