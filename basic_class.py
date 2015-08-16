#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      ankitdeora2856
#
# Created:     07-05-2015
# Copyright:   (c) ankitdeora2856 2015
# Licence:     <your licence>
#-------------------------------------------------------------------------------

def main():
    pass

if __name__ == '__main__':
    main()

task = 'run'
class Animals:
    "class for animals"
    def __init__(self,name,word):
        global task
        self.petname = name
        self.echo = word
        self.place = 'jodhpur'
        task = "play"
        print task

    def speak(self):
        global task
        task = 'sit'
        print self.echo
        print task


dog = Animals('tommy','bhow')
cat = Animals('marcy', 'meow')

i=0
while i<10:
    i+=1
    task = "jump" + str(i)
    print task


