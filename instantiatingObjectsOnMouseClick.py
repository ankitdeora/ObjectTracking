import cv2
myobj=[]

class Animals:
    "class for animals"
    count = 0

    def __init__(self,name='foo',word='woo'):
        self.petname = name
        self.echo = word
        Animals.count+=1

    def speak(self):
        print self.echo

#dog = Animals('tommy','bhow')
#cat = Animals('marcy', 'meow')
#lion = Animals ('simba','dahad')

#TrObj = range(10)

def draw_circle(event,x,y,flags,param):
    global myobj


    if event == cv2.EVENT_LBUTTONDBLCLK:
        pass
        #dog.petname = "scooby"
        #print "dog name changed"

    elif event == cv2.EVENT_LBUTTONDOWN:
        #print dog.petname
        myobj.append(Animals())
        print Animals.count



cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_circle)

while(1):
    if cv2.waitKey(20) == 27:
        break
cv2.destroyAllWindows()

