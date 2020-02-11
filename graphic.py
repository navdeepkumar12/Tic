#import imutils
import cv2
import sys
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import tools as tl

def location(state_in_string):
    if len(state_in_string) !=9:
        print('location:- state_in_string is of length {}'.format(len(state_in_string)))
        return 
    circle_int_location = []
    star_int_location  = []    
    for num, loc in enumerate(state_in_string):
        if loc =='1':
            circle_int_location.append(num)
        if loc == '2':
            star_int_location.append(num)
    return circle_int_location, star_int_location

def location_to_cordinate(player_int_location):
    coordinate = []
    for l in player_int_location:
        x = 200*(l%3) + 100     # X coordinate
        y = 200*int(np.floor(l/3)) + 100  # Y coordinate 
        coordinate.append((x,y))
        # print('x,y,l)= ({},{},{})'.format(x,y,l))
    return coordinate

def star(output, centre):
    t = 40
    (x,y) = centre
    output = cv2.line(output, (x-t, y-t), (x+t, y+t), (200,0,0), thickness=5, lineType=8, shift=0)
    output = cv2.line(output, (x-t, y+t), (x+t, y-t), (200,0,0), thickness=5, lineType=8, shift=0)
    return output

def circle(output,centre):
    cv2.circle(output, centre , 40 , (0, 0, 255), 2)    
    return output

def wline(output, finish_check_state_in_string):
    temp = np.array(map(int,finish_check_state_in_string))
    temp = np.where(temp > 1)
    coordinate = location_to_cordinate(temp)
    (x1,y1) = coordinate[0]
    (x2,y2) = coordinate[2]
    output = cv2.line(output, (x1, y1), (x2, y2), (100,100,0), thickness=5, lineType=8, shift=0)
    

def plot(output, coordinate1, coordinate2)  :
    for centre1 in coordinate1:
        output = circle(output, centre1 )
    for centre2 in coordinate2:
        output = star(output, centre2)
    return output



def grid(state_in_string):
    image = cv2.imread("grid3.png")
    (h,w,d) = image.shape
    #print('shape of image', (h,w,d))
    output = image.copy()

    circle_int_location, star_int_location = location(state_in_string)
    coordinate1 = location_to_cordinate(circle_int_location)
    coordinate2 = location_to_cordinate(star_int_location)
    print('coordinate1 =',coordinate1,'coordinate2 =', coordinate2)
    output = plot(output,coordinate1,coordinate2)    
    #cv2.imshow(state_in_string, output)
    # cv2.imwrite('grid/'+state_in_string+'.png', output)
    return output
#state_in_string = sys.argv[1]
#grid(state_in_string)


##OTHER PLOTING FUNCTION
def dict_to_list(D,fil='Null'):
    if type(D)==dict:
        D = D.values()
        D = list(D)
        D = np.array(D)
        D = D.flatten()
        if fil != 'Null':
            D = filter(lambda d: d != fil, D)
            D = list(D)
        return D
def Q_hist(Q, fil ='Null') :
    if type(Q) ==str:
        Q = tl.load_Q(Q)
    if type(Q) == dict:
        D = dict_to_list(Q,fil) 

    plot = plt.hist(D)
    return plot