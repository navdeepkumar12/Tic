import cv2
import pickle
import numpy as np
from graphic import grid
import matplotlib.pyplot as plt
import tools as tl
import time
from graphic import grid
import random


def load_state(filename):
    Q = pickle.load(open("/home/navdeep/TicTac/" + filename, "rb"))
    print("load_state:- load_state:- state loaded")
    return Q

def plot(state):
    # cv2.destroyAllWindows()
    output = grid(state)
    plt.close()
    plt.imshow(output)
    plt.show(block=False)
    # plt.pause(3)
    # cv2.imshow(state, output)
    # cv2.waitKey(2)

def play():
    filename = input('Enter Q file adrress')
    Q = load_state(filename)
    toss = input('Do you want to start')
    state = '000000000'
    plot(state)

    print('you are STAR')
    p2 = 2
    p1 = 1
    if toss in {'y','Y', 'yes', 'Yes'}:
        while tl.Result(state)[0]==0:
            
            print('Current state is {}'.format(state))
            action = input('Enter your action')
            state = tl.SAS(state,action, p2)
            plot(state)
            
            print('Current state is {}'.format(state))
            action = tl.Action(state, Q[state])
            print('Computer chose action {}'.format(action))
            state = tl.SAS(state, action,p1)
            
            plot(state)
            
    else :       
        while tl.Result(state)[0]==0:
            print('Current state is {}'.format(state))
            action = tl.Action(state, Q[state])
            print('Computer choose action {}'.format(action))
            state = tl.SAS(state, action,p1)
            plot(state)
            print('Current state is {}'.format(state))
            action = input('Enter your action')
            state = tl.SAS(state,action, p2)
            plot(state)
            

def play2(a1,a2, no_of_matches):
    Q1 = load_state(a1)
    Q2 = load_state(a2)
    Q = [Q1,Q2]
    R = []
    for i in range(no_of_matches):
        state = '000000000'
        p = 1
        while tl.Result(state)[0]==0:
            action = tl.Action(tl.Toggle(state,p), Q[p-1][tl.Toggle(state,p)])
            state = tl.SAS(state,action, p)
            print(state)
            p = 3-p
        print(tl.Result(state)[0])    
        R.append(tl.Result(state)[0])    
        print(R)
    return R    
play()
#play2('Q1', 'Q129', 100)    