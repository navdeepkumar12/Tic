import numpy as np
import sys
import pickle
import time
import datetime
import pm 
import tools as tl 
import sys 
import os

def init():
    
    if os.path.exists('index.npy'):
        index = np.loadtxt('index.npy')
        index = int(index) +1
    else :
        index = 10
    np.savetxt('index.npy', [index])
    
    Q = tl.initialize_Q()
    fl_Qtemp = os.getcwd()+'/data/Q' + str(index)
    pickle.dump(Q,open(fl_Qtemp,'wb'))
    print('init:- State saved at /TicTac/state and Q saved at TicTac/{}'.format(fl_Qtemp))
    print('init:- Q initialized to zero value, filename ={}'.format('Q' + str(index)))

    if os.path.exists('state'):
        S = pickle.load(open('state', 'rb'))
        print('init:- State file found and loaded')
    else:
        S = list(Q.keys())
        print('init:- len of all state = {}'.format(len(S)))
        S = list(filter(tl.Valid_start_state, S))
        print('init:- Valid start state filtered in, now valid state len = {}'.format(len(S)))
        pickle.dump(S,open('state','wb'))

    # valid start start filtering
    tl.cprint('init:- Successfully done!\n')

