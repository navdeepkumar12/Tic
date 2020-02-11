import numpy as np
import sys
import pickle
import time
import datetime
import pm 
import tools as tl 
import sys 
import init
import matplotlib.pyplot as plt
import os
# # Q = tl.load_state('/home/navdeep/TicTac/Q')
# # Q = pickle(open('Q2', 'wb'))
# Q, filename = tl.initialize_Q()
# S = list(Q.keys())
# # valid start start filtering
# print('list of state generated')
os.system('cat pm.py') #print parameter on the terminal
init.init()  # increment the file index, creates if not available
index = np.loadtxt('index.npy')
index = int(index)

if pm.initial_Q == True:   # change here for Q initialization file
    index_q = index-1
    Q_init_file = 'Q' + str( index_q)
if type(pm.initial_Q) == str:
    Q_init_file = pm.initial_Q
if pm.initial_Q == False:
    index_q = index
    Q_init_file = 'Q' + str( index_q)

Q = pickle.load(open(Q_init_file,'rb'))
print('Q value initialized from {}'.format(Q_init_file))
S = pickle.load(open('state','rb'))
print('main:- Q and S loaded with len {} and {}'.format(len(Q), len(S)))

##Printing parameter file
os.system('cat pm.py >> pm'+str(index)+'.txt')
print('parameter file {} printed'.format('pm'+str(index)+'.txt'))
print('\n TRAINING STARTS \n')
Del_q = []
Mean_q = []
Unexplored_a =[] 
state1 = tl.load_Q('state1')   #states where player 1 plays
for i in range(pm.iterations):
    s = np.random.choice(S)
    #print("{}:- Random state = {}".format(sys.argv[0],s))
    H = tl.Path(s,Q)
    Q, del_q = tl.Update(H,Q)
    Del_q.append(del_q)
    #print('init:- Compeletions {}% \n'.format(i/pm.iterations))

    if i%1000 == 0:
        pickle.dump(Q,open('Q' + str(index),'wb')) 
        #np.savetxt(index+'del_q.npy',Del_q)   
        D = [Q[s] for s in state1]
        D = tl.Filter(D, -1)
        unexplored_action_length = len(list(filter(lambda x:x==pm.init_reward,D)))
        Unexplored_a.append(unexplored_action_length)
        mean_q = np.mean(D)
        Mean_q.append(mean_q)
        print('main:- Q checkpint saved {}th times, convergence mean = {}, unexpld_a ={}'.format(round(i/1000),mean_q,unexplored_action_length))

pickle.dump(Q,open('Q' + str(index),'wb'))    
print('\n int:- TRAINING DONE')

# Del_q, saving and plot
np.savetxt('Del_q'+str(index) +'.npy',Del_q)   
q = abs(np.array(Del_q))
q = np.convolve(q, np.ones(pm.window_length)/pm.window_length)
plt.title('Delta_q')
plt.plot(q)
plt.savefig('Delta_q'+str(index)+ '.png')
plt.close()

# Mean_q
np.savetxt('Mean_q'+str(index) + '.npy',Mean_q )
plt.plot(Mean_q)
plt.title('Mean_q')
plt.savefig('Mean_q'+str(index)+ '.png')
plt.close()
# Unexplored actions length
np.savetxt('Unexplored_a'+str(index) + '.npy',Unexplored_a    )
plt.plot(Unexplored_a   )
plt.title('Unexplored_a')
plt.savefig('Unexplored_a'+str(index)+ '.png')
plt.close()
print('main:- Mean_q, Delta_q fig ,q, Q saved, index = {}'.format(index))