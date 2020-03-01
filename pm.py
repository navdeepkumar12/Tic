import os

                            #### TRAINING PARAMETERS##########

iterations = 1000*100     #number of iterations for training
alpha= 0.2       # update rate   a = a + ALPHA*(a'-a)
delta = 0
epsilon = 0
path_length = 4
symmetry_update = True
init_reward = 4
initial_Q = False #'data/mnist/param16'     #r# True :- last index init, False:- Constant init, file addres:- file address init
#Reward  WIN = 2, LOOSE = -1, TIE = 1,   0,1,2,3,9  nothing, TIE, 1, 2 , multiple win
#Arg str(player)+'str(result_signal)'
imr = -1  #invalid move reward
w = 2; l = -2; t = 0
reward = {'10':0, '11':w, '12':l, '13':t, '20':0, '21':l, '22':w, '23':t} 
####Plotting parametr
window_length = 100   # for plotting q update, convolution length
tictac_board_adress ="/home/navdeep/TicTac/grid/000000000.png"
########--------------------------------------------------------#######################



                           #### NEURAL NETWORK PARAMETERS ########
momentum = 0.8
learning_rate = 0.001



## ----------------------------G0--------------------------#####
#Board
go_board_adress = '/home/navdeep/TicTac/data/go/board.jpg'
imsize = 1000                      #must be less than shape of imagel
output_adress = '/home/navdeep/TicTac/data/go' #state_in_string.jpg