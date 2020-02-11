

# update rate   a = a + ALPHA*(a'-a)
alpha= 0.2
delta = 0
epsilon = 0
path_length = 4
symmetry_update = True
init_reward = 4
initial_Q = False     #r# True :- last index init, False:- Constant init, file addres:- file address init
#Reward  WIN = 2, LOOSE = -1, TIE = 1,   0,1,2,3,9  nothing, TIE, 1, 2 , multiple win
#Arg str(player)+'str(result_signal)'
w = 2; l = -2; t = 0
reward = {'10':0, '11':w, '12':l, '13':t, '20':0, '21':l, '22':w, '23':t} 

####-###-------------------------------------------###
iterations = 1000*20
window_length = 100   # for plotting q update, convolution length