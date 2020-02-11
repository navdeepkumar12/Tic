import numpy as np
import sys
import pickle
number_of_iteration = 10
print('number of iteration = {}'.format(number_of_iteration) )
## CREAATION OF STATE ##
def load_state():
    state = pickle.load(open("/home/navdeep/TicTac/state", "rb"))
    print("load_state:- state loaded")
    return state

def initialize_state(p):
    if p == 0:  #zero initialization of state
        state = {}
        for i in range(3**9):
            temp = np.base_repr(i,base=3)
            temp = "".join(['0'*(9-len(temp)),temp])
            value_temp = np.zeros(9)
            for act, num in enumerate(temp):
                if(num == '0'):
        #             state_action[''.join([temp,str(act)])] = 0
                    value_temp[act] = 0
                else:
                    value_temp[act] = -1
            state[temp] = value_temp
        print('initialize_state:- Zero initialization of state')
    if p ==1 :
        state = load_state()
        return state   

state = initialize_state(1)
state_in_array = {}
for i in state:
    state_in_array[i] = np.array(list(map(int,i)))


finish_check = ['111000000',
               '000111000',
               '000000111',
               '100100100',
               '010010010',
               '001001001',
               '100010001',
               '001010100']
for i, s in enumerate(finish_check):
    finish_check[i] = list(map(int,s))
finish_check = np.array(finish_check,dtype=np.int8)
finish_check = np.array(finish_check,dtype = bool)


frequency = {}
for i in state:
    frequency[i] = np.ones(9)


def result(state_in_string):
    
    for i in finish_check:
        if(np.sum(state_in_array[state_in_string][i] == 1) ==3):
            return 1
        elif(np.sum(state_in_array[state_in_string][i] == 2) == 3):
            return 2
    return 0
    
#     if np.sum(finish_check@state_in_array[state_in_string] ==3) >=1:
#         print('result:- Player 1 won')
#         return 1
#     if np.sum(finish_check@state_in_array[state_in_string] ==6) >+1:
#         print('result:- Player 2 won')
#         return 2
    
#     return 0

def array_to_string(array):
    temp = list(map(str,array))
    return ''.join(temp)    

def transition_sas(current_state_in_string, action, player):
    temp = state_in_array[current_state_in_string]
    if(temp[action] != 0):
        print("transition_sas:- invalid move")
        return
    temp = list(map(str,temp))
    temp[action] = str(player)
#     print(temp)
    return ''.join(temp)

def choose_action(current_state_in_string, k, randomness=0):
    q_value = state[current_state_in_string]
    p_value = np.exp(k*q_value)*(q_value>=0)
    p_value = p_value/np.sum(p_value)
    p_value = p_value + randomness*np.random.rand(9)*(q_value>0)
    return np.argmax(p_value)

# def play(current_state_in_string, k, discount):
    
#     action1 = choose_action(current_state_in_string, k) #choose action from state
#     next_state_in_string1 = transition_sas(current_state_in_string, action1, 1) # transition to next state 
#     #value update1, 
#     if result(next_state_in_string) == 1: #terminal state
#         reward = 1
#         state[current_state_in_string][action] = discount + state[current_state_in_string][action1] 
#         print("play:- Player {} won, game terminate".format(1))    
#         return next_state_in_string


#     action2 = choose_action(current_state_in_string1, k) #choose action from state
#     next_state_in_string2 = transition_sas(current_state_in_string1, action2, 2) # transition to next state 
#     #value update2, 
#     if result(next_state_in_string) == 2: #terminal state
#         reward = 1
#         state[current_state_in_string1][action2] = discount + state[current_state_in_string][action] 
#         print("play:- Player {} won, game terminate".format(1))    
#         return player, next_state_in_string,
#     return 0, next_state_in_string


#     else: #non_terminal state
#         next_action = choose_action(next_state_in_string,k)#this action for only update, not real next action
#         state[current_state_in_string][action] = state[current_state_in_string][action] + discount*state[next_state_in_string][next_action]
#     return    

player_1 = [('000000000',0)]
player_2 = [('000000000',0)]
discount = 0.9
def last_state(player, current_state, current_action):
    if(player == 1):
        temp = player_1.pop()
        player_1.append((current_state,current_action))
        return temp
    elif(player == 2):
        temp = player_2.pop()
        player_2.append((current_state,current_action))
        return temp
def toggle_player(player):
    if(player == 1):
        return 2
    else:
        return 1
    
def play(current_state,player,k):
    print(player,':-',current_state)
    action = choose_action(current_state,k)
    next_state = transition_sas(current_state,action,player)
#     print(next_state)
    frequency[current_state][action] += 1
    
    previous_state, previous_action = last_state(player, current_state, action)
    state[previous_state][previous_action] += 1/frequency[previous_state][previous_action] * \
                                            ( discount * state[current_state][action] - state[previous_state][previous_action] )
    if( result(next_state) == player ):
        state[current_state][action] += 1/frequency[current_state][action] *( 1 - state[current_state][action] )
#         print(state[current_state])
        print("play:- Player {} won, game terminate".format(player))
        return player 
    elif( np.sum(state_in_array[next_state] >0) == 9 ):
        state[current_state][action] += 1/frequency[current_state][action] *( 0.5 - state[current_state][action] )
        state[previous_state][previous_action] += 1/frequency[previous_state][previous_action] *(0.5 - state[previous_state][previous_action] )
        print("tie")
        return 0
    
#     except:
#         print("first chance of player {}".format(player))
    play(next_state, toggle_player(player), k)
    
n = number_of_iteration
c = 0
d = 0
for i in range(n):
    k = np.round(np.log(i+1))/20
    
    print('\n play {}, k = {}, completions ={}, player 1 winning percentage {}, draw {}'.format(i,k,i/n,c/n,d))
    winner = play('000000000',1, k)
    
    if winner ==1:
        c = c+1
    if winner == 0:
        d = d+1



with open('state','wb') as f:
    pickle.dump(state,f)            
