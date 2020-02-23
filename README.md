# Tic

#TicTacToe
This trains agent  to play TicTac via reinforcement learnig. 

To Train:-  Run main.py.  It will output Q<index> file which contains Q value function, and other plots and file. One set of output file indexed with 129 is provided.

To change parameters for the training :- edit parameters values in pm.py

To play with trained agent interactively:- 1) Run play.py, 2)enter 'Q<>' with which you want to play. ex Q10, Q129 (see
     which file was created.   3) Enter yes if want to play first , no if you want agent to play first.
     You must have opencv installed for graphics to run.



#ALPHA GO 

nn.py has neural network library, like linear, relu, softmax, convolution, cre, mse forward and backpropagation.

Backprop for convolution:-  input x,output y, filter w, inward grad dy, outward grad dx, mode {'full', 'valid', same','custom'}. Algorithm is below.

X = pad(x, required for mode)

y = correlate(X,w, 'valid')

dX = convolve(dy, w, 'full')

dw = correlate(X,dy, 'valid')

dx = uppad(dX, same as padding)

               #Opitimizer ADAM
See original paper ADAM: A METHOD  FORSTOCHASTICOPTIMIZATION for excellent reference.
https://arxiv.org/pdf/1412.6980.pdf               
