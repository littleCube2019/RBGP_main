import numpy as np
def get_fitness_SAD(program , X , y):
   
    y_hat = program.execute(X)
    
    return np.sum([np.abs(y_hat[i] - y[i])  for i in range(len(y))])

def get_fitness_MAD_operon(program , X , y , offset , scale ):


    y_hat = program.execute(X)
    y_hat = offset + scale * y_hat

    #print(program.program , y_hat)
    res = np.mean( np.abs( y_hat - y ) )
    return res 


def get_fitness_MAD(program , X , y ):


    y_hat = program.execute(X)

    #print(program.program , y_hat)
    res = np.mean( np.abs( y_hat - y ) )
    return res 

def get_fitness_MAD_for_list(program , X , y):


    y_hat = program.execute(X)
    
    #print(program.program , y_hat)
    res = np.mean( np.abs( [y_hat[i] - y[i] for i in range(len(y))] ) )
    return res 