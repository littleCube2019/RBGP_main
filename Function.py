import math
import numpy as np
def addOne(x):
    return x+1

def subOne(x):
    return x-1

def log(x):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(x > 0.001, np.log(x), 0.)

def exp(x):
    return np.where(np.exp(x) > 1E+10, x , np.exp(x))

def sqrt(x):
    return np.sqrt(np.abs(x))
def pow2(x):
    return x*x

def add(x,y):
    
    return x+y
def sub(x,y):
    return x-y
def mul(x,y):
    return x*y
def div(x,y):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(y) > 0.001, np.divide(x, y), 1.)

def _protected_inverse(x1):
    """Closure of inverse for zero arguments."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x1) > 0.001, 1. / x1, 0.)

class Function:
    def __init__(self,func,name , arity):
        self.func = func
        self.name = name
        self.arity = arity
    def __call__(self,*x):
        return self.func(*x)

addOne_ = Function(addOne ,"addOne" ,1)
subOne_ = Function(subOne , "subOne",1)
log_ = Function(log , "log",1)
exp_ = Function(exp , "exp",1)
sqrt_ = Function(sqrt , "sqrt",1)
pow2_ = Function(pow2 , "pow2",1)

add_ = Function(add , "add", 2)
sub_ = Function(sub,  "sub" , 2)
mul_ = Function(mul , "mul", 2)
div_ = Function(div , "div", 2)

sin_ = Function(np.sin , "sin" , 1)
cos_ = Function(np.cos, "cos" , 1)

inverse_ = Function(_protected_inverse,"inv", 1)
neg_ = Function(np.negative, 'neg', 1)

CommonFunction = {
    "addOne" : addOne_ ,
    "subOne" : subOne_ ,
    "log" : log_ ,
    "exp" : exp_ ,
    "sqrt" : sqrt_ ,
    "pow2" : pow2_ ,
    "add" : add_,
    "sub" : sub_,
    "mul" : mul_,
    "div" : div_,
    "sin" : sin_,
    "cos" : cos_,
    "inv": inverse_,
    "neg": neg_,
}