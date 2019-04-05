'''
Cross-Entropy Error (cost) function
J = - SUM{ ( t*log(y) + (1-t)*log(1-y) ) } , where t = target, y = output of logistic
ex:
t = 1, y = 1 -> J = 0
t = 0, y = 0 -> J = 0
t = 1, y = 0,9 -> 0,11
t = 1, y = 0,1 -> 2,3
'''
# calculate the cross-entropy error
def cross_entropy(T, Y):
    E = 0
    for i in range(len(T)):
        if T[i] == 1:
            E -= np.log(Y[i])
        else:
            E -= np.log(1 - Y[i])
    return E
