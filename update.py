import numpy as np
def rule1(pred_y,Y_train,X_train):
    m=len(Y_train)
    a=pred_y-Y_train
    return (1/m)*(np.sum(a*X_train))


def rule2(pred_y,Y_train):
    m=len(Y_train)
    a=pred_y-Y_train
    return (1/m)*(np.sum(a))