
import numpy as np

def cost(y1,y):
    m=len(y)
    squared_cost=np.square(y1-y)
    return (1/(2*m))*(np.sum(squared_cost))






def rmse(pred_y,y):
    print(y)
    squared_cost=np.square(pred_y-y)
    return np.sqrt(np.mean(squared_cost))
    
