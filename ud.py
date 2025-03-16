import pyunidoe
import numpy as np

def generate_ud(n, s, block, mean=0, sigma=1, crit='CD2', random_state = 777):
    glp = pyunidoe.gen_ud(n, s, n, crit=crit, random_state=random_state)['final_design']
    glp = normalize(glp, n)
    res = shift(glp, block)
    return  mean + (res - 0.5)* sigma * 2

def normalize(design, q):
    return  design / q

def shift(design, block):
    n = design.shape[0]
    s = np.random.random((block, 1, 1))
    res = design[np.newaxis, :, :] * s
    final_matrix = res.transpose(1, 0, 2).reshape(n, -1)
    return final_matrix














print('Have a nice day!')