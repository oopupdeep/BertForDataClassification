from torch import tensor
import torch
def class_to_list(c):
    pattern_list = [[1,0,0,0],
                    [0,1,0,0],
                    [0,0,1,0],
                    [0,0,0,1],
                    [1,1,0,0],
                    [1,0,1,0],
                    [1,0,0,1],
                    [0,1,1,0],
                    [0,1,0,1],
                    [0,0,1,1],
                    [1,1,1,0],
                    [1,0,1,1],
                    [1,1,0,1],
                    [0,1,1,1],
                    [1,1,1,1],
                    [0,0,0,0]]
    c_list = c
    if type(c)==torch.Tensor:
        c_list = c.tolist()
    new_p = []
    for i in range(len(c_list)):
        new_p.append(pattern_list[c_list[i]])
    return tensor(new_p,device='cuda:0')
def lis_to_class(p_list):
    pattern_list = [[1,0,0,0],
                    [0,1,0,0],
                    [0,0,1,0],
                    [0,0,0,1],
                    [1,1,0,0],
                    [1,0,1,0],
                    [1,0,0,1],
                    [0,1,1,0],
                    [0,1,0,1],
                    [0,0,1,1],
                    [1,1,1,0],
                    [1,0,1,1],
                    [1,1,0,1],
                    [0,1,1,1],
                    [1,1,1,1],
                    [0,0,0,0]]
    if type(p_list)==torch.Tensor:
        p_list = p_list.tolist()
    new_c = []
    for i in range(len(p_list)):
        if p_list[i] in pattern_list:
            new_c.append(pattern_list.index(p_list[i]))
    return tensor(new_c,device='cuda:0')