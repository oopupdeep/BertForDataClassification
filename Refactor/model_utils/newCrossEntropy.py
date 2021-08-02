import torch.nn.functional as F
from torch import tensor

def MutiTask_Cross_Entropy(logits, label):
    sum_loss = tensor(0.0,device='cuda:0')
    label_list = label.tolist()
    row_num = label.shape[0]
    col_num = label.shape[1]
    for i in range(col_num):
        s_list = []
        for j in range(row_num):
            s_list.append(label_list[j][i])
        sum_loss += F.cross_entropy(logits, tensor(s_list,device='cuda:0'))
    return sum_loss

# logits = tensor([[0.13,0.23], [0.24,0.24], [0.25,0.25], [0.26,0.25], [0.35,0.56], [0.56,0.78], [0.23,0.90], [0.25,0.23],[0.13,0.23], [0.24,0.24], [0.25,0.25], [0.26,0.25], [0.35,0.56], [0.56,0.78], [0.23,0.90], [0.25,0.23]])
# label = tensor([[1,0,0,0],
#                 [0,1,0,0],
#                 [0,0,1,0],
#                 [0,0,0,1],
#                 [1,1,0,0],
#                 [1,0,1,0],
#                 [1,0,0,1],
#                 [0,1,1,0],
#                 [0,1,0,1],
#                 [0,0,1,1],
#                 [1,1,1,0],
#                 [1,0,1,1],
#                 [1,1,0,1],
#                 [0,1,1,1],
#                 [1,1,1,1],
#                 [0,0,0,0]])
# sum_loss = MutiTask_Cross_Entropy(logits,label)
# print(sum_loss)
