from model_utils.label_pattern import class_to_list
def HammingLoss(pre,gt):
    pre_list = class_to_list(pre).tolist()
    gt_list = class_to_list(gt).tolist()
    D = len(pre_list)
    L = 4
    sum_corr = 0
    for i in range(D):
        lis = pre_list[i]
        lis0 = gt_list[i]
        for j in range(len(lis)):
            if lis[j] == lis0[j]:
                sum_corr+=1
    return sum_corr/(4*D)

def subset_loss(pre,gt):
    sum_corr = 0
    D=len(pre)
    for i in range(len(pre)):
        if pre[i] == gt[i]:
            sum_corr+=1
    return sum_corr/D
