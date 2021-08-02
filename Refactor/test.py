from model_utils.model import CodeReviewClassifier
from model_utils.data_process import Label
from model_utils.data_reader import HuaWeiDataReader
from model_utils.data_process import EngProcessTools
from model_utils.metrics import HammingLoss,subset_loss
from model_utils.label_pattern import class_to_list,lis_to_class
from allennlp.predictors import TextClassifierPredictor
from tqdm import tqdm
import torch
import collections
import pandas as pd

father_model_path = "\u6a21\u578b/bert-base-chinese-20210429T072130Z-001/bert-base-chinese"
saved_model = "TrainOutput/best.th"

label = Label("UltimateCategory","UltimateCategory")
out_put_res = collections.defaultdict(int)

dataset_reader = HuaWeiDataReader(father_model_path, label=label,
                                  inference=True,
                                  eng_data_tool=EngProcessTools(),
                                  max_tokens=512)
datas = dataset_reader.read("TestData/newtest.csv")
model = CodeReviewClassifier(father_model_path, label=label, out_res=out_put_res)
with open(saved_model, 'rb') as f:
    model.load_state_dict(
        torch.load(f, map_location=torch.device('cpu')))
predictor = TextClassifierPredictor(model, dataset_reader)
for data in tqdm(datas):
    predictor.predict_instance(data)

outdict = model.out_put_res
outlist = []
for i in outdict:
    outlist.append(outdict[i])


outlist = class_to_list(outlist).tolist()
for i in range(len(outlist)):
    if outlist[i][3]==1 and outlist[i][1]==0:
        outlist[i][1]=1
outlist = lis_to_class(outlist).tolist()
df = pd.read_csv("TestData/newtest.csv",encoding='utf-8')
UltimateValue = df["UltimateCategory"].values

print("Hamming:",HammingLoss(outlist,UltimateValue))
print("Subset:",subset_loss(outlist,UltimateValue))
# ac = 0
# for i in range(len(UltimateValue)):
#     if UltimateValue[i] == outlist[i]:
#         ac+=1
# print(ac/199)

