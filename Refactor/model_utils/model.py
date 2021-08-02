from typing import Dict, Any

import torch
from torch import tensor
import torch.nn as nn
import torch.nn.functional as F
from allennlp.models import Model
from allennlp.training.metrics import CategoricalAccuracy, F1Measure
from transformers import BertTokenizer
from model_utils.label_pattern import class_to_list, lis_to_class
from model_utils.BertModel import BertModel
from model_utils.data_process import Label
from model_utils.newCrossEntropy import MutiTask_Cross_Entropy

class CodeReviewClassifier(Model):
    """
    用于训练、验证、预测的监督学习模型，主要以bert为主体
    """

    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def __init__(
            self,
            model_path: str,
            label: Label,
            out_res=None,
    ) -> None:

        super().__init__(vocab=None)
        # 标签的个数
        self.num_labels = label.num_labels.value
        self.out_put_res = out_res

        # 读取预训练模型
        self.bert_model = BertModel.from_pretrained(model_path)
        self.bert_tokenizer = BertTokenizer.from_pretrained(model_path)
        bert_hidden_dim = self.bert_model.config.hidden_size

        # 初始化权重
        self.classifier = self.init_projector(bert_hidden_dim)
        nn.init.xavier_normal_(self.classifier.weight)

        # 定义指标
        self._accuracy = CategoricalAccuracy()
        self._f1 = [F1Measure(i) for i in range(self.num_labels)]

    def init_projector(self, bert_hidden_dim):
        # 线性层
        classifier = torch.nn.Linear(bert_hidden_dim, self.num_labels)
        return classifier

    def forward(
            self,
            input_tensor: torch.Tensor,
            index: torch.Tensor = None,
            label: torch.Tensor = None,
            adv=None,

    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        param:
            input_tensor: 传入的codeReview进过emb后的数据
            index: 评审意见对应的唯一ID
            label: 训练模型下，会有标签
        """
        # mask掉无效字符
        pad_mask = self.get_mask(input_tensor)
        last_hidden, _ = self.bert_model(input_ids=input_tensor, attention_mask=pad_mask, return_dict=False, adv=adv)
        # print(last_hidden)

        # 只用[CLS]来分类
        cls_hidden = last_hidden[:, 0, :]

        logits = self.classifier(cls_hidden)  # [b,3]
        pred = torch.argmax(logits)
        output_dict = {'pred': pred,'logits':logits}

        if label is not None:
            # 计算loss和指标
            # label = class_to_list(label)
            # loss = MutiTask_Cross_Entropy(logits, label)
            # label = lis_to_class(label)
            # print(label)
            loss = F.cross_entropy(logits,label)
            self._accuracy(logits, label)
            for f1_measure in self._f1:
                f1_measure(logits, label)
            output_dict.update({'loss': loss})

        if self.out_put_res is not None:
            # 把inference的结果输出到dataframe中
            self.output_res(index, pred)

        return output_dict

    def output_res(self, index, pred):
        """
        输出结果
        param:
            index： 唯一ID
            pred: 模型预测出的结果
        """
        self.out_put_res[index[0]] = int(pred.detach().cpu().data)

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        """
        输出指标
        """
        acc = self._accuracy.get_metric(reset)
        metrics = {}
        p = 0
        r = 0
        micro_f1 = 0

        # 分类别计算F1score
        for i, f1_measure in enumerate(self._f1):
            f1_score = f1_measure.get_metric(reset)
            p += f1_score['precision']
            r += f1_score['recall']
            metrics.update({f'p_{i}': f1_score['precision']})
            metrics.update({f'r_{i}': f1_score['recall']})
            metrics.update({f"f1_{i}": f1_score['f1']})
            micro_f1 += f1_score['f1']

        # 计算整体F1score和Accuracy
        metrics.update({'P': p / len(self._f1)})
        metrics.update({'R': r / len(self._f1)})
        metrics.update({'F1': micro_f1 / len(self._f1)})
        metrics.update({'acc': acc})
        return metrics

    @staticmethod
    def get_mask(tensor_array: torch.Tensor):
        """
        获取mask，直接计算最后一个不为0的字符的位置即可
        """
        # print(tensor_array)
        mask = tensor_array.new_zeros(tensor_array.shape, dtype=torch.float)
        mask_len = (tensor_array != 0).sum(dim=1)
        for i, row in enumerate(mask):
            row[:mask_len[i]] = 1
        return mask
