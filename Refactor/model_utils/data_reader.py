import os
import re

import pandas as pd
from transformers import BertTokenizer

from typing import Iterable, List

import numpy as np
from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import ArrayField
from allennlp.data.fields import MetadataField
from transformers import BertTokenizer

from model_utils.data_process import EngProcessTools, Label


class HuaWeiDataReader(DatasetReader):
    def __init__(self,
                 model_path: str,
                 eng_data_tool: EngProcessTools,
                 label: Label,
                 inference: bool = False,
                 max_tokens: int = 512,
                 index_col=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.label = label
        self.index_col = index_col
        self.inference = inference
        self.model_path = model_path
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.max_tokens = max_tokens
        self.eng_data_tool = eng_data_tool
        # if self.label.preprocess_type == ProcessType.SplitCodeToWord:
        #     self.vocab_text = open(os.path.join(self.model_path, 'vocab.txt'), 'r', encoding='utf-8').read().split('\n')
        #     self.code_pattern = SPLIT_CODE_PATTERN
        # elif self.label.preprocess_type == ProcessType.MatchDict:
        #     self.dic_list = [KEYWORD, STRUCTURE, TERM]
        #     self.code_pattern = MATCH_DICT_CODE_PATTERN
        #     self.number_pattern = NUMBER_PATTERN

    def text_to_instance(self, index: int, sent: list, label: int = None, ul_sent: list = None) -> Instance:
        if self.max_tokens is not None:
            # 给[CLS]和[SEP]留个位置
            sent = sent[:self.max_tokens - 2]

        sent = self.wrap_for_bert(sent)
        token_sent = self.tokenizer.convert_tokens_to_ids(sent)

        fields = {'input_tensor': ArrayField(np.array(token_sent), dtype=np.int64),

                  'index': MetadataField(index)}

        if ul_sent is not None:
            if self.max_tokens is not None:
                # 给[CLS]和[SEP]留个位置
                ul_sent = ul_sent[:self.max_tokens - 2]
            ul_sent = self.wrap_for_bert(ul_sent)
            token_ul_sent = self.tokenizer.convert_tokens_to_ids(ul_sent)
            fields.update({'ul_input_tensor': ArrayField(np.array(token_ul_sent), dtype=np.int64)})

        if not self.inference and label is not None:
            fields.update({'label': ArrayField(np.array(label), dtype=np.int64)})

        return Instance(fields)

    # 重写了AllenNLP的_read方法
    def _read(self, filepath) -> Iterable[Instance]:
        """
        给定filepath,返回一个instance
        :param filepath:
        :return: instance
        """
        splited_list = filepath.split()
        if len(splited_list) > 1:
            ul_filepath = splited_list[1]
            filepath = splited_list[0]
        else:
            ul_filepath = None
        print(filepath, ul_filepath)
        iterator = self.huawei_dataset_iterator(filepath)

        if ul_filepath is not None:
            ul_iterator = self.huawei_dataset_iterator(ul_filepath, labeled=False)

            for (index, sent, label), (_, ul_sent, _) in zip(iterator, ul_iterator):
                yield self.text_to_instance(index, sent, label, ul_sent)
        else:
            for index, sent, label in iterator:
                yield self.text_to_instance(index, sent, label)

    def readAllFiles(self, filepath):
        file_list = os.listdir(filepath)
        df = None
        for file in file_list:
            if file.split('.')[-1] == 'csv':
                tmp = pd.read_csv(os.path.join(filepath, file), index_col=0,encoding='gbk')
                if df is None:
                    df = tmp
                else:
                    df = pd.concat([df, tmp], axis=0)
            if file.split('.')[-1] == 'xlsx':
                tmp = pd.read_excel(os.path.join(filepath, file), index_col=0)
                if df is None:
                    df = tmp
                else:
                    df = pd.concat([df, tmp], axis=0)
        return df

    def huawei_dataset_iterator(self, filepath, labeled=True) -> Iterable:
        """
        filepath
        :param filepath:
        :return Iterable
        """
        if len(filepath.split('.'))<=1:
            data = self.readAllFiles(filepath)
        else:
            # test的时候encoding用utf-8，其余时候用gbk
            data = pd.read_csv(filepath, index_col=self.index_col, encoding='gbk')
        data = data[~data['cr'].isna()]
        data['cr'], _ = self.eng_data_tool.processDataByProcessTypes(data['cr'], self.label.code_preprocess_type,
                                                                     self.label.eng_preprocess_type)
        # print(data)
        for i, row in data.iterrows():
            if type(row) == float: continue

            label = None
            if not self.inference and labeled and self.label.column_name is not None:
                col_label_name = self.label.column_name
                label = int(row[col_label_name])
            text = row['cr']
            yield i, text, label

    @staticmethod
    def wrap_for_bert(sent: List[str]):
        sent = ['[CLS]'] + sent + ['[SEP]']
        return sent


