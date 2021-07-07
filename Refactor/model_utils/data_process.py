import re
from itertools import chain

import pandas as pd

# 引入领域词典
from .domain_dict import KEYWORD, STRUCTURE, TERM
from .enums import NumLabels, EngProcessType, LabelToCodeProcessType, LabelToEngProcessType, CodeProcessType
from .re_pattern import NO_CHINESE_PATTERN, MATCH_DICT_CODE_PATTERN, NUMBER_PATTERN, SPLIT_CODE_PATTERN

ENGLISH_PATTERN = r'[\[a-zA-Z\]0-9]+'


class Label:
    """
    定义一个类来提供标签的一些属性，例如类别数量，和预处理方法
    """

    def __init__(self, label_name: str, column_name: str, code_process_type: CodeProcessType = None,
                 eng_process_type: EngProcessType = None):
        self.num_labels = NumLabels[label_name]
        if code_process_type is None:
            self.code_preprocess_type = LabelToCodeProcessType[label_name].value
        else:
            self.code_preprocess_type = code_process_type
        if eng_process_type is None:
            self.eng_preprocess_type = LabelToEngProcessType[label_name].value
        else:
            self.eng_preprocess_type = eng_process_type
        self.name = label_name
        self.column_name = column_name


class EngProcessTools:
    """
    定义一个类来处理英文
    """

    def __init__(self, number_pattern=NUMBER_PATTERN, split_code_pattern=SPLIT_CODE_PATTERN
                 , match_dict_pattern=MATCH_DICT_CODE_PATTERN, dict_list=None):
        """
        初始化函数
       param:
           number_pattern：匹配数字的正则，从re_pattern中引入，用默认值即可
           split_code_pattern：使用切分代码方法时，用于匹配代码段的正则，从re_pattern中引入，用默认值即可
          match_dict_pattern：使用词典匹配法时，用于匹配代码段的正则，从re_pattern中引入，用默认值即可
          dict_list： 使用词典匹配法时，所用的词典，为列表，列表中一个元素为一个词典，使用默认值即可

       return:
           df: 处理后的dataFrame
       """
        if dict_list is None:
            dict_list = [TERM, STRUCTURE, KEYWORD]
        self.number_pattern = number_pattern
        self.split_code_pattern = split_code_pattern
        self.match_dict_pattern = match_dict_pattern
        self.dict_list = dict_list

    def SplitCodeToWordHelper(self, row):
        """
        切分代码，根据驼峰命名规则，将代码进行切分
        """


        #切分代码为英文
        #self.split_code_pattern = r'([a-zA-Z\$][0-9a-z\$]+|[A-Z\$][_A-Z0-9a-z\$]*)\s*'
        text = re.split(self.split_code_pattern, row)

        #去除空白字符
        text = list(filter(lambda p: len(p) != 0, text))
        res = []

        for word in text:
            word = word.lower()
            # 若为英文片段，则保留单词
            # ENGLISH_PATTERN = r'[\[a-zA-Z\]0-9]+'
            if re.fullmatch(ENGLISH_PATTERN, word) is not None:
                word = [word]

            # 若为中文，则将词拆分为字
            else:
                word = [char for char in word]
            res += word

        # 再次去除空白字符
        res = list(filter(lambda p: len(p) != 0, res))
        return res

    # 匹配词典
    def MatchDictHelper(self, row, splited=False):
        """
        匹配词典法，匹配词典中的词，进行映射
        """

        # 判断在匹配词典之前，是否已经用切分代码方法切分过了
        if splited:
            # 若切分过了，则简单做一个替换即可
            # self.dict_list是列表，列表中每个元素为一个领域词典
            for i, dic in enumerate(self.dict_list):
                for j, word in enumerate(row):
                    if word in dic:
                        # 为避在后续处理中重复替换，先将所有词替换为[@@@i]的格式
                        # i为该Token类别，5为数字，4为代码，1-3对应术语、数据结构和关键字
                        row[j] = fr' [@@@{i + 1}] '
            row = "".join(row)
        else:
            # 如果没切分过，还需要用正则切分+替换
            for i, dic in enumerate(self.dict_list):
                for term in dic:
                    row = re.sub('([^a-z])' + term.lower(), fr'\1[@@@{i + 1}] ', row)


        # self.match_dict_pattern = r'[_a-zA-Z][_a-zA-Z0-9().=: ]*'
        # 替换代码为->[unused4]
        row = re.sub(self.match_dict_pattern, '[unused4]', row)

        # 将上文映射为[@@@的字符->[unused
        row = re.sub('\[@@@', '[unused', row)

        # self.number_pattern = r'([^a-z])[0-9]+'
        # 将数字替换->[unused5]
        row = re.sub(self.number_pattern, r'\1[unused5]', row)

        # 将替换Token后的文本转换为数组
        text = re.split('(\[unused.\])', row)
        text = list(filter(lambda p: len(p) != 0, text))
        res = []
        for word in text:
            word = word.lower()
            # 如果没有match到unused，那是英文或标点，按字切分
            if re.match('(\[unused.\])', word) is None:
                word = [char for char in word]
            else:
                #如果match到，那说明是token，全部保留
                word = [word]
            res += word
        res = list(filter(lambda p: len(p) != 0, res))
        return res

    def MatchCodeOnlyHelper(self, row):
        """
        用词典过滤英文单词，匹配剩下的代码
        """
        # print(row)
        all_term = list(chain(*self.dict_list))
        row_list = [item for item in row]
        res = []
        lastEnd = 0

        # 找到文本中所有代码片段
        for m in re.finditer(MATCH_DICT_CODE_PATTERN, row):
            # 将非代码片段加入结果中
            res += row_list[lastEnd:m.start()]

            #lastEnd:m.start()为非代码片段，m.start():m.end()为代码片段
            lastEnd = m.end()

            # 如果找到的代码片段其实是词典中的词
            if m.group(0) in all_term:
                #将词加入词典
                res += f' {m.group(0)} '
                continue

            # 否则就一定是代码，将代码映射为->[unused1]
            res += ' [unused1] '

        # 将剩余部分加入结果中
        if lastEnd != len(row_list):
            res += row_list[lastEnd:]

        #数组转为string，再去空
        text = "".join(res).split(' ')
        text = list(filter(lambda p: len(p) != 0, text))

        # print(text)
        res = []
        for word in text:
            word = word.lower()
            # 如果是英文则保留
            if re.fullmatch(ENGLISH_PATTERN, word) is not None:
                word = [word]
            else:
                #如果是中文则以字为粒度进行切分
                word = [char for char in word]
            res += word
        res = list(filter(lambda p: len(p) != 0, res))

        return res

    def baseProcess(self, row):
        return [char for char in row.lower()]

    def processDataByProcessTypes(self, df: pd.DataFrame, codeProcessType, engProcessType):
        if engProcessType == EngProcessType.NoProcess:
            df = df.apply(self.baseProcess)
            return df, None

        if codeProcessType == CodeProcessType.SplitCodeToWord and engProcessType == EngProcessType.NoProcessAndAddVocab:
            df = df.apply(self.SplitCodeToWordHelper)
            new_vocab = list(chain(*df.tolist()))

            return df, new_vocab

        if codeProcessType == CodeProcessType.SplitCodeToWord and engProcessType == EngProcessType.MatchEngToKey:
            df = df.apply(self.SplitCodeToWordHelper)
            df = df.apply(self.MatchDictHelper, args=(True,))
            return df, None

        if codeProcessType == CodeProcessType.MatchCodeToKey and engProcessType == EngProcessType.MatchEngToKey:
            df = df.apply(self.MatchDictHelper)
            return df, None

        if codeProcessType == CodeProcessType.MatchCodeToKey and engProcessType == EngProcessType.NoProcessAndAddVocab:
            df = df.apply(self.MatchCodeOnlyHelper)
            new_vocab = list(chain(*df.tolist()))
            return df, new_vocab


class DataTools:
    """
    定义一个类来封装对数据的操作
    """

    def readOrgCodeReview(self, org_data, index_col=None, column_name: str = 'data'):
        """
        从原始数据中切分出严重级别、一级分类、二级分类
        param:
            org_data: 源数据的excel文件地址或dataFrame
            to_csv: 存为csv文件，None或地址
            index_col: 是否将某一列作为index读取
            column_name：codeView对应的column名称，默认为data
        return:
            df: 处理后的dataFrame
        """
        if isinstance(org_data, str):
            df = pd.read_excel(org_data, index_col=index_col)
        else:
            df = org_data

        # 依次切分
        df_split = df[column_name].str.split('\[严重级别\]:', expand=True)
        df_cr = df_split[0]
        df_split = df_split[1].str.split('\[一级分类\]:', expand=True)
        df_warn = df_split[0]
        df_split = df_split[1].str.split('\[二级分类\]:', expand=True)
        df_one = df_split[0]
        df_two = df_split[1]

        df = pd.concat([df_cr, df_warn, df_one, df_two], axis=1)
        df.columns = [column_name, '严重级别', '一级分类', '二级分类']

        # 处理一些常见异常
        df[column_name] = df[column_name].str.strip()
        df[column_name] = df[column_name].str.strip('\n >')
        df[column_name] = df[column_name].str.split('严重级别', expand=True)[0]
        df[column_name] = df[column_name].str.split('Done', expand=True)[0]
        df['严重级别'] = df['严重级别'].str.strip()
        df['严重级别'] = df['严重级别'].str.strip('\n >')
        df['一级分类'] = df['一级分类'].str.strip()
        df['一级分类'] = df['一级分类'].str.strip('\n >')
        df['二级分类'] = df['二级分类'].str.strip()
        df['二级分类'] = df['二级分类'].str.strip('\n >')
        df['严重级别'] = df['严重级别'].str.split('\n', expand=True)[0]
        df['严重级别'] = df['严重级别'].str.split(' ', expand=True)[0]
        df['一级分类'] = df['一级分类'].str.split('\n', expand=True)[0]
        df['一级分类'] = df['一级分类'].str.split(' ', expand=True)[0]
        df['二级分类'] = df['二级分类'].str.split('\n', expand=True)[0]
        df['二级分类'] = df['二级分类'].str.split(' ', expand=True)[0]

        df = self.dropNaAndDuplicate(df)

        return df

    @staticmethod
    def dropNa(df: pd.DataFrame):
        """
        去重和空
        param:
            df: dataFrame对象
        return:
            df: 处理后的dataFrame
        """
        if isinstance(df, pd.Series):
            df = df.dropna(how='all')
            return df

        df = df.dropna(axis=1, how='all')
        df = df.dropna(axis=0, how='any')
        return df

    @staticmethod
    def dropNaAndDuplicate(df: pd.DataFrame):
        """
        去重和空
        param:
            df: dataFrame对象
        return:
            df: 处理后的dataFrame
        """
        if isinstance(df, pd.Series):
            df = df.dropna(how='all')
            return df

        df = df.dropna(axis=1, how='all')
        df = df.dropna(axis=0, how='any')
        df.drop_duplicates(inplace=True)
        return df

    @staticmethod
    def sentimentHelper(row):
        """
        对于老版的标准，情感有三个等级（0，1，2）。在新版中，1、2合并为一档，这是用来合并的函数。
        """
        if pd.isna(row):
            return None
        return 1 if row >= 1 else 0

    def applyAndDropNaAndDuplicate(self, df, helper_func, column_name: str = None, args=None):
        """
        param：
            df: dataFrame对象
            helper_func: df.apply要执行的函数，包含了合并的规则
            column_name: 要处理的列名
        return:
            df: 处理后的dataFrame
        """
        if column_name is not None:
            df[column_name] = df[column_name].apply(helper_func)
            df = self.dropNaAndDuplicate(df)
            return df
        df = df.apply(helper_func, args=args, axis=1)
        df = self.dropNaAndDuplicate(df)
        return df

    def applyAndDropNa(self, df, helper_func, column_name: str = None, args=None):
        """
        param：
            df: dataFrame对象
            helper_func: df.apply要执行的函数，包含了合并的规则
            column_name: 要处理的列名
        return:
            df: 处理后的dataFrame
        """
        if column_name is not None:
            df[column_name] = df[column_name].apply(helper_func)
            df = self.dropNa(df)
            return df
        df = df.apply(helper_func, args=args, axis=1)
        df = self.dropNa(df)
        return df

    @staticmethod
    def generateLabelHelper(row, point_problem_name: str = 'CrType', sentiment_name: str = 'Sentiment',
                            share_knowledge_name: str = 'Knowledge'):
        """
        根据三个维度的标签、生成最终的标签的helper函数。
        param:
            point_problem_name：指出问题（评审类型）的列名
            sentiment_name：情感倾向（评审类型）的列名
            share_knowledge_name：传播知识（评审类型）的列名
        """
        point_problem = int(row[point_problem_name])
        sentiment = int(row[sentiment_name])
        share_knowledge = int(row[share_knowledge_name])

        # 无效问题---差，结束
        # 疑问-消极---差，结束
        if point_problem == 0 or (point_problem == 1 and sentiment == 0):
            return 0

        # 疑问-其他--中等，结束
        # 有效-消极--中等，结束
        if (point_problem == 1 and sentiment != 0) or (point_problem == 2 and sentiment == 0):
            return 1

        # 传播知识--优秀
        # 未传播知识-良好
        if share_knowledge == 1:
            return 3

        return 2

    @staticmethod
    def toCsv(df, path, encoding: str = 'utf-8'):
        """
        默认以utf-8的格式生成csv文件
        """
        df.to_csv(path, encoding=encoding)

    @staticmethod
    def shuffleData(df):
        """
        打乱数据
        """
        df = df.sample(frac=1.0)
        return df

    @staticmethod
    def dropEngHelper(row):
        """
        去除全英文的评审意见
        """
        no_chinese_patter = NO_CHINESE_PATTERN
        if isinstance(row, float): return None
        if re.fullmatch(no_chinese_patter, row) is not None:
            return None
        return row

    def generateTestData(self, df):
        """
        留出测试数据
        param:
            df: dataFrame
        return:
            test_df: 10%的测试数据
            df: 剩下90%数据
        """
        df = self.shuffleData(df)
        test_df = df.sample(frac=0.1)
        df = df[~df.index.isin(test_df.index)]
        return test_df, df

    @staticmethod
    def generateTrainAndValDataByKfolder(df, test_frac: float = 0.3, kfolder=3):
        """
        根据标签，生成数据集，K折验证的方法
        param:
            df：dataFrame，须包含codeReview及对应的类别标签
            test_frac: 测试集比例，默认0.3
            kfolder: k折交叉检验，最后返回K组训练/测试集
        return:
            df_test: 测试集
            df_train: 训练集
        """
        # 按类别储存个数

        res = []
        df = df.sample(frac=1.0)
        test_num = int(df.shape[0] * test_frac)
        # print(test_num)
        for i in range(kfolder):
            df_test = df[i * test_num:(i + 1) * test_num]
            df_train = df[~df.index.isin(df_test.index)]
            res.append((df_test, df_train))
        return res

    @staticmethod
    def generateTrainAndValData(label: Label, df, test_frac: float = 0.3, kfolder=3):
        """
        根据标签，生成数据集
        param:
            label：Label类对象，需要生成训练集的类别
            df：dataFrame，须包含codeReview及对应的类别标签
            test_frac: 测试集比例，默认0.3
            kfolder: k折交叉检验，最后返回K组训练/测试集
        return:
            df_test: 测试集
            df_train: 训练集
        """
        # 按类别储存个数

        res = []
        for i in range(kfolder):
            # 按类别切分，保证训练集和数据集分布一致。
            df_train_dict = dict()
            df_test_dict = dict()
            print(label.num_labels.value)
            for j in range(label.num_labels.value):
                tmp = df[df[label.column_name] == j]
                print("tmp",tmp.shape[0])
                test_num = int(test_frac * tmp.shape[0])
                print("test_num",test_num)
                # print(test_num,tmp.shape[0],i*test_num,(i+1)*test_num)
                tmp_test = tmp[i * test_num:(i + 1) * test_num]
                print("tmp_test",tmp_test)
                df_train_dict[j] = tmp[~tmp.index.isin(tmp_test.index)]
                df_test_dict[j] = tmp_test

            df_test = None
            df_train = None

            for k, v in df_train_dict.items():
                if df_train is None:
                    df_train = v
                else:
                    df_train = pd.concat([df_train, v], axis=0)
            for k, v in df_test_dict.items():
                if df_test is None:
                    df_test = v
                else:
                    df_test = pd.concat([df_test, v], axis=0)
            res.append((df_test, df_train))
        return res
