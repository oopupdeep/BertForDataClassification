from enum import Enum, unique


class NumLabels(Enum):
    """
    不同标签有不同数量的类别
    """
    # def __init__(self, ):
    Quality = 4  # 最终标签
    Sentiment = 2  # 情感倾向
    CrType = 3  # 评审类别
    ShareKnowledge = 2  # 传播知识
    UltimateCategory = 8 #终极标签



@unique
class CodeProcessType(Enum):
    """
    不同的预处理方法及对应的预处理模型
    """
    SplitCodeToWord = 'split'  # 切分代码片段
    MatchCodeToKey = 'match'  # 匹配代码为特殊字符
    NoProcess = 'none'


class LabelToSemiParam(Enum):
    Quality = {'eps': 8.0, 'xi': 1e-6, "delay_epoch": 10}
    Sentiment = {'eps': 8.0, 'xi': 1e-6, "delay_epoch": 5}
    ShareKnowledge = {'eps': 10.0, 'xi': 1e-6, "delay_epoch": 5}
    CrType = {'eps': 2.0, 'xi': 1e-6, "delay_epoch": 5}


@unique
class EngProcessType(Enum):
    """
    不同的预处理方法及对应的预处理模型
    """
    MatchEngToKey = 'match'  # 用词典匹配英文为特殊字符
    NoProcessAndAddVocab = 'add'  # 不做额外处理并扩充词典
    NoProcess = 'none'


class LabelToEngProcessType(Enum):
    """
    不同标签需要使用不同的预处理方法及对应的预处理模型
    """
    Quality = EngProcessType.MatchEngToKey
    Sentiment = EngProcessType.NoProcess
    CrType = EngProcessType.NoProcessAndAddVocab
    ShareKnowledge = EngProcessType.NoProcessAndAddVocab
    UltimateCategory = EngProcessType.NoProcessAndAddVocab


class LabelToCodeProcessType(Enum):
    """
    不同标签需要使用不同的预处理方法及对应的预处理模型
    """
    Quality = CodeProcessType.SplitCodeToWord
    Sentiment = CodeProcessType.SplitCodeToWord
    CrType = CodeProcessType.SplitCodeToWord
    ShareKnowledge = CodeProcessType.SplitCodeToWord
    UltimateCategory = CodeProcessType.SplitCodeToWord
