# 匹配词典方法，代码正则
MATCH_DICT_CODE_PATTERN = r'[_a-zA-Z][_a-zA-Z0-9().=: ]*'
# 数字正则
NUMBER_PATTERN = r'([^a-z])[0-9]+'
# 切分代码段方法，代码正则
SPLIT_CODE_PATTERN = r'([a-zA-Z\$][0-9a-z\$]+|[A-Z\$][_A-Z0-9a-z\$]*)\s*'
# 不含中文正则
NO_CHINESE_PATTERN = r'[^\u4e00-\u9fa5]+'
