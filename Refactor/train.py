# print(data)

# # data["id"]
import json
import os
import subprocess

from model_utils.data_process import Label, DataTools
from model_utils.enums import LabelToSemiParam
from model_utils.train import train_loop
import pandas as pd
dataTools = DataTools()
def readAllFiles(filepath):
    file_list = os.listdir(filepath)
    df = None
    for file in file_list:
        if file.split(".")[-1] == "csv":
            tmp = pd.read_csv(os.path.join(filepath, file), index_col=0)
            if df is None:
                df = tmp
            else:
                df = pd.concat([df, tmp], axis=0)
        if file.split(".")[-1] == "xlsx":
            tmp = pd.read_excel(os.path.join(filepath, file), index_col=0)
            if df is None:
                df = tmp
            else:
                df = pd.concat([df, tmp], axis=0)
    return df



def generateDataByFunction(dataset_path, label,kfolder):

    if os.path.exists(os.path.join(dataset_path, f"train_{label.name}_0_folder.csv")):
        print(f"训练文件{os.path.join(dataset_path, f'train_{label.name}_0_folder.csv')}已经存在，直接开始训练")
        return

    df = readAllFiles(dataset_path)
    df = dataTools.dropNa(df)
    # print(df)
    df.columns = ["cr", "CrType", "Sentiment", "Knowledge", "UltimateCategory"]
    print(df)
    df = dataTools.applyAndDropNa(df, dataTools.dropEngHelper, column_name="cr")
    # pretrainModelTools = PreTrainModelTools(prerain_model_path)

    # if function == "PreTrain.py":
    #     pretrainModelTools.generatePretrainDataAndModelByProcessTypes(df, "cr",
    #                                                                   label.code_preprocess_type,
    #                                                                   label.eng_preprocess_type,
    #                                                                   output_path)
    #     return

    if label.name == "Quality":
        df["Quality"] = dataTools.applyAndDropNa(df, dataTools.generateLabelHelper)

    kFolderData = dataTools.generateTrainAndValData(label, df, kfolder=kfolder)
    kfolder = os.path.join(dataset_path, "kfolder")
    if not os.path.exists(kfolder):
        os.makedirs(kfolder)
    for i, (test_df, train_df) in enumerate(kFolderData):
        trainFilePath = os.path.join(dataset_path, "kfolder", f"train_{label.name}_{i}_folder.csv")
        testFilePath = os.path.join(dataset_path, "kfolder", f"val_{label.name}_{i}_folder.csv")
        dataTools.toCsv(test_df, testFilePath, encoding='gbk')
        dataTools.toCsv(train_df, trainFilePath, encoding='gbk')


# out_path 输出的目录
# targetFunction 情感倾向 整体质量 评审类型 传授知识
# fatherModelPath  预训练模型
# trainData 文件夹 或者 CSV
# semiData 文件夹 或者 CSV
# kfolder 几折交叉验证
# label  label = Label("Quality","Quality")
def data_K_fold(dataset_path, label, kfold):
    generateDataByFunction(dataset_path, label, kfold)
def prepare_train():
    args = {}
    output_path = "TrainOutput"
    # 如果output_path不存在即创建
    # if not os.path.exists(output_path):
    #     os.makedirs(output_path)
    kfold = 3
    dataset_path = "TrainingData"
    pretrain_model_path = "模型/bert-base-chinese-20210429T072130Z-001/bert-base-chinese"

    label = Label("Sentiment", "Sentiment")

    # 先不考虑半监督学习

    # if args["semiData"] is not None:
    #     semi_dataset_path = args["semiData"]
    #
    #     print("检测到半监督路径", semi_dataset_path)
    #     args.update(LabelToSemiParam[label.name].value)
    #     args.update({"useSemi": True})
    #    # semi_file =
    # else:
    #     args.update({"useSemi": False})
    # print(f"生成数据中，源文件路径:{dataset_path}")

    # 生成K折数据集
    # generateDataByFunction(dataset_path, label, kfold_0)

    print("Done")
    print("训练中")
    for i in range(kfold):  # 3折交叉验证
        train = dataset_path + "/kfolder" + f"/train_{label.name}_{i}_folder.csv"
        # if args["semiData"] is not None:
        #     train = f"{train} {semi_dataset_path}"
        val = dataset_path + "/kfolder" + f"/val_{label.name}_{i}_folder.csv"

        # args.update({"label": label.name,
        #              "model_path": pretrain_model_path,
        #              "train": train,
        #              "val": val,
        #              "serialization_dir": output_path,
        #              "max_tokens": "512"
        #              })
        # cwd = os.getcwd()
        # file = cwd + "/model_utils/train.py"
        # args_str = json.dumps(args)
        # args_str = args_str.replace(" ", "")
        # args_str = args_str.replace("\"", "\\\"")
        # os.system(f"python {file} '{args_str}'")
        # print(f"{json.dumps(args)}")
        # print("请手动执行",f"python {file} '{json.dumps(args)}'")

if __name__ == '__main__':
    args = {"label": Label("UltimateCategory","UltimateCategory"),
            "model_path": "\u6a21\u578b/bert-base-chinese-20210429T072130Z-001/bert-base-chinese",
            "train": "TrainingData/kfolder/val_Sentiment_0_folder.csv",
            "val": "TrainingData/kfolder/val_Sentiment_0_folder.csv",
            "serialization_dir": "TrainOutput",
            "max_tokens": 512,
            "batch_size": 8,
            "lr": 4e-5,
            "cls_lr": 4e-4,
            "g_acc": 1,
            "num_epoch": 5,
            "patience": 20,
            "useSemi": None,
            "kfold_0":3,
            "dataset_path":"TrainingData"
            }
    generateDataByFunction(args["dataset_path"],args["label"],args["kfold_0"])
    for i in range(args["kfold_0"]):
        train = args["dataset_path"] + "/kfolder" + f"/train_{args['label'].name}_{i}_folder.csv"
        val = args["dataset_path"] + "/kfolder" + f"/val_{args['label'].name}_{i}_folder.csv"
        args.update({"label":args["label"].name,
                     "train":train,
                     "val":val,
                     "serialization_dir":"TrainOutput"})
        print(args)
        train_loop(args)