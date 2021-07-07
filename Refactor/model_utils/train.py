import collections
import getopt
import json
import os
from collections import defaultdict
import sys

import torch
from allennlp.data import allennlp_collate
from torch.utils.data import DataLoader
from transformers import AdamW
from allennlp.training import GradientDescentTrainer

sys.path.append(os.getcwd())
from model_utils.data_process import EngProcessTools, Label
from model_utils.data_reader import HuaWeiDataReader
from model_utils.model import CodeReviewClassifier
from model_utils.warmup_scheduler import WarmUpScheduler
from model_utils.Trainer import GradientDescentTrainerForVAT


def train_loop(args):
    # args['num_epoch']=1
    args['label'] = Label(args['label'],args['label'])
    torch.cuda.empty_cache()

    dataset_reader = HuaWeiDataReader(args['model_path'], max_tokens=args['max_tokens'], label=args['label'],
                                      eng_data_tool=EngProcessTools())

    # train = os.path.join(data_root,'wo-pre-train.csv')
    # val = os.path.join(data_root,'wo-pre-test.csv')
    train_data = dataset_reader.read(args['train'])
    # print("train_data",train_data)
    val_data = dataset_reader.read(args['val'])

    # collate_fn：告诉dataloader是如何取样本的
    train_loader = DataLoader(train_data, batch_size=args['batch_size'], collate_fn=allennlp_collate, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args['batch_size'], collate_fn=allennlp_collate)
    out_put_res = collections.defaultdict(int)

    model = CodeReviewClassifier(args['model_path'], label=args['label'], out_res=out_put_res)
    cls_params = list(map(id, model.classifier.parameters()))
    params_dict = defaultdict(list)

    for k, v in model.named_parameters():
        name = k.split('.')
        if name[0] != 'bert_model':
            continue
        if name[1] == 'embeddings':
            params_dict[0].append(v)
            continue
        if name[1] == 'pooler':
            params_dict[11].append(v)
            continue
        if name[3] in [str(i) for i in range(12)]:
            params_dict[int(name[3])].append(v)
            continue
        params_dict[11].append(v)

    lr = args['lr']

    params = []
    for k, v in params_dict.items():
        params.append({'params': v, 'lr': lr})
    params.append({'params': model.classifier.parameters(), 'lr': args['cls_lr']})

    # base_params = filter(lambda p: id(p) not in cls_params,
    #                      model.parameters())
    # params = [{'params': base_params}, {'params': model.classifier.parameters(), 'lr': args['cls_lr']}]

    optimizer = AdamW(params,
                      lr=args['lr'],  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                      eps=1e-8,  # args.adam_epsilon  - default is 1e-8.
                      correct_bias=True
                      )

    count_instance = len(train_data)
    total_steps = count_instance / args['batch_size'] / args['g_acc'] * args[
        'num_epoch']  # len(train_set) nearly equal to
    num_warmup_steps = total_steps * 0.1

    print(f"total steps: {total_steps}\n"
          f"warmup steps: {num_warmup_steps}\n")

    scheduler = WarmUpScheduler(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps)

    if not os.path.exists(args['serialization_dir']):
        os.makedirs(args['serialization_dir'])

    if torch.cuda.is_available():
        cuda_device = 0
        model = model.cuda(cuda_device)
    else:
        cuda_device = -1

    if args['useSemi']:
        trainer = GradientDescentTrainerForVAT(model=model,
                                               optimizer=optimizer,
                                               data_loader=train_loader,
                                               validation_data_loader=val_loader,
                                               patience=args['patience'],
                                               validation_metric='+F1',
                                               num_epochs=args['num_epoch'],
                                               serialization_dir=args['serialization_dir'],
                                               cuda_device=cuda_device,
                                               learning_rate_scheduler=scheduler,
                                               grad_clipping=1.0,
                                               num_gradient_accumulation_steps=args['g_acc'],
                                               xi=args['xi'],
                                               eps=args['eps'],
                                               delay_epoch=args['delay_epoch'] - 1
                                               )
    else:
        trainer = GradientDescentTrainer(model=model,
                                         optimizer=optimizer,
                                         data_loader=train_loader,
                                         validation_data_loader=val_loader,
                                         patience=args['patience'],  # 可选。在等待patience个epochs之后，如果模型效果没有提升，则终止；如果不选，则不提前终止。
                                         validation_metric='+F1',
                                         num_epochs=args['num_epoch'],
                                         serialization_dir=args['serialization_dir'],
                                         cuda_device=cuda_device,
                                         learning_rate_scheduler=scheduler,
                                         grad_clipping=1.0, # 梯度不大于这个值
                                         num_gradient_accumulation_steps=args['g_acc'], # 在执行优化器步骤之前，针对给定数量的步骤累积梯度。这对于容纳大于 RAM 大小的批次很有用。
                                         )
    metrics = trainer.train()
    metrics_root = args['serialization_dir'] + '/metrics'
    
    if not os.path.exists(metrics_root):
        os.makedirs(metrics_root)
    with open(metrics_root + f'/metrics_{args["label"].name}.txt', 'w') as f:
        print(json.dumps(metrics), file=f)
    # return metrics['best_validation_F1']


# def main(argv):
#     # print("argv[1]",argv[1])
#     # args = json.loads("{\"label\":\"Sentiment\",\"model_path\":\"\u6a21\u578b/bert-base-chinese-20210429T072130Z-001/bert-base-chinese\",\"train\":\"TrainingData/kfolder/train_Sentiment_0_folder.csv\",\"val\":\"TrainingData/kfolder/val_Sentiment_0_folder.csv\",\"serialization_dir\":\"TrainOutput\",\"max_tokens\":512,\"batch_size\":10,\"lr\":4e-5,\"cls_lr\":5e-5,\"g_acc\":1,}")
#     args = {"label":"UltimateCategory",
#             "model_path":"\u6a21\u578b/bert-base-chinese-20210429T072130Z-001/bert-base-chinese",
#             "train":"TrainingData/kfolder/val_Sentiment_0_folder.csv",
#             "val":"TrainingData/kfolder/val_Sentiment_0_folder.csv",
#             "serialization_dir":"TrainOutput",
#             "max_tokens":512,
#             "batch_size":8,
#             "lr":4e-5,
#             "cls_lr":4e-4,
#             "g_acc":1,
#             "num_epoch":5,
#             "patience":20,
#             "useSemi":None
#             }
#     train_loop(args)


# if __name__ == '__main__':
#     # print(os.getcwd())
#
#     print('参数个数为:', len(sys.argv), '个参数。')
#     print(sys.argv)
#     main(sys.argv)

    # 这里切换不同的label来跑结果
#
# quality = Label("Quality", 'label')
# sentiment = Label("Sentiment", 'sentiment')
# crType = Label("CrType", 'findP')
# shareKnowledge = Label("ShareKnowledge", 'knowledge')
# labels = [crType,shareKnowledge]
# # labels = [shareKnowledge]
#
# best_score = collections.defaultdict(float)
# avg_score = collections.defaultdict(float)
#
#
# eps_list = [8.0,10.0,4.0,2.0]
# xi = 1e-6
# data_root = '/content/drive/MyDrive/CRJ/data/kfolder'
# delay_epoch_list=[10,5]
#
# for label in labels:# 4个
#   for delay_epoch in delay_epoch_list:
#     for eps in eps_list:
#       for i in range(3):# 3折交叉验证
#             model_path = f'tmp/pretrained/pretrain_code_{label.code_preprocess_type.value}_eng_{label.eng_preprocess_type.value}'
#             serialization_dir =f'tmp/model_save/task_{label.name}_eps_{eps}_xi_{xi}'
#             # !rm -rf $serialization_dir
#             l_file = data_root+f'/train_{label.name}_{i}_folder.csv'
#             ul_file = os.path.join(args['data_root'], f'unlabel.csv')
#             train = f'{l_file} {ul_file}'
#             val = data_root+f'/val_{label.name}_{i}_folder.csv'
#             args.update({'label': label,
#                         'model_path': model_path,
#                         "train": train,
#                         "val": val,
#                         "serialization_dir": serialization_dir,
#                           'xi':xi,
#                         'delay_epoch':delay_epoch,
#                           'eps':eps})
#             fscore = train_loop(args)
#
#             best_score[f'{label.name}_{i}_folder'] = fscore
#             best_score[f'{label.name}'] += fscore
#
#
#             ## 拷贝到云盘
#             # local_model_path = serialization_dir + '/best.th'
#             # cloud_root = f'/content/drive/MyDrive/Exp3/model_save'
#             # if not os.path.exists(cloud_root):
#             #   os.makedirs(cloud_root)
#             # cloud_model_path = cloud_root + f'/task_{label.name}_eps_{eps}_xi_{xi}.th'
#             # !cp  $local_model_path $cloud_model_path
#             # !rm -rf $serialization_dir
#             # avg_score[f'{label.name}_eps_{eps}_xi_{xi}']+=fscore
#             with open(f'/content/drive/MyDrive/Exp3/best_metrics_{label.name}_eps_{eps}_xi_{xi}_delay_{delay_epoch}.txt', 'w') as f:
#                   print(json.dumps(best_score),file=f)
#             # with open(f'/content/drive/MyDrive/Exp3/avg_metrics_{label.name}.txt', 'w') as f:
#             #       print(json.dumps(avg_score), file=f)
#
#       best_score[f'{label.name}'] /=3
#       with open(f'/content/drive/MyDrive/Exp3/best_metrics_{label.name}_eps_{eps}_xi_{xi}_delay_{delay_epoch}.txt', 'w') as f:
#                 print(json.dumps(best_score),file=f)
#
