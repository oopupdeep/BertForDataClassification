from allennlp.training.metrics import CategoricalAccuracy, F1Measure
import torch

gold_labels = torch.tensor([2, 1])
predictions = torch.tensor([[0.1, 0.5, 0.5, 0.1, 0.1], [0.4, 0.2, 0.1, 0, 0.4]])

accuracy = CategoricalAccuracy()
print(accuracy(predictions, gold_labels).get)