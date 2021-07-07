from allennlp.training.learning_rate_schedulers.learning_rate_scheduler import _PyTorchLearningRateSchedulerWrapper
from allennlp.training.optimizers import Optimizer
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup
from overrides import overrides


class WarmUpScheduler(_PyTorchLearningRateSchedulerWrapper):
    def __init__(
            self, optimizer: Optimizer, num_warmup_steps, num_training_steps
    ) -> None:
        # Create the learning rate scheduler.
        scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=num_warmup_steps,  # Default value in run_glue.py
                                                    num_training_steps=num_training_steps,num_cycles=10)

        super().__init__(scheduler)

    @overrides
    def step_batch(self, batch_num_total: int = None) -> None:
        self.lr_scheduler.step()

    @overrides
    def step(self, metric: float = None, epoch: int = None) -> None:
        pass
