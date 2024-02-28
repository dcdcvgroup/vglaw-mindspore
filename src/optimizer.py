import mindspore as ms
from mindspore.nn.optim.optimizer import Optimizer
from mindspore.nn.optim.adam import AdamWeightDecay
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.common.api import jit
from mindspore import Tensor
_adam_opt = C.MultitypeFuncGraph("adam_opt")
_fused_adam_weight_decay = C.MultitypeFuncGraph("fused_adam_weight_decay")

class MyAdamWeightDecay(AdamWeightDecay):
    _support_parallel_optimizer = True
    def __init__(self, params, learning_rate=1e-3, beta1=0.9, beta2=0.999, eps=1e-6, weight_decay=0.0,drop_epoch=90,drop_rate=0.1, warmup_epoch=1, warmup_rate = 0.1):
        super().__init__(params, learning_rate, beta1, beta2, eps, weight_decay)
        self.drop_epoch = drop_epoch
        self.drop_rate = drop_rate
        self.warmup_epoch = warmup_epoch
        self.warmup_rate = warmup_rate

    def get_group_lr(self, rate):
        lr = ()
        for learning_rate in self.learning_rate:
            test_lr = learning_rate.reshape(())
            current_lr = (learning_rate * rate).reshape(())
            lr += (current_lr,)
        return lr

    def get_lr(self, cur_epoch):
        lr = self.learning_rate

        # warm up
        if cur_epoch < self.warmup_epoch:
            if self.is_group_lr:
                lr = self.get_group_lr(self.warmup_rate)
        elif cur_epoch < self.drop_epoch: # normal
            if self.is_group_lr:
                lr=()
                for learning_rate in self.learning_rate:
                    current_lr = learning_rate.reshape(())
                    lr+=(current_lr,)
            else:
                lr = self.learning_rate.reshape(())
        else:   # drop
            if self.is_group_lr:
                lr = self.get_group_lr(self.drop_rate)
            else:
                lr = (self.learning_rate * self.drop_rate).reshape(())
        return lr

    def construct(self, gradients, cur_epoch):
        gradients = self.flatten_gradients(gradients)
        weight_decay = self.get_weight_decay()
        lr = self.get_lr(cur_epoch)

        if self.use_fused_opt:
            if self.is_group:
                if self.is_group_lr:
                    optim_result = self.hyper_map(
                        F.partial(_fused_adam_weight_decay, self.fused_opt, self.beta1, self.beta2, self.eps),
                        lr, weight_decay, self._parameters, self.moments1,
                        self.moments2, gradients, self.decay_flags, self.optim_filter)
                else:
                    optim_result = self.hyper_map(
                        F.partial(_fused_adam_weight_decay, self.fused_opt, self.beta1, self.beta2, self.eps, lr),
                        weight_decay, self._parameters, self.moments1, self.moments2,
                        gradients, self.decay_flags, self.optim_filter)
            else:
                optim_result = self.hyper_map(
                    F.partial(_fused_adam_weight_decay, self.fused_opt, self.beta1, self.beta2, self.eps, lr,
                              weight_decay),
                    self._parameters, self.moments1, self.moments2,
                    gradients, self.decay_flags, self.optim_filter)
        else:
            if self.is_group:
                if self.is_group_lr:
                    optim_result = self.hyper_map(F.partial(_adam_opt, self.beta1, self.beta2, self.eps),
                                                  lr, weight_decay, self._parameters, self.moments1,
                                                  self.moments2, gradients, self.decay_flags, self.optim_filter)
                else:
                    optim_result = self.hyper_map(F.partial(_adam_opt, self.beta1, self.beta2, self.eps, lr),
                                                  weight_decay, self._parameters, self.moments1, self.moments2,
                                                  gradients, self.decay_flags, self.optim_filter)
            else:
                optim_result = self.hyper_map(F.partial(_adam_opt, self.beta1, self.beta2, self.eps, lr, weight_decay),
                                              self._parameters, self.moments1, self.moments2,
                                              gradients, self.decay_flags, self.optim_filter)
        if self.use_parallel:
            self.broadcast_params(optim_result)

        return optim_result
    
class MyAdamWeightDecayV2(AdamWeightDecay):
    _support_parallel_optimizer = True
    def __init__(self, params, learning_rate=1e-3, beta1=0.9, beta2=0.999, eps=1e-6, weight_decay=0.0, accumulate_step=1):
        super().__init__(params, learning_rate, beta1, beta2, eps, weight_decay)
        self.global_step_increase_tensor = Tensor(accumulate_step, ms.int32)
    def watch_lr(self):
        lr = self.learning_rate
        if self.dynamic_lr:
            if self.is_group_lr:
                lr = ()
                for learning_rate in self.learning_rate:
                    current_dynamic_lr = learning_rate(self.global_step).reshape(())
                    lr += (current_dynamic_lr,)
            else:
                lr = self.learning_rate(self.global_step).reshape(())
        return lr[-1]