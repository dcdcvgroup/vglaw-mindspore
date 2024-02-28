from mindspore import Tensor
import datetime
import mindspore as ms
import mindspore.ops as ops
import mindspore.nn as nn
import xenv as xenv
from xenv import console
import os
import time
from mindspore.communication import init, get_rank, get_group_size
from mindspore.communication.management import get_group_size, get_rank
from vglaw_final.src.dataset import create_dataset
from model_law import get_network
from mindspore.dataset import vision
from mindspore.dataset.vision import Inter
import datetime
import time
from loss import loss_bbox, loss_xiou, loss_focal, loss_dice
from utils import box_ious_v2, mask_iou
import warnings
from optimizer import MyAdamWeightDecayV2                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              


warnings.filterwarnings("ignore")
args = xenv.parse_args()
os.environ['GLOG_log_dir'] = "./ms_log"
os.environ['GLOG_v'] = '3'


def generate_steps_lr(lr_init, lr_max, step_size, max_epochs, warmup_steps, decay_epochs, gamma=0.1):
    """
    Applies three steps decay to generate learning rate array.

    Args:
        lr_init(float): initial learning rate. m     
        lr_max(float): max learning rate.
        step_size(int): steps that one epoch needs.
        max_epochs(int): max epochs.
        warmup_steps(int): all steps in warmup epochs.
        decay_epochs: a list or array of epochs when the learning rate decays.
        gamma(float): multiplicative factor of learning rate decay. Default:0.1.

    Returns:
        learning rate array.
    """

    if isinstance(decay_epochs, int):
        decay_epochs = [decay_epochs]
    total_steps = max_epochs * step_size
    decay_steps = [step_size * decay_epoch for decay_epoch in decay_epochs]
    decay_steps.append(total_steps)
    num_decay_epochs = len(decay_steps)
    learning_rates = [lr_max * pow(gamma, i) for i in range(num_decay_epochs)]
    lr_each_step = nn.piecewise_constant_lr(milestone=decay_steps, learning_rates=learning_rates)
    for i in range(warmup_steps):
        lr_each_step[i] = lr_init + (lr_max - lr_init) * i / warmup_steps

    return lr_each_step


def init_lr(step_size, whos_lr, max_epochs, warmup_steps, drop_epochs, drop_rate=0.1):
    """
    Initilize learning rate through AdamW.

    Args:
        step_size: steps that one epoch needs.
        whos_lr: who's lr.

    Returns:
        lr list or lr function.
    """

    base_lr = whos_lr
    if args.lr_scheduler == 'step':
        lr_list = generate_steps_lr(0.01 * base_lr, base_lr, step_size, max_epochs, warmup_steps, drop_epochs,
                                    drop_rate)
        return lr_list
    pass

def init_env(args):
    if args.device_target != "None":
        if args.device_target not in ["Ascend", "GPU", "CPU"]:
            raise ValueError(f"Invalid device_target: {args.device_target}, "
                             f"should be in ['None', 'Ascend', 'GPU', 'CPU']")
        ms.set_context(device_target=args.device_target)

    if args.mode_name not in ["GRAPH", "PYNATIVE"]:
        raise ValueError(f"Invalid context_mode: {args.mode_name}, "
                         f"should be in ['graph', 'pynative']")
    context_mode = ms.GRAPH_MODE if args.mode_name == "GRAPH" else ms.PYNATIVE_MODE
    ms.set_context(mode=context_mode)

    args.device_target = ms.get_context("device_target")
    if args.device_target == "CPU":
        args.device_id = 0
        args.device_num = 1
        args.rank_id = 0

    if args.run_distribute:
        print("run distribute!", flush=True)
        if args.device_target == "Ascend":
            device_id = int(os.getenv('DEVICE_ID'))
            ms.set_context(device_id=device_id)
            ms.set_auto_parallel_context(device_num=args.device_num, parallel_mode=ms.ParallelMode.DATA_PARALLEL,
                                         gradients_mean=True)
            ms.parallel.set_algo_parameters(elementwise_op_strategy_follow=True)
            ms.set_auto_parallel_context(all_reduce_fusion_config=args.all_reduce_fusion_config)
            init()
        # GPU target
        else:
            init("nccl")
            ms.reset_auto_parallel_context()
            device_num = get_group_size()
            args.device_num = device_num
            ms.set_auto_parallel_context(device_num=device_num, parallel_mode=ms.ParallelMode.DATA_PARALLEL,
                                         parameter_broadcast=True, gradients_mean=True)
            local_rank = get_rank()
    else:
        local_rank = 0
    ms.set_seed(args.seed + local_rank)


def save_ckpt_train(net, epoch_idx, ckpt_dir, local_rank):
        if local_rank == 0:
            ckpt_path = os.path.join(ckpt_dir, f"epoch-{epoch_idx}.ckpt")
            ms.save_checkpoint(net, ckpt_path)
        else:
            time.sleep(1)
        return 0
      
def save_ckpt(net, epoch_idx, log_dir, local_rank):
    if local_rank == 0:
        ckpt_path = os.path.join(log_dir, f"epoch-{epoch_idx}.ckpt")
        ms.save_checkpoint(net, ckpt_path)
    else:
        time.sleep(1)
    return 0
                           
def write_log(log_str, logFile, local_rank):
    if local_rank == 0:
        with open(logFile, 'a') as f:
            f.write(log_str)
    else:
        time.sleep(1)
    return 0


class Trainer:
    def __init__(self, net, optimizer, train_dataset, args, log_dir,
                 ckpt_dir, logfile, local_rank, device_num, accumulate_step,
                 loss1=loss_bbox(red=args.reduction),
                 loss2=loss_xiou(xiou_loss_type=args.xiou_loss_type),
                 loss3=loss_focal(), loss4=loss_dice(),
                 eval_dataset_list=None, metric=None):
        self.log_dir = log_dir
        self.logFile = logfile
        self.ckpt_dir = ckpt_dir
        self.local_rank = local_rank
        self.device_num = device_num
        self.accumulate_step = accumulate_step
        self.clip_max_norm=args.clip_max_norm
        self.net = net
        self.loss_bbox = loss1
        self.loss_xiou = loss2
        self.loss_focal = loss3
        self.loss_dice = loss4
        self.opt = optimizer
        self.train_dataset = train_dataset
        self.train_data_size = self.train_dataset.get_dataset_size()
        self.weights = self.opt.parameters
        self.value_and_grad = ms.value_and_grad(self.forward_fn, None, weights=self.weights, has_aux=True)

        self.grad_reducer = self.get_grad_reducer()
        self.args = args
        self.run_eval = eval_dataset_list is not None
        if self.run_eval:
            self.eval_loader_list = eval_dataset_list
            self.metric = metric
            self.best_acc = 0
            
        self.count = 0
        
        self.inner_grads = optimizer.parameters.clone(prefix="accumulate_", init="zeros")
        self.zeros = optimizer.parameters.clone(prefix="zeros_", init="zeros")
        self.counter = ms.Parameter(Tensor(0, ms.int32), 'counter_')
        assert accumulate_step > 0
        self.accumulate_step = Tensor(accumulate_step, ms.int32)
        self.map = ops.HyperMap()
    
    # @profile
    def get_grad_reducer(self):
        grad_reducer = ops.identity
        parallel_mode = ms.get_auto_parallel_context("parallel_mode")
        reducer_flag = (parallel_mode != ms.ParallelMode.STAND_ALONE)
        if reducer_flag:
            grad_reducer = nn.DistributedGradReducer(self.weights)
        return grad_reducer
    
    # @profile
    def forward_fn(self, inputs, labels):
        image, text = inputs
        bbox, ref_mask = labels
        pred_box, logit_mask = self.net(image, text)
        loss1 = self.loss_bbox(pred_box, bbox)
        loss2 = self.loss_xiou(pred_box, bbox)
        loss3 = self.loss_focal(logit_mask, ref_mask)
        loss4 = self.loss_dice(logit_mask, ref_mask)

        loss = loss1 + loss2 + loss3 + loss4
        # loss = (self.args.bbox_loss_coef * loss1 +
        #         self.args.xiou_loss_coef * loss2 +
        #         self.args.focal_loss_coef * loss3 +
        #         self.args.dice_loss_coef * loss4)
        return loss, loss1, loss2, loss3, loss4


    def train_single(self, inputs, labels):
        (loss, loss1, loss2, loss3, loss4), grads = self.value_and_grad(inputs, labels)
        self.map(ops.partial(ops.assign_add), self.inner_grads, grads)
        ops.assign_add(self.counter, Tensor(1, ms.int32))
        lr = self.opt.watch_lr()

        if self.counter % self.accumulate_step == 0:
            grads = ops.clip_by_global_norm(self.inner_grads, self.clip_max_norm)
            grads = self.grad_reducer(grads)
            self.map(ops.partial(ops.assign), self.inner_grads, self.zeros)
            self.opt(grads)
        return loss * self.accumulate_step, loss1, loss2, loss3, loss4, lr
    

    def train(self, epochs):
        step_size = self.train_dataset.get_dataset_size()
        lr_list = init_lr(step_size, args.lr_base, args.max_epochs, args.warmup_steps, args.drop_epochs, args.drop_rate)
        lr_base = init_lr(step_size, args.lr_base, args.max_epochs, args.warmup_steps, args.drop_epochs, args.drop_rate)
        lr_visual = init_lr(step_size, args.lr_visual, args.max_epochs, args.warmup_steps, args.drop_epochs, args.drop_rate)
        lr_lang = init_lr(step_size, args.lr_lang, args.max_epochs, args.warmup_steps, args.drop_epochs, args.drop_rate)
        args.lr_len = len(lr_list)
        args.step_size = step_size
        train_dataset = self.train_dataset.create_tuple_iterator(num_epochs=self.args.max_epochs)
        
        precision_best = dict()
        best_epoch = 0
        for split in args.eval_splits:
            precision_best[split] = 0
        for epoch in range(epochs):
            self.net.set_train(True)
            t0 = time.time()
            loss_sum = 0
            loss_bbox_sum = 0
            loss_xiou_sum = 0
            loss_focal_sum = 0
            loss_dice_sum = 0
            for batch, data in enumerate(train_dataset):
                images = data[0]
                bbox = data[1]
                texts = data[2]
                ref_mask = data[4]
                loss, loss1, loss2, loss3, loss4, lr = self.train_single((images, texts), (bbox, ref_mask))
                if self.local_rank == 0:
                    loss_sum += loss
                    loss_bbox_sum += loss1
                    loss_xiou_sum += loss2
                    loss_focal_sum += loss3
                    loss_dice_sum += loss4
                    if batch % 100 == 0:
                        t1 = time.time()
                        loss_mean = round(float(loss_sum / (batch + 1)), 4)
                        loss_bbox_mean = round(float(loss_bbox_sum / (batch + 1)), 4)
                        loss_xiou_mean = round(float(loss_xiou_sum / (batch + 1)), 4)
                        loss_focal_mean = round(float(loss_focal_sum / (batch + 1)), 4)
                        loss_dice_mean = round(float(loss_dice_sum / (batch + 1)), 4)
                        lr = round(float(lr), 8)
                        log_str = f'epoch: {str(epoch).zfill(3)}/{args.max_epochs} |' \
                                f' step: {batch}/{step_size} | loss: {loss_mean} |' \
                                f' loss_bbox: {loss_bbox_mean} | loss_xiou: {loss_xiou_mean} |' \
                                f' loss_focal: {loss_focal_mean} |loss_dice: {loss_dice_mean} |' \
                                f' lr: {lr} | time: {round(t1 - t0)}s\n'
                        print(log_str)
                        write_log(log_str, self.logFile, self.local_rank)
                        t0 = t1
                    if batch % 10000 == 0:
                        save_ckpt_train(self.net, 'latest_in_train_4', self.ckpt_dir, self.local_rank)
                        write_log('Save_checkpoint!\n', self.logFile, self.local_rank)
            if self.run_eval:
                all_reduce = ops.AllReduce()
                self.net.set_train(False)
                split_correct_total = []
                for i in range(len(self.eval_loader_list)):
                    split = args.eval_splits[i]
                    eval_loader = self.eval_loader_list[i]
                    sent_counter = Tensor([0, 0, 0], ms.float32)
                    for step_idx, data in enumerate(eval_loader):
                        image = data[0]
                        bbox = data[1]
                        text = data[2]
                        ref_mask = data[4]
                        pred_box, logit_mask = self.net(image, text)
                        det_iou = box_ious_v2(pred_box[:, :4], bbox, 'iou')['iou']
                        
                        seg_iou, seg_ap= mask_iou(ref_mask, logit_mask.sigmoid()>args.mask_threshold)

                        sent_counter[0] += (det_iou >= args.iou_threshold).sum()
                        sent_counter[1] += seg_iou.sum()
                        sent_counter[2] += image.shape[0]
                    all_reduce(sent_counter)
                    split_correct_total.append((split, sent_counter[0], sent_counter[1], sent_counter[2]))
                precision = dict()
                for split, correct_det, correct_seg, total in split_correct_total:
                    if self.local_rank == 0:
                        print(f'precision/{split}_det: {correct_det/total} | epoch: {epoch}')
                        print(f'precision/{split}_seg: {correct_seg/total} | epoch: {epoch}')
                for split, correct, _, total in split_correct_total:
                    precision[split] = float(correct / total)
                    
                log_str = f"Eval epoch: {epoch} | "
                for split in args.eval_splits:
                    log_str += f"{split}: {round(precision[split], 4)} | "
                log_str += '\n'
                if self.local_rank == 0:
                    print(log_str)
                write_log(log_str, self.logFile, self.local_rank)
                if precision[args.eval_splits[-1]] > precision_best[args.eval_splits[-1]]:
                    precision_best = precision
                    best_epoch = epoch
                    save_ckpt(self.net, "best", self.log_dir, self.local_rank)
                write_log(f"Best epoch: {best_epoch}\n", self.logFile, self.local_rank)
            save_ckpt(self.net, "last", self.log_dir, self.local_rank)
            time.sleep(10)

        log_str = f"Best epoch: {best_epoch} | "
        for split in args.eval_splits:
            log_str += f"Eval split: {split}, precision: {precision_best[split]} | "
        log_str += '\n'
        write_log(log_str, self.logFile, self.local_rank)
                    
                      
def train_net():
    init_env(args)
    if args.run_distribute:
        local_rank = get_rank()
        device_num = get_group_size()
    else:
        local_rank = 0
        device_num = 1
    print('start to create dataset')
    args.dataset_part = 'train'
    train_dataset = create_dataset(args, batch_size=args.batch_size, split=args.train_split, is_train=True,
                                   rank=local_rank, group_size=device_num, shuffle=True)
    print("dataset_len:", train_dataset.get_dataset_size())
    print("args.batch_size=", args.batch_size)
    print("batch size in dataset:", train_dataset.get_batch_size())
    
    eval_dataset_list = create_dataset(args, batch_size=args.batch_size, split=args.eval_splits, is_train=False,
                                       rank=local_rank, group_size=device_num, shuffle=False)
    eval_loader_list = []
    args.eval_step_size = []
    for eval_dataset in eval_dataset_list:
        args.eval_step_size.append(eval_dataset.get_dataset_size())
        eval_loader_list.append(eval_dataset.create_tuple_iterator(num_epochs=1))

    net = get_network(args, num_hidden_layers=6)
    
    param_groups = [
        {'params': [p for n, p in net.parameters_and_names() if n.startswith('visual_encoder') and p.requires_grad],
         'lr': args.lr_visual},
        {'params': [p for n, p in net.parameters_and_names() if n.startswith('text_encoder') and p.requires_grad],
         'lr': args.lr_lang},
        {'params': [p for n, p in net.parameters_and_names() if
                    not n.startswith(('visual_encoder', 'text_encoder')) and p.requires_grad],
         'lr': args.lr_base}
    ]
    optimizer = MyAdamWeightDecayV2(params=param_groups, learning_rate=args.lr_base,
                                    weight_decay=args.weight_decay)
    
    if args.pretrained_path is not None and os.path.exists(args.pretrained_path):
        print(f'Initializing model from pre-trained checkpoint "{os.path.basename(args.pretrained_path)}" ...')
        param_not_load, _ = ms.load_param_into_net(net, ms.load_checkpoint(args.pretrained_path))
    else:
        print("Initializing model from random state ...")
        
    
    log_dir = args.log_dir
    logFile = os.path.join(log_dir, f"rank_{local_rank}.txt")
    ckpt_dir = args.save_ckpt_dir
    if local_rank == 0:
        time_info = str(datetime.datetime.now()) + '\n'
        args_info_list = args._get_kwargs()
        args_info_str = ""
        for arg_tuple in args_info_list:
            args_info_str += f'{arg_tuple[0]}:{arg_tuple[1]}\n'
        with open(logFile, 'a') as f:
            f.write("Start time:")
            f.write(time_info)
            f.write(args_info_str)
            
    accumulate_step = args.batch_sum / args.batch_size
    accumulate_step = accumulate_step / device_num
    
    print(f'accumulate_step:{accumulate_step}')

    trainer = Trainer(net, optimizer, train_dataset, args, log_dir,
                      ckpt_dir, logFile, local_rank, device_num,
                      accumulate_step=accumulate_step,
                      eval_dataset_list=eval_loader_list,)
    trainer.train(args.max_epochs)
    
    
if __name__ == '__main__':
    train_net()