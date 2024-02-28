import os
import sys
import logging
import random
import vglaw_final.src.path_info as path_info
from datetime import datetime
from rich.console import Console
from argparse import ONE_OR_MORE, ArgumentParser

sys.path.append(path_info.extra_lib_path)
try:
    import hfai

    get_whole_life_state = hfai.client.get_whole_life_state
    set_whole_life_state = hfai.client.set_whole_life_state
except:
    get_whole_life_state = lambda: 0
    set_whole_life_state = lambda x: None

# config pytorch cache
# os.environ['TORCH_HOME'] = ms_path_info.torch_cache

# config huggingface transformers cache
# os.environ['TRANSFORMERS_CACHE'] = ms_path_info.transformers_cache
# from transformers.modeling_utils import logger

# logger.setLevel(logging.ERROR)

################## Config color console output for all modules ##################
console = Console(highlight=False)
seed = random.randint(1, 65535)

def parse_args(args=None):
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=seed, help='random number generator seed to use')
    parser.add_argument('--device_target', type=str, default='GPU', help='the type of hardware')
    # distributed data parallel
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='node local rank for distributed training, -1 no distributed')
    # dataset config
    parser.add_argument('--image_root', type=str, default='images', help='root path of the images')
    parser.add_argument('--config_root', type=str, default='data', help='root path of the annotation file')
    parser.add_argument('--mask_path', type=str, default='masks', help='mask path')
    parser.add_argument('--dataset', type=str, default='refclef', help='refcoco/refcoco+/refcocog/refclef/merge')
    parser.add_argument('--splitBy', type=str, default=None, help='splits this dataset: unc/google/umd/berkeley')
    parser.add_argument('--img_size', type=int, default=512, help='image size')
    parser.add_argument('--max_len', type=int, default=40, help='max length of sentence tokens')
    parser.add_argument('--crop_prob', type=float, default=0.5, help="probability of random crop augmentation")
    parser.add_argument('--reduction', type=str, default='sum', help="dataset config reduction")
    # parser.add_argument('--blur_prob', type=float, default=0.5, help="probability of gaussian blur augmentation")
    parser.add_argument('--translate', action='store_true', help="use random translate augmentation")
    parser.add_argument('--multi_scale', action='store_true', help="use multi-scale augmentation")
    # training config
    parser.add_argument('--run_distribute', type=bool, default=True, help='whether to run distribute')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size used for training and validation')
    parser.add_argument('--batch_sum', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4, help='worker numbers used for dataloader')
    parser.add_argument('--resume', action='store_true', help='resume from the latest checkpoint')
    parser.add_argument('--start_epoch', type=int, default=0, help='start epoch of training process, >0 will resume')
    parser.add_argument('--train_split', type=str, default='train', help='dataset train split')
    parser.add_argument('--eval_splits', type=str, nargs=ONE_OR_MORE, default=('val',), help='dataset eval splits')
    parser.add_argument('--max_epochs', type=int, default=120, help='max epoch of training process')
    parser.add_argument('--warmup_steps', type=int, default=8000, help='noam warmup steps')  # 8000
    parser.add_argument('--clip_max_norm', type=float, default=0.15,
                        help='gradient clipping max norm, <=0 indicates not use gradient clipping')
    parser.add_argument('--xiou_loss_type', type=str, default='giou', choices=('iou', 'giou', 'diou', 'ciou'),
                        help='iou type used for iou_loss')
    parser.add_argument('--xiou_loss_coef', type=float, default=1.0, help='coefficient of iou loss(giou, diou, ciou)')
    parser.add_argument('--bbox_loss_coef', type=float, default=1.0,
                        help='coefficient of bbox loss(L1 Loss / Smooth-L1 Loss)')
    parser.add_argument('--arch_loss_coef', type=float, default=1e-3, help='coefficient of select/architecture loss')
    parser.add_argument('--dice_loss_coef', type=float, default=1.0)
    parser.add_argument('--focal_loss_coef', type=float, default=1.0)
    parser.add_argument('--lr_scheduler', type=str, default='noam', choices=('noam', 'poly', 'step', 'cosine'),
                        help='lr_scheduler type')
    parser.add_argument('--lr_base', type=float, default=1e-4,
                        help='the max learning rate in noam learning rate scheduler')
    parser.add_argument('--lr_lang', type=float, default=1e-5, help='the max learning rate of the language encoder')
    parser.add_argument('--lr_visual', type=float, default=1e-5, help='the max learning rate of the visual encoder')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--poly_power', type=float, default=0.9, help='power of poly lr scheduler')
    parser.add_argument('--noam_gamma', type=float, default=-0.5, help='gamma used in noam_warmup')
    parser.add_argument('--drop_epochs', type=int, nargs=ONE_OR_MORE, default=(50,),
                        help='drop epochs in multi-step scheduler')
    parser.add_argument('--trainable_layers', type=int, default=3, help='#trainable_layers of visual backbone(1 - 5)')
    parser.add_argument('--pretrained_path', type=str, default=None, help='path to the pre-trained model checkpoints')
    # validation config
    parser.add_argument('--iou_threshold', type=float, default=0.5, help='iou threshold used for validation')
    parser.add_argument('--eval_step', type=int, default=1, help='evaluation triggered every eval_step epochs')
    # network config
    parser.add_argument('--parameter_server', type=bool, default=False, help="whether to use server")
    parser.add_argument('--target_rate', type=float, default=1.0, help='target rate')
    parser.add_argument('--relu_gate', action='store_true', help='apply ReLU to the gates')
    parser.add_argument('--no_abs', action='store_true', help='apply ReLU to the loss')
    parser.add_argument('--pre_norm', action='store_true', help='normalize before in Transformer encoder')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout probablity in the network')  # 0.1
    parser.add_argument('--text_encoder_layer', type=int, default=6, help='layer numbers of lang encoder')
    parser.add_argument('--task_encoder_layer', type=int, default=6, help='layer numbers of task encoder')
    parser.add_argument('--task_encoder_dim', type=int, default=256, help='dimensions of task encoder')
    parser.add_argument('--task_ffn_dim', type=int, default=2048,
                        help='intermediate dimensions of task feedforward network')
    parser.add_argument('--task_encoder_head', type=int, default=8, help='head numbers of task encoder')
    parser.add_argument('--use_selector', action='store_true', help='use dynamic subnet selector')
    parser.add_argument('--use_dilation', action='store_true', help='use dilated backbone')
    parser.add_argument('--token_select', type=str, default='all', choices=('all', 'rand', 'greedy'),
                        help='token select strategy')
    parser.add_argument('--cnn_model', type=str, default='resnet50-coco', choices=(
    'resnet50-coco', 'resnet50-imagenet', 'resnet101-imagenet', 'resnet50-unc', 'resnet101-unc', 'resnet50-gref',
    'resnet101-gref', 'resnet50-referit', 'resnet101-referit'), help='visual backbone')
    parser.add_argument('--bert_model', type=str, default='bert-base-uncased',
                        choices=('bert-base-uncased', 'roberta-base'), help='bert model')
    parser.add_argument('--vit_model', type=str, help='vit model')
    parser.add_argument('--vit_model_path', type=str, help='vit model path')
    parser.add_argument('--law_type', type=str, default=None, help='law_type, choose in [la, od1, od2, od3]')
    parser.add_argument('--groups', type=int, nargs=ONE_OR_MORE, default=None, help='group number of each qkv layer')
    parser.add_argument('--use_activation', action='store_true', help='apply activation to the dynamic weights')
    parser.add_argument('--use_mask', action='store_true', help='predict masks!')
    parser.add_argument('--mask_threshold', type=float, default=0.35, help='mask_threshold')
    # checkpoint/log config
    parser.add_argument('--work_dir', type=str, default='work_dir', help='work_dir of the current experiment')
    parser.add_argument('--save_ckpt_dir', type=str, default='save_ckpt_dir', help='save_ckpt_dir of the current experiment')
    parser.add_argument('--log_freq', type=int, default=10,
                        help='the frequency of print log information during training')
    parser.add_argument('--experiment_name', type=str, default='experiment0', help='name of the current experiment')
    parser.add_argument('--comments', type=str, default='no comment', help='some extra comments')
    parser.add_argument('--short_comment', type=str, default='', help='short comment appended to the log filename')
    # final config
    parser.add_argument('--model_share_type', type=str, default='sh', choices=('sh', 'id'), help='model_share_type')
    parser.add_argument('--model_fix_layer', type=int, default=0, choices=(0, 1, 6), help='model_fix_layer')
    parser.add_argument('--local_entropy_coef', type=float, default=0.1, help='local_entropy_coef')
    parser.add_argument('--layer_arch_coef', type=float, default=0.05, help='layer_arch_coef')
    parser.add_argument('--trans_arch_coef', type=float, default=0.025, help='trans_arch_coef')
    parser.add_argument('--mode_name', type=str, default='GRAPH', help='GRAPH or PYNATIVE')

    parser.add_argument('--drop_rate', type=float, default=0.1)
    args = parser.parse_args(args)

    ############# hard coded config #############

    args.work_dir = path_info.work_dir
    args.config_root = path_info.config_root
    args.mask_path = path_info.mask_path
    args.vit_model_path = path_info.vit_model_path
    ########### config image root path ##########
    if 'merge' in args.dataset:
        args.image_root = path_info.image_root_merge
        args.config_root = path_info.config_root1
    elif 'coco' in args.dataset:
        args.image_root = path_info.image_root_coco
    else:
        args.image_root = path_info.image_root_referit

    args.use_index = False if 'merge' in args.dataset else True
    ########### config splits for eval ##########
    if args.dataset == 'refcoco' or args.dataset == 'refcoco+':
        args.eval_splits = ('val', 'testA', 'testB')
    elif args.dataset == 'refcocog':
        if args.splitBy == 'google':
            args.eval_splits = ('val',)
        else:
            args.eval_splits = ('val', 'test')
    elif args.dataset == 'refclef':
        args.eval_splits = ('test',)
    else:
        args.eval_splits = ('val',)

    ############# auto set config #############
    # log_name = f"{datetime.now().strftime('%b%d_%H-%M')}_{args.short_comment}"
    log_name = f"{datetime.now().strftime('(%b%d_%H)')}{args.short_comment}"
    args.root_dir = os.path.join(args.work_dir, args.experiment_name)
    args.log_dir = os.path.join(args.root_dir, 'runs', log_name)
    # args.root_dir = args.log_dir
    os.makedirs(args.root_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    return args
