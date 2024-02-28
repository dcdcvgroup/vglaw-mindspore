"""
Preprocess a raw pth dataset
Output: pth file has
{
    items: {
        split: [{phrase, image, bbox, category, attributes}]
    }
    index: {
        category: {cat_id -> cat_name},
        image: {img_id -> file_name}
    }
}
"""
import os
import io
import pickle
import sys
import yaml
import torch
import argparse
import os.path as osp
from prettytable import PrettyTable

if not hasattr(yaml, 'dumps'):
    def dumps(data, *args, **kwargs):
        strio = io.StringIO()
        yaml.dump(data, strio, *args, **kwargs)
        return strio.getvalue()
    yaml.dumps = dumps

parser = argparse.ArgumentParser()
parser.add_argument('--output', help='output file')
parser.add_argument('--data_root', default='data', type=str, help='data folder containing images and four datasets.')
parser.add_argument('--dataset', default='refcocog', type=str, help='refclef/refcoco/refcoco+/refcocog')
parser.add_argument('--splitBy', default='umd', type=str, help='berkeley/unc/google')

# argparse
args = parser.parse_args()
# args.data_root = '/home/suwei/Storage/Dataset/Referring'
if args.dataset == 'refclef':
    args.data_root = '/data0/zekang'
else:
    args.data_root = '/home/zekang/vglaw/data'

if args.output is None:
    # args.output = f'{args.dataset}({args.splitBy}).pth'
    args.output = f'{args.dataset}({args.splitBy}).pkl'
print('\033[32mparse input config:\033[0m')
print(yaml.dumps(vars(args)))

# mkdir and write pth file
# 读取当前脚本的绝对地址，去除文件名，作为.cache文件的存储地址，再创建.cache文件夹
os.makedirs(osp.join(osp.dirname(osp.abspath(__file__)), '.cache'), exist_ok=True)
output_path = osp.join(osp.dirname(osp.abspath(__file__)), '.cache', args.output)

# load refer
# sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
from refer import REFER
refer = REFER(args.data_root, args.dataset, args.splitBy)

dataset = {}
for sent_id, sent in refer.Sents.items():
    ref = refer.sentToRef[sent_id]
    split = ref['split']
    ann_id = ref['ann_id']
    item = {
        'bbox':         refer.Anns[ann_id]['bbox'],
        'phrase':       sent['sent'],
        'ref_id':       ref['ref_id'],
        'image_id':     ref['image_id'],
        'category_id':  ref['category_id'],
        'attributes':   None
    }
    dataset[split] = dataset.get(split, []) + [item]

print('Summarize:')
pt = PrettyTable(['split', 'number'])
for split, items in dataset.items():
    pt.add_row([split, len(items)])
print(pt)

# save the results to pth file
print(f'writting result into {output_path} ...')
"""
# 保存为pytorch的pth文件
torch.save(
    {
        'items': dataset,
        'index': {
            'category': refer.Cats,
            'image': {img_id: img['file_name'] for img_id, img in refer.Imgs.items()}
        }
    },
    output_path
)
"""

# 保存为python序列化文件pickle
save_dict = {
    'items': dataset,
    'index': {
        'category': refer.Cats,
        'image': {img_id: img['file_name'] for img_id, img in refer.Imgs.items()}
    }
}
with open(output_path, "wb") as f:
    pickle.dump(save_dict, f)
