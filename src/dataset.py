import os
import multiprocessing
import mindspore as ms
import mindspore.ops as ops
from mindspore import dataset as ds
import numpy as np
from PIL import Image
from time import time
import transforms as T
from transforms import MyTokenizer
from xenv import console
import pickle
from mindformers import AutoTokenizer
from mindspore.dataset import vision
from mindspore.dataset.vision import Inter
from mindspore.communication.management import init, get_rank, get_group_size
from distributed_sampler import DistributedSampler
from memory_profiler import profile


# @ms.jit_class
class ReferDataset(object):
    def __init__(self, config_root, image_root,
                 dataset, splitBy, split,
                 trans_args, use_index=True):
        start_time = time()
        self.dataset = dataset
        self.use_index = use_index
        splits = [split] if isinstance(split, str) else split

        if splitBy is None:
            config_path = os.path.join(config_root, f"{dataset}.pkl")
            console.print(f'Preparing {dataset} using split({",".join(splits)})...', style='cyan')
        else:
            config_path = os.path.join(config_root, f'{dataset}({splitBy}).pkl')
            console.print(f'Preparing {dataset}({splitBy}) using split({",".join(splits)})...', style='cyan')

        if os.path.exists(config_path):
            with open(config_path, 'rb') as f:
                annotation = pickle.load(f)
        else:
            raise FileNotFoundError(f"[{config_path}] not found. Run 'tools/preprocess.py' first.")

        self.use_mask = trans_args.use_mask
        if self.use_mask:
            self.mask_root = os.path.join(trans_args.mask_path, f'{dataset}({splitBy})')
            assert os.path.exists(self.mask_root), f'{self.mask_root} not exist.'

        self.index = annotation['index']
        self.image_root = image_root
        self.items = sum([items for split, items in annotation['items'].items() if split in splits], [])
        assert len(self.items) > 0,\
            f"Not supported split({', '.join(splits)}), select in [{', '.join(annotation['items'].keys())}]"

        max_len = trans_args.max_len
        if max_len is None or max_len <= 0:
            max_len = int(1e30)

        support_split_types = ','.join(annotation['items'].keys())
        split_text_nums = len(self.items)
        total_text_nums = sum(len(items) for items in annotation['items'].values())
        console.print(
            f'  |- split: {support_split_types}',
            f'  |- texts: {split_text_nums}/{total_text_nums}',
            f'  Done({time() - start_time:.3f}s).',
            sep='\n', style='cyan'
        )

    def __getitem__(self, index):
        item = self.items[index]
       
        image_id = item['image_id']
        if self.use_index:
            image_path = os.path.join(self.image_root, self.index['image'][image_id])
            if self.dataset == 'refclef':
                image_path = os.path.join(self.image_root, str(image_id) + '.jpg')
        else:
            image_path = os.path.join(self.image_root, image_id)


        image = Image.open(image_path).convert("RGB")

        xtl, ytl, w, h = item['bbox']
        cx, cy = xtl + 0.5 * w, ytl + 0.5 * h

        bbox = np.array([cx, cy, w, h])
        text = item['phrase']
        text = np.array(text)
        input_dict = {'image': image, 'bbox': bbox, 'text': text, 'raw_text': text}

        if self.use_mask:
            ref_mask_np = np.load(os.path.join(self.mask_root, '%d.npy' % item['ref_id']))
            ref_mask = np.expand_dims(ref_mask_np, axis=0)  # assume only contain masks of one class!!!
            input_dict['ref_mask'] = ref_mask

        if self.use_mask:
            return input_dict['image'], input_dict['bbox'], input_dict['text'], input_dict['raw_text'], input_dict['ref_mask']
        return input_dict['image'], input_dict['text'], input_dict['bbox'], input_dict['raw_text'], None

    def __len__(self):
        return len(self.items)


def create_dataset(args, batch_size, split, is_train=False,
                   rank=0, group_size=1, shuffle=True):
    ds.config.set_prefetch_size(10)
    ds.config.set_enable_watchdog(False)

    if 'train' in split:
        if args.multi_scale:
            sizes = [args.img_size - 32 * r for r in range(5)]
            sizes.extend([args.img_size + 32 * r for r in range(1, 5)])
        else:
            sizes = [args.img_size]
        transform = T.Compose(
            T.RandomResize(args.img_size, *sizes),
            T.RandomCrop(args.img_size, args.img_size, args.img_size),
            T.RandomHorizontalFlip(),
            T.ToTensor(None),
        )
    else:
        transform = T.Compose(
            T.RandomResize(args.img_size, args.img_size),
            T.ToTensor(None),
        )

    if is_train == True:
        referdataset = ReferDataset(args.config_root, args.image_root, args.dataset,
                                    args.splitBy, split=split,
                                    use_index=args.use_index, trans_args=args)
        sampler = ms.dataset.DistributedSampler(num_shards=group_size, shard_id=rank)
        num_parallel_workers = 2
        print("<=================start to generate dataset=================>")
        data_set = ms.dataset.GeneratorDataset(source=referdataset,
                                               column_names=['image', 'bbox', 'text', 'raw_text', 'ref_mask'],
                                               sampler=sampler, num_parallel_workers=num_parallel_workers,
                                               python_multiprocessing=is_train)

        data_set = data_set.map(operations=transform,
                                input_columns=['image', 'bbox', 'text', 'raw_text', 'ref_mask'],
                                output_columns=['image', 'bbox', 'text', 'raw_text', 'ref_mask'],
                                num_parallel_workers=num_parallel_workers)

        data_set = data_set.map(operations=T.NormalizeAndPad_AllNumpy(size=args.img_size, translate=args.translate),
                                input_columns=['image', 'bbox', 'text', 'raw_text', 'ref_mask'],
                                output_columns=['image', 'bbox', 'text', 'raw_text', 'ref_mask', 'mask'],
                                num_parallel_workers=num_parallel_workers)

        data_set = data_set.batch(batch_size, drop_remainder=is_train, num_parallel_workers=num_parallel_workers)
        data_set = data_set.map(operations=MyTokenizer(bert_model='bert_base_uncased',
                                                       token_max_len=args.max_len),
                                input_columns=['text'],
                                output_columns=['text'],
                                num_parallel_workers=num_parallel_workers)
        m_type = ms.float32
        data_set = cast_type(data_set, m_type, ['image', 'bbox', 'mask', 'ref_mask'])
        data_set = cast_type(data_set, ms.int32, ['text'])
        print("<=================generating dataset finished=================>")
        return data_set

    elif is_train == False:
        print('==>Generating eval dataset list')
        eval_dataset_list = []
        num_parallel_workers = 2  # min(4, cores // group_size)
        for split in args.eval_splits:
            single_evalset = ReferDataset(args.config_root, args.image_root, args.dataset,
                                          args.splitBy, split=split,
                                          use_index=args.use_index, trans_args=args)
            distributed_sampler = DistributedSampler(len(single_evalset), group_size, rank, shuffle=False)
            single_evalset = ms.dataset.GeneratorDataset(source=single_evalset,
                                                         column_names=['image', 'bbox', 'text', 'raw_text', 'ref_mask'],
                                                         sampler=distributed_sampler, num_parallel_workers=num_parallel_workers,
                                                         python_multiprocessing=is_train)

            single_evalset = single_evalset.map(operations=transform,
                                                input_columns=['image', 'bbox', 'text', 'raw_text', 'ref_mask'],
                                                output_columns=['image', 'bbox', 'text', 'raw_text', 'ref_mask'],
                                                num_parallel_workers=num_parallel_workers)

            single_evalset = single_evalset.map(operations=T.NormalizeAndPad_AllNumpy(size=args.img_size, translate=args.translate),
                                                input_columns=['image', 'bbox', 'text', 'raw_text', 'ref_mask'],
                                                output_columns=['image', 'bbox', 'text', 'raw_text', 'ref_mask', 'mask'],
                                                num_parallel_workers=num_parallel_workers)

            single_evalset = single_evalset.batch(batch_size, drop_remainder=is_train, num_parallel_workers=num_parallel_workers)

            single_evalset = single_evalset.map(operations=MyTokenizer(bert_model='bert_base_uncased',
                                                token_max_len=args.max_len),
                                                input_columns=['text'],
                                                output_columns=['text'],
                                                num_parallel_workers=num_parallel_workers)

            m_type = ms.float32
            single_evalset = cast_type(single_evalset, m_type, ['image', 'bbox', 'mask', 'ref_mask'])
            single_evalset = cast_type(single_evalset, ms.int32, ['text'])
            eval_dataset_list.append(single_evalset)

        return eval_dataset_list

# @profile
def cast_type(dataset: ms.dataset.Dataset, type: ms.Type, columns: list):
    type_cast_operation = ds.transforms.transforms.TypeCast(type)
    for column in columns:
        dataset = dataset.map(operations=type_cast_operation, input_columns=column, num_parallel_workers=2)

    return dataset


def _get_rank_info(distribute):
    """get rank size and rank id"""

    if distribute:
        init()
        rank_id = get_rank()
        device_num = get_group_size()
    else:
        rank_id = 0
        device_num = 1
    return device_num, rank_id

if __name__ == '__main__':
    import xenv as xenv
    args = xenv.parse_args()
    referdataset = ReferDataset(args.config_root, args.image_root, args.dataset,
                                    args.splitBy, split=args.train_split,
                                    use_index=args.use_index, trans_args=args)
    ds_0 = referdataset[0]
    print(f"type(ds_0): {type(ds_0)}, len(ds_0): {len(ds_0)}")
    image, bbox, text, raw_text, ref_mask = ds_0
    """
    transform = T.Compose(
            T.RandomResize(args.img_size, *sizes),
            T.RandomCrop(args.img_size, args.img_size, args.img_size),
            T.RandomHorizontalFlip(),
            T.ToTensor(None),
    """
    if args.multi_scale:
        sizes = [args.img_size - 32 * r for r in range(5)]
        sizes.extend([args.img_size + 32 * r for r in range(1, 5)])
    else:
        sizes = [args.img_size]
    t_resize = T.RandomResize(args.img_size, *sizes)
    image, bbox, text, raw_text, ref_mask = t_resize(image, bbox, text, raw_text, ref_mask)

    t_random_crop = T.RandomCrop(args.img_size, args.img_size, args.img_size)
    image, bbox, text, raw_text, ref_mask = t_random_crop(image, bbox, text, raw_text, ref_mask)

    t_randomHF = T.RandomHorizontalFlip()
    image, bbox, text, raw_text, ref_mask = t_randomHF(image, bbox, text, raw_text, ref_mask)

    t_totensor = T.ToTensor(None)
    image, bbox, text, raw_text, ref_mask = t_totensor(image, bbox, text, raw_text, ref_mask)

    """
    data_set = data_set.map(operations=T.NormalizeAndPad_AllNumpy(size=args.img_size, translate=args.translate),
                                input_columns=['image', 'bbox', 'text', 'raw_text', 'ref_mask'],
                                output_columns=['image', 'bbox', 'text', 'raw_text', 'ref_mask', 'mask'],
                                num_parallel_workers=2)
    """
    t_norm_pad = T.NormalizeAndPad_AllNumpy(size=args.img_size, translate=args.translate)
    image, bbox, text, raw_text, ref_mask, mask = t_norm_pad(image, bbox, text, raw_text, ref_mask)

    print("===")
