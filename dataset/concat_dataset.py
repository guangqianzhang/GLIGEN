from .catalog import DatasetCatalog
from ldm.util import instantiate_from_config
import torch


class ConCatDataset():
    def __init__(self, dataset_name_list, ROOT, train=True, repeats=None):
        self.datasets = []
        cul_previous_dataset_length = 0
        offset_map = []
        which_dataset = []
        # repeats？？
        if repeats is None:  # 1
            repeats = [1] * len(dataset_name_list)  # config中设置，可以为list
        else:
            assert len(repeats) == len(dataset_name_list)
        # seg：ADEChallengeData2016 ，depth:tsv
        Catalog = DatasetCatalog(ROOT)  # 所有数据集类型target和路径train_params
        for dataset_idx, (dataset_name, yaml_params) in enumerate(dataset_name_list.items()): # config中设置，可以为list
            repeat = repeats[dataset_idx]

            dataset_dict = getattr(Catalog, dataset_name)  # 获得对应数据集路径

            target = dataset_dict['target']  # 数据类型
            params = dataset_dict['train_params'] if train else dataset_dict['val_params']  # 数据路径
            if yaml_params is not None:  # yaml中设置的数据参数
                params.update(yaml_params)
            dataset = instantiate_from_config(dict(target=target, params=params))

            self.datasets.append(dataset)
            for _ in range(repeat):
                offset_map.append(torch.ones(len(dataset)) * cul_previous_dataset_length)  # tensor(37476)[0,0...]
                which_dataset.append(torch.ones(len(dataset)) * dataset_idx)  # 数据集id
                cul_previous_dataset_length += len(dataset)  # 37476
        offset_map = torch.cat(offset_map, dim=0).long()
        self.total_length = cul_previous_dataset_length

        self.mapping = torch.arange(self.total_length) - offset_map
        self.which_dataset = torch.cat(which_dataset, dim=0).long()

    def total_images(self):
        count = 0
        for dataset in self.datasets:
            print(dataset.total_images())
            count += dataset.total_images()
        return count

    def __getitem__(self, idx):
        dataset = self.datasets[self.which_dataset[idx]]
        return dataset[self.mapping[idx]]

    def __len__(self):
        return self.total_length
