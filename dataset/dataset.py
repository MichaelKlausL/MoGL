from torch.utils.data import Dataset
import os.path as osp
import csv
import pickle
import os
import torch
import json
import glob

class ChEBI_20_data_Dataset(Dataset):
    def __init__(
        self,
        data_path,
        dataset,
        split,
        ):
        self.data_path = data_path
        self.cids = []
        self.descriptions = {}
        self.smiles = {}
        
        #load data
        with open(osp.join(data_path, dataset, split+'.txt')) as f:
            reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE, fieldnames = ['cid', 'smiles', 'desc'], skipinitialspace=True)
            next(reader)
            for n, line in enumerate(reader):
                self.descriptions[line['cid']] = line['desc']
                self.smiles[line['cid']] = line['smiles']
                self.cids.append(line['cid'])

    def __len__(self):
        return len(self.cids)

    def __getitem__(self, idx):

        cid = self.cids[idx]

        smiles = self.smiles[cid]

        description = self.descriptions[cid]


        return {
                'description':description,
                'smiles':smiles
                }        
        
class PubChem_Dataset(Dataset):

    def __init__(
        self,
        data_path,
        dataset,
        split,
    ):
        self.data_path = data_path
        jsonl_path = osp.join(data_path, dataset, split + '.jsonl')
        self.data = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                smiles = item.get("smiles")
                # 兼容不同字段名
                description = item.get("description") or item.get("desc") or item.get("text")
                self.data.append({"smiles": smiles, "description": description})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        smiles = self.data[idx]["smiles"]
        description = self.data[idx]["description"]
        return {
            'description': description,
            'smiles': smiles
        }

class PCdes_CLMP_Dataset(Dataset):
    def __init__(
        self,
        data_path,
        dataset,
        split,
        ):
        self.data_path = data_path
        self.descriptions = {}
        self.smiles = {}

        #load data
        with open(osp.join(data_path, dataset, split+'.txt')) as f:
            reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE, fieldnames = ['smiles', 'desc'], skipinitialspace=True)
            next(reader)
            for n, line in enumerate(reader):
                self.descriptions[n] = line['desc']
                self.smiles[n] = line['smiles']
            

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):

        smiles = self.smiles[idx]
        description = self.descriptions[idx]


        return {
                'description':description,
                'smiles':smiles
                }  


class MolFrag_Dataset(Dataset):
    def __init__(
        self,
        data_path,
        dataset,
        split,
    ):
        # 读取指定 split 的单个 jsonl 文件
        # 文件格式为多行 JSON 对象，空行分隔
        # 与其他数据集保持一致：data_path / dataset / {split}.jsonl
        self.data_path = data_path
        self.dataset = dataset
        self.split = split
        if not split:
            raise ValueError("split must be provided for MolFrag_Dataset")
        data_file = osp.join(data_path, dataset, f"{split}.jsonl")
        self.data_file = data_file
        self.records = []
        if not osp.exists(data_file):
            raise FileNotFoundError(f"jsonl file not found: {data_file}")

        # 累积到空行为止，再解析成一条记录
        with open(data_file, "r", encoding="utf-8") as handle:
            buffer = []
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    if buffer:
                        record_str = "".join(buffer)
                        self.records.append(json.loads(record_str))
                        buffer = []
                    continue
                buffer.append(line)
            if buffer:
                record_str = "".join(buffer)
                self.records.append(json.loads(record_str))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = dict(self.records[idx])
        # DataLoader 默认拼接要求长度一致，这里把变长列表转为字符串
        fragments = record.get("fragments")
        if isinstance(fragments, (list, tuple)):
            record["fragments"] = " ".join([str(item) for item in fragments if item is not None])
        keywords = record.get("keywords")
        if isinstance(keywords, (list, tuple)):
            record["keywords"] = " ".join([str(item) for item in keywords if item is not None])
        # 所有任务共享同一条数据，返回完整字段供上层构造任务
        return record