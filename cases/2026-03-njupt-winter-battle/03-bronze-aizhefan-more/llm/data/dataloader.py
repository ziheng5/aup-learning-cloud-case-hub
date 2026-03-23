import torch
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, SequentialSampler
from typing import Optional, Dict, Union
from .dataset import DataCollatorForLM


def create_dataloader(
    dataset,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = True,
    prefetch_factor: int = 2,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    tokenizer = None
) -> DataLoader:
    """
    创建数据加载器
    
    Args:
        dataset: 数据集
        batch_size: 批次大小
        shuffle: 是否打乱
        num_workers: 工作进程数
        pin_memory: 是否锁页内存
        drop_last: 是否丢弃最后一个批次
        prefetch_factor: 预取因子
        distributed: 是否分布式训练
        rank: 进程rank
        world_size: 总进程数
        tokenizer: 分词器（用于数据整理器）
    
    Returns:
        DataLoader实例
    """
    if distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle
        )
    else:
        sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
    
    collate_fn = DataCollatorForLM(tokenizer) if tokenizer else None
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=drop_last,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=num_workers > 0
    )
    
    return dataloader


class PrefetchDataLoader:
    """
    带预取的数据加载器
    使用CUDA流异步预取数据到GPU
    """
    
    def __init__(
        self,
        dataloader: DataLoader,
        device: torch.device,
        prefetch_count: int = 2
    ):
        self.dataloader = dataloader
        self.device = device
        self.prefetch_count = prefetch_count
        
        self.stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        self.next_batch = None
    
    def _preload(self):
        """预加载下一批数据"""
        if self.stream is None:
            return
        
        try:
            self.next_batch = next(self.dataloader_iter)
            with torch.cuda.stream(self.stream):
                self.next_batch = {
                    k: v.to(self.device, non_blocking=True)
                    for k, v in self.next_batch.items()
                }
        except StopIteration:
            self.next_batch = None
    
    def __iter__(self):
        self.dataloader_iter = iter(self.dataloader)
        self._preload()
        return self
    
    def __next__(self):
        if self.stream is not None:
            self.stream.synchronize()
        
        batch = self.next_batch
        
        if batch is None:
            raise StopIteration
        
        self._preload()
        return batch
    
    def __len__(self):
        return len(self.dataloader)
