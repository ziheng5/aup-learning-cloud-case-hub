import os
import math
import logging
import torch
import torch.nn as nn
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, Union
from tqdm import tqdm
from pathlib import Path

from configs.model_config import TrainingConfig, ModelConfig
from training.optimizer import create_optimizer
from training.scheduler import create_scheduler

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class Trainer:
    """
    训练器类
    支持混合精度、梯度累积、梯度检查点等
    """
    
    def __init__(
        self,
        model: nn.Module,
        args: TrainingConfig,
        model_config: ModelConfig,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: Optional[torch.device] = None,
        local_rank: int = -1,
        world_size: int = 1
    ):
        self.model = model
        self.args = args
        self.model_config = model_config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.local_rank = local_rank
        self.world_size = world_size
        
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.optimizer = optimizer if optimizer else self._create_optimizer()
        self.scheduler = scheduler if scheduler else self._create_scheduler()
        
        self.scaler = amp.GradScaler() if args.mixed_precision == "fp16" else None
        
        self.global_step = 0
        self.epoch = 0
        self.total_loss = 0.0
        
        self._setup_model()
    
    def _setup_model(self):
        """设置模型（编译、分布式等）"""
        if self.args.compile_model and hasattr(torch, 'compile'):
            logger.info("Compiling model with torch.compile...")
            self.model = torch.compile(self.model, mode="reduce-overhead")
        
        if self.local_rank != -1:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False
            )
        
        if self.args.gradient_checkpointing:
            if hasattr(self.model, "enable_gradient_checkpointing"):
                self.model.enable_gradient_checkpointing()
            elif hasattr(self.model, "module") and hasattr(self.model.module, "enable_gradient_checkpointing"):
                self.model.module.enable_gradient_checkpointing()
        
        self.model.to(self.device)
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """创建优化器"""
        return create_optimizer(
            self.model,
            learning_rate=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
            betas=self.args.betas,
            eps=self.args.eps
        )
    
    def _create_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        """创建学习率调度器"""
        return create_scheduler(
            self.optimizer,
            scheduler_type="cosine",
            warmup_steps=self.args.warmup_steps,
            max_steps=self.args.max_steps,
            min_lr=self.args.min_lr
        )
    
    def _save_checkpoint(self, path: str, is_best: bool = False):
        """保存检查点"""
        if self.local_rank not in [-1, 0]:
            return
        
        os.makedirs(path, exist_ok=True)
        
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        
        model_path = os.path.join(path, "pytorch_model.bin")
        torch.save(model_to_save.state_dict(), model_path)
        
        config_path = os.path.join(path, "config.pt")
        torch.save(self.model_config, config_path)
        
        optimizer_path = os.path.join(path, "optimizer.pt")
        torch.save(self.optimizer.state_dict(), optimizer_path)
        
        scheduler_path = os.path.join(path, "scheduler.pt")
        torch.save(self.scheduler.state_dict(), scheduler_path)
        
        trainer_state = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "total_loss": self.total_loss
        }
        torch.save(trainer_state, os.path.join(path, "trainer_state.pt"))
        
        if is_best:
            best_path = os.path.join(path, "best_model")
            os.makedirs(best_path, exist_ok=True)
            import shutil
            for f in os.listdir(path):
                if os.path.isfile(os.path.join(path, f)):
                    shutil.copy2(os.path.join(path, f), best_path)
        
        logger.info(f"Checkpoint saved to {path}")
    
    def _load_checkpoint(self, path: str):
        """加载检查点"""
        model_path = os.path.join(path, "pytorch_model.bin")
        optimizer_path = os.path.join(path, "optimizer.pt")
        scheduler_path = os.path.join(path, "scheduler.pt")
        trainer_state_path = os.path.join(path, "trainer_state.pt")
        
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
            model_to_load = self.model.module if hasattr(self.model, "module") else self.model
            model_to_load.load_state_dict(state_dict)
        
        if os.path.exists(optimizer_path):
            self.optimizer.load_state_dict(torch.load(optimizer_path, weights_only=False))
        
        if os.path.exists(scheduler_path):
            self.scheduler.load_state_dict(torch.load(scheduler_path, weights_only=False))
        
        if os.path.exists(trainer_state_path):
            state = torch.load(trainer_state_path, weights_only=False)
            self.global_step = state["global_step"]
            self.epoch = state["epoch"]
            self.total_loss = state["total_loss"]
        
        logger.info(f"Checkpoint loaded from {path}")
    
    def _training_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """执行单个训练步骤"""
        self.model.train()
        
        batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
        
        if self.scaler is not None:
            with amp.autocast():
                outputs = self.model(**batch)
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
                loss = loss / self.args.gradient_accumulation_steps
            
            self.scaler.scale(loss).backward()
        else:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.args.mixed_precision == "bf16"):
                outputs = self.model(**batch)
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
                loss = loss / self.args.gradient_accumulation_steps
            
            loss.backward()
        
        return loss.item()
    
    def _optimization_step(self):
        """执行优化步骤"""
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
        
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.args.max_grad_norm
        )
        
        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
    
    def train(self, resume_from_checkpoint: Optional[str] = None):
        """
        开始训练
        
        Args:
            resume_from_checkpoint: 可选的检查点路径用于恢复训练
        """
        if resume_from_checkpoint:
            self._load_checkpoint(resume_from_checkpoint)
        
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(self.train_dataloader.dataset)}")
        logger.info(f"  Num Epochs = {self.args.max_steps // len(self.train_dataloader) + 1}")
        logger.info(f"  Batch size = {self.args.batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {self.args.max_steps}")
        
        self.model.train()
        self.optimizer.zero_grad()
        
        train_iterator = tqdm(
            range(self.global_step, self.args.max_steps),
            initial=self.global_step,
            desc="Steps",
            disable=self.local_rank not in [-1, 0]
        )
        
        epoch_loss = 0.0
        for step in train_iterator:
            if step % len(self.train_dataloader) == 0 and step > self.global_step:
                self.epoch += 1
                if hasattr(self.train_dataloader.sampler, "set_epoch"):
                    self.train_dataloader.sampler.set_epoch(self.epoch)
            
            try:
                batch = next(train_dataloader_iter)
            except (StopIteration, NameError):
                train_dataloader_iter = iter(self.train_dataloader)
                batch = next(train_dataloader_iter)
            
            loss = self._training_step(batch)
            epoch_loss += loss
            
            if (step + 1) % self.args.gradient_accumulation_steps == 0:
                self._optimization_step()
                self.global_step += 1
                
                if self.global_step % self.args.logging_steps == 0:
                    avg_loss = epoch_loss / self.args.logging_steps
                    epoch_loss = 0.0
                    
                    lr = self.scheduler.get_last_lr()[0]
                    train_iterator.set_postfix({
                        "loss": f"{avg_loss:.4f}",
                        "lr": f"{lr:.2e}",
                        "epoch": self.epoch
                    })
                    
                    logger.info(f"Step {self.global_step}: loss={avg_loss:.4f}, lr={lr:.2e}")
                
                if self.global_step % self.args.save_steps == 0:
                    checkpoint_path = os.path.join(
                        self.args.output_dir,
                        f"checkpoint-{self.global_step}"
                    )
                    self._save_checkpoint(checkpoint_path)
                
                if self.global_step % self.args.eval_steps == 0 and self.eval_dataloader:
                    eval_loss = self.evaluate()
                    logger.info(f"Evaluation at step {self.global_step}: loss={eval_loss:.4f}")
        
        final_path = os.path.join(self.args.output_dir, "final")
        self._save_checkpoint(final_path, is_best=True)
        logger.info("Training complete!")
    
    @torch.no_grad()
    def evaluate(self) -> float:
        """评估模型"""
        logger.info("***** Running evaluation *****")
        
        self.model.eval()
        total_loss = 0.0
        num_steps = 0
        
        eval_iterator = tqdm(
            self.eval_dataloader,
            desc="Evaluating",
            disable=self.local_rank not in [-1, 0]
        )
        
        for batch in eval_iterator:
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
            
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.args.mixed_precision == "bf16"):
                outputs = self.model(**batch)
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            
            total_loss += loss.item()
            num_steps += 1
        
        avg_loss = total_loss / num_steps
        ppl = math.exp(avg_loss)
        
        logger.info(f"Evaluation results: loss={avg_loss:.4f}, ppl={ppl:.2f}")
        
        self.model.train()
        return avg_loss
    
    def get_model(self) -> nn.Module:
        """获取模型（解包DDP）"""
        return self.model.module if hasattr(self.model, "module") else self.model
