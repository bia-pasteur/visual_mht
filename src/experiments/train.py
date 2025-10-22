"""Train a model to predict contrastive features for patches"""

import dataclasses
import enum
import pathlib
from typing import cast, Dict, Optional, Tuple

import dacite
import torch.utils.data
import yaml  # type: ignore

import deep_trainer
from deep_trainer.pytorch import metric

from .. import encoder
from .. import optim
from .. import utils
from ..data import dataset
from ..loss import triplet, val_metrics


class ContrastiveMethod(enum.Enum):
    TRIPLET = "triplet"
    INFONCE = "infonce"
    BARLOW = "barlow"


@dataclasses.dataclass
class TrainingConfig:
    batch_size: int = 64
    epochs: int = 100
    epoch_size: int = 100
    method: ContrastiveMethod = ContrastiveMethod.TRIPLET
    triplet_margin: float = 1.0
    infonce_tau: float = 0.1
    barlow_lambda: float = 0.01
    validation_pairs_overlap: bool = False
    num_workers: int = 8
    use_amp: bool = False
    save_mode: str = "small"
    checkpoint: Optional[pathlib.Path] = None
    seed: int = 666


@dataclasses.dataclass
class ExperimentConfig:
    dataset: dataset.DatasetConfig
    training: TrainingConfig
    model: encoder.ModelConfig
    optimizer: optim.OptimizerConfig


class TripletTrainer(deep_trainer.PytorchTrainer):
    def eval_step(self, batch) -> Dict[str, float]:
        inputs, *targets = batch

        inputs = inputs.to(self.device, non_blocking=True)
        if hasattr(self, "val_transforms"):
            inputs = getattr(self, "val_transforms")(inputs)

        with torch.autocast(self.device.type, enabled=self.use_amp):
            predictions = self.model(inputs)
            self.metrics_handler.update((None, targets), predictions)

        return self.metrics_handler.last_values

    def train_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], criterion) -> Dict[str, float]:
        criterion = cast(triplet.TripletLoss, criterion)

        anchor, positive, negative = batch
        inputs = torch.cat((anchor, positive, negative)).to(self.device, non_blocking=True)

        if hasattr(self, "train_transforms"):
            inputs = getattr(self, "train_transforms")(inputs)

        with torch.autocast(self.device.type, enabled=self.use_amp):
            outputs: torch.Tensor = self.model(inputs)

            anchor_out, positive_out, negative_out = outputs.split(anchor.shape[0])

            loss = criterion(anchor_out, positive_out, negative_out)
            self.metrics_handler.update(batch, outputs)

        self.backward(loss)

        metrics = self.metrics_handler.last_values
        metrics["Loss"] = loss.item()
        metrics["BatchUsage"] = criterion.triplet_used_proportion

        return metrics


def main(name: str, cfg_data: dict) -> None:
    print("Running:", name)
    print(yaml.dump(cfg_data))

    cfg = dacite.from_dict(ExperimentConfig, cfg_data, dacite.Config(cast=[pathlib.Path, tuple, enum.Enum]))

    # Seed
    utils.enforce_all_seeds(cfg.training.seed)

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU is", torch.cuda.get_device_name(device))
    else:
        device = torch.device("cpu")
        print("Only CPU available")

    # Dataset
    if cfg.training.method == ContrastiveMethod.TRIPLET:
        trainset = cfg.dataset.build_triplet_dataset()
        criterion = triplet.TripletLoss(cfg.training.triplet_margin)
        Trainer = TripletTrainer
    else:
        raise NotImplementedError("Other methods are not implemented yet")

    train_loader = torch.utils.data.DataLoader(
        trainset,
        cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        worker_init_fn=utils.create_seed_worker(cfg.training.seed),
        persistent_workers=True,
        pin_memory=True,
        drop_last=True,
    )
    valset = cfg.dataset.build_val_dataset()
    val_loader = torch.utils.data.DataLoader(  # type: ignore
        valset,
        cfg.training.batch_size * 10,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        worker_init_fn=utils.create_seed_worker(cfg.training.seed),
        persistent_workers=True,
        pin_memory=True,
    )

    print(f"Training with {len(trainset)} patches ({trainset.dataset.root})")
    print(f"Validation with {len(valset)} patches, with {(sum(valset.annotated))} annotations ({valset.root})")

    # Model
    model = cfg.model.build().to(device)
    print("MODEL\n", model)

    # Single gpu and no EMA yet
    # if cfg["model"].get("use_dp", False):
    #     print(f"Using {torch.cuda.device_count()} GPUs with DP")
    #     model = torch.nn.DataParallel(model)
    # if cfg["model"].get("ema", 0) > 0:
    #     model = EMA(model, cfg["model"]["ema"])

    # Metrics
    dist_prereq = val_metrics.PatchDist(
        val_metrics.OutputsSaver(),
        val_metrics.ValidationTargetsSaver(),
        across_pairs=not cfg.training.validation_pairs_overlap,
    )
    link_prereq = val_metrics.LinkSolver(dist_prereq)

    metric_handler = metric.MetricsHandler(
        [
            val_metrics.HardestTripletLoss(dist_prereq, margin=cfg.training.triplet_margin),
            val_metrics.LinkAP(link_prereq),
            val_metrics.LinkBestF1(link_prereq),
            val_metrics.LinkBestF1Recall(link_prereq),
            val_metrics.LinkBestF1Precision(link_prereq),
        ]
    )
    metric_handler.set_validation_metric(0)

    # Optimizer & Scheduler
    optimizer = cfg.optimizer.build(model)
    # scheduler = build_scheduler(optimizer, cfg) Not supported yet

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        metrics_handler=metric_handler,
        device=device,
        save_mode=cfg.training.save_mode,
        use_amp=cfg.training.use_amp,
    )

    # Let's transforms directly in the trainer (by batch and on device)
    setattr(trainer, "train_transforms", trainset.transforms)
    setattr(trainer, "val_transforms", valset.transforms)
    trainset.transforms = dataset.identity
    valset.transforms = dataset.identity

    if cfg.training.checkpoint is not None:
        trainer.load(str(cfg.training.checkpoint))

    try:
        trainer.train(cfg.training.epochs, train_loader, criterion, val_loader, epoch_size=cfg.training.epoch_size)
    except KeyboardInterrupt:  # Handle Ctrl+C by stopping the training and storing the results
        pass

    print("Reloading best model and evaluating...")
    trainer.load("experiments/checkpoints/best.ckpt")

    # Trigger evaluation of links metrics
    link_prereq.last_call = link_prereq.EVALUATE_EVERY

    metrics = trainer.evaluate(val_loader)

    result_string = yaml.dump(metrics)

    print("--------RESULTS---------")
    print(result_string)

    with open("results.yml", "w", encoding="utf-8") as f:
        f.write(result_string)
