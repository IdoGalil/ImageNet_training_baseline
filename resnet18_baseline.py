import os
from datetime import datetime
import pandas as pd
from config import args

import torch
from torch.nn import functional as F

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from pytorch_lightning.utilities.metrics import metrics_to_scalars
from torchmetrics import Accuracy
import wandb

from torchvision.models import resnet18, resnet34, resnet50
from data import all_data_modules

from utils import uncertainty_metrics


class DevModule(pl.LightningModule):
    def __init__(self, hparams):
        super(DevModule, self).__init__()
        self.save_hyperparameters(hparams)
        self.accuracy = Accuracy()
        self.csv_logger = None

        self.model = resnet18(num_classes=self.hparams.num_classes)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            momentum=0.9,
            nesterov=True
        )
        steps_per_epoch = len(self.train_dataloader()) // self.hparams.gpus
        total_steps = self.hparams.max_epochs * steps_per_epoch

        s = torch.optim.lr_scheduler.StepLR(optimizer, step_size=steps_per_epoch * 30, gamma=0.1)

        scheduler = {
            "scheduler": s,
            "interval": "step",
            "name": "learning_rate"
        }

        return [optimizer], [scheduler]

    def _log_uncertainty(self, log_title, samples_certainties):
        indices_sorting_by_confidence = torch.argsort(samples_certainties[:, 0], descending=True)
        sorted_samples_certainties = samples_certainties[indices_sorting_by_confidence]
        self.log(f'confidence_statistics/confidence_variance_{log_title}',
                 uncertainty_metrics.confidence_variance(sorted_samples_certainties), rank_zero_only=True)
        self.log(f'confidence_statistics/confidence_mean_{log_title}',
                 uncertainty_metrics.confidence_mean(sorted_samples_certainties), rank_zero_only=True)
        self.log(f'confidence_statistics/confidence_median_{log_title}',
                 uncertainty_metrics.confidence_median(sorted_samples_certainties), rank_zero_only=True)
        self.log(f'confidence_statistics/confidence_gini_{log_title}',
                 uncertainty_metrics.gini(sorted_samples_certainties), rank_zero_only=True)
        gamma_correlation = uncertainty_metrics.gamma_correlation(sorted_samples_certainties, sort=False)
        self.log(f'ranking/gamma_{log_title}',
                 gamma_correlation['gamma'], rank_zero_only=True)
        self.log(f'ranking/auroc_{log_title}',
                 gamma_correlation['AUROC'], rank_zero_only=True)
        AURC = uncertainty_metrics.AURC_calc(sorted_samples_certainties, sort=False)
        self.log(f'ranking/aurc_{log_title}',
                 AURC, rank_zero_only=True)
        # TODO: Fix the E-AURC
        self.log(f'ranking/eaurc_{log_title}',
                 uncertainty_metrics.EAURC_calc(AURC, self.accuracy.compute()), rank_zero_only=True)
        ece = uncertainty_metrics.ECE_calc(sorted_samples_certainties)
        self.log(f'ece/ece_{log_title}',
                 ece[0], rank_zero_only=True)
        self.log(f'ece/mce_{log_title}',
                 ece[1], rank_zero_only=True)
        if samples_certainties.shape[0] > 200:  # Smaller than that would be too noisy
            self.log(f'SAC/coverage_for_0.90_accuracy_{log_title}',
                     uncertainty_metrics.coverage_for_desired_accuracy(sorted_samples_certainties, sort=False,
                                                                       accuracy=0.90, start_index=200),
                     rank_zero_only=True)
            self.log(f'SAC/coverage_for_0.95_accuracy_{log_title}',
                     uncertainty_metrics.coverage_for_desired_accuracy(sorted_samples_certainties, sort=False,
                                                                       accuracy=0.95, start_index=200),
                     rank_zero_only=True)
            self.log(f'SAC/coverage_for_0.99_accuracy_{log_title}',
                     uncertainty_metrics.coverage_for_desired_accuracy(sorted_samples_certainties, sort=False,
                                                                       accuracy=0.99, start_index=200),
                     rank_zero_only=True)

        if self.local_rank == 0 and type(self.logger) is WandbLogger and not self.trainer.sanity_checking:
            selective_risks, coverages = uncertainty_metrics.selective_risks_calc(sorted_samples_certainties,
                                                                                  sort=False)
            best_selective_risks, _ = uncertainty_metrics.best_selective_risks_calc(sorted_samples_certainties)
            selected_indices_to_present = torch.arange(1, selective_risks.shape[0] + 1) % (
                    selective_risks.shape[0] / 1000) == 0
            self.logger.log_metrics({f'rc_curve/{log_title}':
                wandb.plot.line_series(
                    xs=coverages[selected_indices_to_present].tolist(),
                    ys=[selective_risks[selected_indices_to_present].tolist(),
                        best_selective_risks[selected_indices_to_present].tolist()],
                    keys=['softmax', 'best'],
                    title=f'rc_curve/{log_title}',
                    xname='coverage')})

    def training_step(self, batch, batch_id):
        images, labels = batch
        preds = self.model(images)
        probs = F.softmax(preds, dim=1)
        loss = F.cross_entropy(preds, labels)
        acc = self.accuracy(probs, labels)

        self.log('loss/train', loss)
        self.log('acc/train', acc)

        return loss

    def validation_step(self, batch, batch_id):
        images, labels = batch
        preds = self.model(images)
        probs = F.softmax(preds, dim=1)
        loss = F.cross_entropy(preds, labels)
        acc = self.accuracy(probs, labels)

        self.log('loss/val', loss)
        self.log('acc/val', acc)

        confidence = probs.max(dim=1)[0]
        correctness = probs.argmax(dim=1) == labels
        samples_certainties = torch.stack([confidence, correctness], dim=1)
        return samples_certainties

    def validation_epoch_end(self, outputs):
        if self.hparams.gpus > 1:
            samples_certainties = torch.cat(list(torch.cat(self.all_gather(outputs), dim=1)))
        else:
            samples_certainties = torch.cat(outputs)
        self._log_uncertainty('val', samples_certainties)

        if self.local_rank == 0 and type(self.logger) is WandbLogger and not self.trainer.sanity_checking:
            if self.csv_logger is None:
                self.csv_logger = CSVLogger(save_dir=self.hparams.project, name=self.hparams.log_id, version='logs')
            self.csv_logger.log_metrics(self.trainer.logged_metrics, step=self.global_step)
            self.csv_logger.save()

    def test_step(self, batch, batch_id):
        images, labels = batch
        preds = self.model(images)
        probs = F.softmax(preds, dim=1)
        loss = F.cross_entropy(preds, labels)
        acc = self.accuracy(probs, labels)

        self.log('loss/test', loss)
        self.log('acc/test', acc)

        confidence = probs.max(dim=1)[0]
        correctness = probs.argmax(dim=1) == labels
        samples_certainties = torch.stack([confidence, correctness], dim=1)
        return samples_certainties

    def test_epoch_end(self, outputs):
        if self.hparams.gpus > 1:
            samples_certainties = torch.cat(list(torch.cat(self.all_gather(outputs), dim=1)))
        else:
            samples_certainties = torch.cat(outputs)
        self._log_uncertainty('test', samples_certainties)


def train(args):
    learning_rate_callback = pl.callbacks.LearningRateMonitor(logging_interval='step')
    checkpoint_callback = pl.callbacks.ModelCheckpoint(filename='best', monitor='acc/val', mode='max', save_last=True)

    callbacks = [checkpoint_callback]
    if args.logger:
        callbacks.append(learning_rate_callback)

    logger = None
    if args.logger:
        args.project = 'experiments'
        args.log_id = f'{args.log_name}-{datetime.now().strftime("%Y%m%d-%H%M%S")}'
        logger = WandbLogger(name=args.log_name, id=args.log_id, project=args.project)
    trainer = pl.Trainer(gpus=args.gpus,
                         accelerator='ddp' if args.gpus > 1 else None,
                         logger=logger,
                         deterministic=True,
                         weights_summary=None,
                         log_every_n_steps=1,
                         max_epochs=args.max_epochs,
                         fast_dev_run=args.dev,
                         callbacks=callbacks,
                         # resume_from_checkpoint=f'experiments/{log_id}/checkpoints/last.ckpt'
                         )

    model = DevModule(args)
    # Use this line to train the model. commenting it would cause the model to only be tested instead
    trainer.fit(model, train_dataloaders=data.train_dataloader(), val_dataloaders=data.test_dataloader())

    # Use the following lines to test the model
    weights_generic_name = 'best.ckpt'
    # model = DevModule.load_from_checkpoint(f'experiments/{log_id}/checkpoints/best.ckpt')
    trainer.test(model, data.test_dataloader())


if __name__ == '__main__':
    pl.seed_everything(args.seed)

    args.gpus = len(args.gpu_id.split(','))
    args.max_epochs = 90
    args.data = 'imagenet'
    args.batch_size = 64

    args.learning_rate = 0.1
    args.weight_decay = 1e-4
    args.log_name = f'ResNet18_seed_{args.seed}'

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    data = all_data_modules[args.data](
        data_dir=args.data_dir,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=False,
        drop_last=False,
        seed=args.seed
    )
    args.num_classes = data.num_classes
    data.prepare_data()
    data.setup()

    train(args)
