import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from torchvision.transforms.functional import to_tensor

import source.utils.corner_utils as cutil


class MagicPointTrainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """

    def __init__(self, model, losses, metrics, optimizer, resume, config,
                 data_loader, valid_data_loader=None, lr_scheduler=None, train_logger=None):
        super(MagicPointTrainer, self).__init__(model, losses, metrics, optimizer, resume, config, train_logger)
        self.config = config
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = config['trainer'].get('log_step',int(np.sqrt(data_loader.batch_size)))

    def _eval_metrics(self, output, target):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output, target)
            self.writer.add_scalar('{}'.format(metric.__name__), acc_metrics[i])
        return acc_metrics

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        np.random.seed()
        self.model.train()

        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))
        for batch_idx, data in enumerate(self.data_loader):

            image=data[0]
            kp_map=data[1]
            valid_mask=data[2]

            image=image.to(self.device)
            kp_map=kp_map.to(self.device)
            valid_mask=kp_map.to(self.device)


            self.optimizer.zero_grad()
            output = self.model(image)
            detector_loss=self.losses['DetectorLoss'][0]

            loss=detector_loss.forward(output,kp_map)
            loss.backward()
            self.optimizer.step()

            loss_val=loss.item()
            self.writer.set_step((epoch - 1) * len(self.data_loader) + batch_idx)
            self.writer.add_scalar('loss', loss_val)
            total_loss += loss_val
            #total_metrics += self._eval_metrics(output, target)

            if self.verbosity >= 2 and batch_idx % self.log_step == 0:
                self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                    epoch,
                    batch_idx * self.data_loader.batch_size,
                    self.data_loader.n_samples,
                    100.0 * batch_idx / len(self.data_loader),
                    loss_val))
            if self.verbosity >= 2 and batch_idx % self.log_step*5 == 0:
                img_data=image.cpu().numpy()[0]
                semi=output.cpu().detach().numpy()[0]
                gt_heatmap=kp_map.cpu().numpy()[0]
                #Draw the ground truth. Doing this before so the detected keypoints can cover the gt
                img=cutil.draw_heatmap(img_data,gt_heatmap,color=(0,1.0,0))
                img=cutil.draw_model_output_with_nms(img,semi)
                t_img=to_tensor(img)
                self.writer.add_image('pts',t_img)

        log = {
            'loss': total_loss / len(self.data_loader),
           # 'metrics': (total_metrics / len(self.data_loader)).tolist()
        }

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log = {**log, **val_log}

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        total_val_loss = 0
        total_val_metrics = np.zeros(len(self.metrics))
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.loss(output, target)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.writer.add_scalar('loss', loss.item())
                total_val_loss += loss.item()
                total_val_metrics += self._eval_metrics(output, target)
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        return {
            'val_loss': total_val_loss / len(self.valid_data_loader),
            'val_metrics': (total_val_metrics / len(self.valid_data_loader)).tolist()
        }
