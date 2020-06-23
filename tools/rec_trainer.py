#!/usr/bin/env python
# # -*- coding: utf-8 -*-
# # @Time : 2020/6/3 16:22
# # @Author : jj.wang

import torch
import time
from tqdm import tqdm
import torch.nn.functional as F
from base_trainer import BaseTrainer
from utils import WarmupPolyLR, filter_params_assign_lr


class RecTrainer(BaseTrainer):
    def __init__(self, config, model, criterion, train_loader, validate_loader, metric, converter, post_process=None):
        super(RecTrainer, self).__init__(config, model, criterion)
        self.show_images_iter = self.config['trainer']['show_images_iter']
        self.train_loader = train_loader
        if validate_loader is None:
            assert post_process is not None and metric is not None
        self.validate_loader = validate_loader
        self.post_process = post_process
        self.metric = metric
        self.train_loader_len = len(train_loader)
        # model.preprocess.TPS_SpatialTransformerNetwork 对于tps 模块使用不同的学习率
        if config.arch.get('preprocess', False) and config.arch.preprocess.get('lr_scale', False):
            self.loc_lr = config.arch.preprocess.get('lr_scale') * self.lr
            params_list = filter_params_assign_lr(model, {'preprocess': self.loc_lr})
            self.optimizer = self._initialize('optimizer', torch.optim, params_list)
        if self.config['lr_scheduler']['type'] == 'WarmupPolyLR':
            warmup_iters = int(config['lr_scheduler']['args']['warmup_epoch'] * self.train_loader_len)
            if self.start_epoch > 1:
                self.config['lr_scheduler']['args']['last_epoch'] = (self.start_epoch - 1) * self.train_loader_len
            self.scheduler = WarmupPolyLR(self.optimizer, max_iters=self.epochs * self.train_loader_len,
                                          warmup_iters=warmup_iters, **config['lr_scheduler']['args'])
        if self.validate_loader is not None:
            self.logger_info(
                'train dataset has {} samples,{} in dataloader, validate dataset has {} samples,{} in dataloader'.format(
                    len(self.train_loader.dataset), self.train_loader_len, len(self.validate_loader.dataset), len(self.validate_loader)))
        else:
            self.logger_info('train dataset has {} samples,{} in dataloader'.format(len(self.train_loader.dataset), self.train_loader_len))

        self.converter = converter
        self.best_acc = 0


    def _train_epoch(self, epoch):
        self.model.train()
        epoch_start = time.time()
        batch_start = time.time()
        train_loss = 0.
        # running_metric_text = runningScore(2)
        lr = self.optimizer.param_groups[0]['lr']
        self.metric.reset()
        for i, batch in enumerate(self.train_loader):
            if i >= self.train_loader_len:
            # if i >= 1:
                break
            self.global_step += 1
            lr = self.optimizer.param_groups[0]['lr']
            # 解析label
            batch['text'], batch['length'] = self.converter.encode(batch['labels'])
            # 数据进行转换和丢到gpu
            for key, value in batch.items():
                if value is not None:
                    if isinstance(value, torch.Tensor):
                        batch[key] = value.to(self.device)
            cur_batch_size = batch['img'].size()[0]
            preds = self.model(batch['img'])
            loss_dict = self.criterion(preds, batch['text'], batch['length'], cur_batch_size)
            # backward
            self.optimizer.zero_grad()
            loss_dict['loss'].backward()
            self.optimizer.step()
            if self.config['lr_scheduler']['type'] == 'WarmupPolyLR':
                self.scheduler.step()

            # loss 和 acc 记录到日志
            loss_str = 'loss: {:.4f}, '.format(loss_dict['loss'].item())
            for idx, (key, value) in enumerate(loss_dict.items()):
                loss_dict[key] = value.item()
                if key == 'loss':
                    continue
                loss_str += '{}: {:.4f}'.format(key, loss_dict[key])
                if idx < len(loss_dict) - 1:
                    loss_str += ', '

            train_loss += loss_dict['loss']
            preds_prob = F.softmax(preds, dim=2)
            preds_prob, pred_index = preds_prob.max(dim=2)
            pred_str = self.converter.decode(pred_index)
            self.metric.measure(pred_str, batch['labels'], preds_prob)
            acc = self.metric.avg['acc']['true']
            edit_distance = self.metric.avg['edit']

            if self.global_step % self.log_iter == 0:
                batch_time = time.time() - batch_start
                self.logger_info(
                    '[{}/{}], [{}/{}], global_step: {}, speed: {:.1f} samples/sec, acc: {:.4f}, edit_distance: {:.4f}, {}, lr:{:.6}, time:{:.2f}'.format(
                        epoch, self.epochs, i + 1, self.train_loader_len, self.global_step, self.log_iter * cur_batch_size / batch_time, acc,
                        edit_distance, loss_str, lr, batch_time))
                batch_start = time.time()

            # if self.tensorboard_enable and self.config['local_rank'] == 0:
            #     # write tensorboard
            #     for key, value in loss_dict.items():
            #         self.writer.add_scalar('TRAIN/LOSS/{}'.format(key), value, self.global_step)
            #     self.writer.add_scalar('TRAIN/ACC_DIS/acc', acc, self.global_step)
            #     self.writer.add_scalar('TRAIN/ACC_DIS/edit_distance', edit_distance, self.global_step)
            #     self.writer.add_scalar('TRAIN/lr', lr, self.global_step)
            #     if self.global_step % self.show_images_iter == 0:
            #         # show images on tensorboard
            #         self.inverse_normalize(batch['img'])
            #         self.writer.add_images('TRAIN/imgs', batch['img'], self.global_step)
            #         # shrink_labels and threshold_labels
            #         shrink_labels = batch['labels']
            #         threshold_labels = batch['threshold_map']
            #         shrink_labels[shrink_labels <= 0.5] = 0
            #         shrink_labels[shrink_labels > 0.5] = 1
            #         show_label = torch.cat([shrink_labels, threshold_labels])
            #         show_label = vutils.make_grid(show_label.unsqueeze(1), nrow=cur_batch_size, normalize=False, padding=20, pad_value=1)
            #         self.writer.add_image('TRAIN/gt', show_label, self.global_step)
            #         # model output
            #         show_pred = []
            #         for kk in range(preds.shape[1]):
            #             show_pred.append(preds[:, kk, :, :])
            #         show_pred = torch.cat(show_pred)
            #         show_pred = vutils.make_grid(show_pred.unsqueeze(1), nrow=cur_batch_size, normalize=False, padding=20, pad_value=1)
            #         self.writer.add_image('TRAIN/preds', show_pred, self.global_step)
        return {'train_loss': train_loss / self.train_loader_len, 'lr': lr, 'time': time.time() - epoch_start,
                'epoch': epoch}

    def _eval(self, epoch):
        self.model.eval()
        total_frame = 0.0
        total_time = 0.0
        self.metric.reset()
        for i, batch in tqdm(enumerate(self.validate_loader), total=len(self.validate_loader), desc='test model'):
            with torch.no_grad():
                # 数据进行转换和丢到gpu
                for key, value in batch.items():
                    if value is not None:
                        if isinstance(value, torch.Tensor):
                            batch[key] = value.to(self.device)
                start = time.time()
                preds = self.model(batch['img'])
                preds_prob = F.softmax(preds, dim=2)
                preds_prob, pred_index = preds_prob.max(dim=2)
                pred_str = self.converter.decode(pred_index)
                self.metric.measure(pred_str, batch['labels'], preds_prob)
                total_frame += batch['img'].size()[0]
                total_time += time.time() - start
        acc = self.metric.avg['acc']['true']
        edit = self.metric.avg['edit']
        self.logger_info('FPS:{}'.format(total_frame / total_time))

        return acc, edit

    def _on_epoch_finish(self):
        self.logger_info('[{}/{}], train_loss: {:.4f}, time: {:.4f}, lr: {}'.format(
            self.epoch_result['epoch'], self.epochs, self.epoch_result['train_loss'], self.epoch_result['time'],
            self.epoch_result['lr']))
        net_save_path = '{}/model_latest.pth'.format(self.checkpoint_dir)

        if self.config['local_rank'] == 0:
            save_best = False
            if self.validate_loader is not None and self.metric is not None:  # 使用f1作为最优模型指标
                acc, edit = self._eval(self.epoch_result['epoch'])

                # if self.tensorboard_enable:
                #     self.writer.add_scalar('EVAL/recall', recall, self.global_step)
                #     self.writer.add_scalar('EVAL/precision', precision, self.global_step)
                #     self.writer.add_scalar('EVAL/hmean', hmean, self.global_step)
                self.logger_info('test: precision: {:.6f}, edit_distance: {:.4f}'.format(acc, edit))

                if acc >= self.best_acc:
                    self.best_acc = acc
                    save_best = True
            else:
                if self.epoch_result['train_loss'] <= self.metrics['train_loss']:
                    save_best = True
                    self.metrics['train_loss'] = self.epoch_result['train_loss']
            self._save_checkpoint(self.epoch_result['epoch'], net_save_path, save_best)

    def _on_train_finish(self):
        for k, v in self.metrics.items():
            self.logger_info('{}:{}'.format(k, v))
        self.logger_info('finish train')

    def inverse_normalize(self, batch_img):
        if self.UN_Normalize:
            batch_img[:, 0, :, :] = batch_img[:, 0, :, :] * self.normalize_std[0] + self.normalize_mean[0]
            batch_img[:, 1, :, :] = batch_img[:, 1, :, :] * self.normalize_std[1] + self.normalize_mean[1]
            batch_img[:, 2, :, :] = batch_img[:, 2, :, :] * self.normalize_std[2] + self.normalize_mean[2]

