"""Train Faster-RCNN end to end."""
import argparse
import os

# disable autotune


os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
import logging
import time
import numpy as np
import mxnet as mx
from mxnet import gluon
from mxnet import autograd
from mxnet.contrib import amp
import gluoncv as gcv
from gluoncv import utils as gutils
from gluoncv.data.batchify import FasterRCNNTrainBatchify, Tuple, Append
from gluoncv.data.transforms.presets.rcnn import FasterRCNNDefaultTrainTransform, \
    FasterRCNNDefaultValTransform
from gluoncv.utils.metrics.coco_detection import COCODetectionMetric
from gluoncv.utils.parallel import Parallelizable, Parallel
from gluoncv.utils.metrics.rcnn import RPNAccMetric, RPNL1LossMetric, RCNNAccMetric, \
    RCNNL1LossMetric

try:
    import horovod.mxnet as hvd
except ImportError:
    hvd = None

# import sys

# reload(sys)

# sys.setdefaultencoding('utf8')

def parse_args():
    parser = argparse.ArgumentParser(description='Train Faster-RCNN networks e2e.')
    parser.add_argument('--network', type=str, default='resnet50_v1b',
                        help="Base network name which serves as feature extraction base.")
    parser.add_argument('--dataset', type=str, default='voc',
                        help='Training dataset. Now support voc and coco.')
    parser.add_argument('--details', type=str, default='coco',
                        help='details dataset')
    parser.add_argument('--num-workers', '-j', dest='num_workers', type=int,
                        default=4, help='Number of data workers, you can use larger '
                                        'number to accelerate data loading, '
                                        'if your CPU and GPUs are powerful.')
    parser.add_argument('--batch-size', type=int, default=8, help='Training mini-batch size.')
    parser.add_argument('--gpus', type=str, default='0',
                        help='Training with GPUs, you can specify 1,3 for example.')
    parser.add_argument('--epochs', type=str, default='',
                        help='Training epochs.')
    parser.add_argument('--resume', type=str, default='',
                        help='Resume from previously saved parameters if not None. '
                             'For example, you can resume from ./faster_rcnn_xxx_0123.params')
    parser.add_argument('--start-epoch', type=int, default=0,
                        help='Starting epoch for resuming, default is 0 for new training.'
                             'You can specify it to 100 for example to start from 100 epoch.')
    parser.add_argument('--lr', type=str, default='',
                        help='Learning rate, default is 0.001 for voc single gpu training.')
    parser.add_argument('--lr-decay', type=float, default=0.1,
                        help='decay rate of learning rate. default is 0.1.')
    parser.add_argument('--lr-decay-epoch', type=str, default='',
                        help='epochs at which learning rate decays. default is 14,20 for voc.')
    parser.add_argument('--lr-warmup', type=str, default='',
                        help='warmup iterations to adjust learning rate, default is 0 for voc.')
    parser.add_argument('--lr-warmup-factor', type=float, default=1. / 3.,
                        help='warmup factor of base lr.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum, default is 0.9')
    parser.add_argument('--wd', type=str, default='',
                        help='Weight decay, default is 5e-4 for voc')
    parser.add_argument('--log-interval', type=int, default=100,
                        help='Logging mini-batch interval. Default is 100.')
    parser.add_argument('--save-prefix', type=str, default='',
                        help='Saving parameter prefix')
    parser.add_argument('--save-interval', type=int, default=1,
                        help='Saving parameters epoch interval, best model will always be saved.')
    parser.add_argument('--val-interval', type=int, default=1,
                        help='Epoch interval for validation, increase the number will reduce the '
                             'training time if validation is slow.')
    parser.add_argument('--seed', type=int, default=233,
                        help='Random seed to be fixed.')
    parser.add_argument('--verbose', dest='verbose', action='store_true',
                        help='Print helpful debugging info once set.')
    parser.add_argument('--mixup', action='store_true', help='Use mixup training.')
    parser.add_argument('--no-mixup-epochs', type=int, default=20,
                        help='Disable mixup training if enabled in the last N epochs.')

    # Norm layer options
    parser.add_argument('--norm-layer', type=str, default=None,
                        help='Type of normalization layer to use. '
                             'If set to None, backbone normalization layer will be fixed,'
                             ' and no normalization layer will be used. '
                             'Currently supports \'bn\', and None, default is None.'
                             'Note that if horovod is enabled, sync bn will not work correctly.')

    # FPN options
    parser.add_argument('--use-fpn', action='store_true',
                        help='Whether to use feature pyramid network.')

    # Performance options
    parser.add_argument('--disable-hybridization', action='store_true',
                        help='Whether to disable hybridize the model. '
                             'Memory usage and speed will decrese.')
    parser.add_argument('--static-alloc', action='store_true',
                        help='Whether to use static memory allocation. Memory usage will increase.')
    parser.add_argument('--amp', action='store_true',
                        help='Use MXNet AMP for mixed precision training.')
    parser.add_argument('--horovod', action='store_true',
                        help='Use MXNet Horovod for distributed training. Must be run with OpenMPI. '
                             '--gpus is ignored when using --horovod.')
    parser.add_argument('--executor-threads', type=int, default=1,
                        help='Number of threads for executor for scheduling ops. '
                             'More threads may incur higher GPU memory footprint, '
                             'but may speed up throughput. Note that when horovod is used, '
                             'it is set to 1.')
    parser.add_argument('--kv-store', type=str, default='nccl',
                        help='KV store options. local, device, nccl, dist_sync, dist_device_sync, '
                             'dist_async are available.')

    args = parser.parse_args()

    if args.horovod:
        if hvd is None:
            raise SystemExit("Horovod not found, please check if you installed it correctly.")
        hvd.init()

    if args.dataset == 'voc':
        args.epochs = int(args.epochs) if args.epochs else 20
        args.lr_decay_epoch = args.lr_decay_epoch if args.lr_decay_epoch else '14,20'
        args.lr = float(args.lr) if args.lr else 0.001
        args.lr_warmup = args.lr_warmup if args.lr_warmup else -1
        args.wd = float(args.wd) if args.wd else 5e-4
    elif args.dataset == 'coco':
        args.epochs = int(args.epochs) if args.epochs else 26
        args.lr_decay_epoch = args.lr_decay_epoch if args.lr_decay_epoch else '17,23'
        args.lr = float(args.lr) if args.lr else 0.01
        args.lr_warmup = args.lr_warmup if args.lr_warmup else 1000
        args.wd = float(args.wd) if args.wd else 1e-4
    return args


from utils.data.DATA import DATADetection


def get_dataset(dataset, args):
    if dataset.lower() == 'coco':
        if args.details == 'TCT':
            details = 'TCT'

            class TCTDetection(DATADetection):
                CLASSES = ['normal', 'ascus', 'asch', 'lsil', 'hsil_scc_omn', 'agc_adenocarcinoma_em',
                           'vaginalis', 'monilia', 'dysbacteriosis_herpes_act', 'ec']

            DatasetDetection = TCTDetection
        elif args.details == 'ury_cocostyle':
            details = 'ury_cocostyle'

            class URYDetection(DATADetection):
                CLASSES = ['eryth', 'leuko', 'epith', 'epithn',
                           'trichomonad', 'cluecell', 'spore', 'hypha']

            DatasetDetection = URYDetection
        elif args.details == 'voc_cocostyle':
            details = 'voc_cocostyle'

            class VOCDetection(DATADetection):
                CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
                           'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
                           'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

            DatasetDetection = VOCDetection
        else:
            details = None
            DatasetDetection = None

        train_dataset = DatasetDetection(root='./datasets/' + details, splits='train')
        val_dataset = DatasetDetection(root='./datasets/' + details, splits='test', skip_empty=False)
        val_metric = COCODetectionMetric(val_dataset, args.save_prefix + '_eval', cleanup=True)
    else:
        raise NotImplementedError('Dataset: {} not implemented.'.format(dataset))
    return train_dataset, val_dataset, val_metric


def get_dataloader(net, train_dataset, val_dataset, train_transform, val_transform, batch_size, num_shards, args):
    """Get dataloader."""
    train_bfn = FasterRCNNTrainBatchify(net, num_shards)
    train_sampler = \
        gcv.nn.sampler.SplitSortedBucketSampler(train_dataset.get_im_aspect_ratio(),
                                                batch_size,
                                                num_parts=hvd.size() if args.horovod else 1,
                                                part_index=hvd.rank() if args.horovod else 0,
                                                shuffle=True)
    train_loader = mx.gluon.data.DataLoader(train_dataset.transform(
        train_transform(net.short, net.max_size, net, mean=(0.783, 0.799, 0.841), std=(0.053, 0.042, 0.031), ashape=net.ashape, multi_stage=args.use_fpn)),
        batch_sampler=train_sampler, batchify_fn=train_bfn, num_workers=args.num_workers)
    val_bfn = Tuple(*[Append() for _ in range(3)])
    short = net.short[-1] if isinstance(net.short, (tuple, list)) else net.short
    # validation use 1 sample per device
    val_loader = mx.gluon.data.DataLoader(
        val_dataset.transform(val_transform(short, net.max_size, mean=(0.783, 0.799, 0.841), std=(0.053, 0.042, 0.031))), num_shards, False,
        batchify_fn=val_bfn, last_batch='keep', num_workers=args.num_workers)
    return train_loader, val_loader


def save_params(net, logger, best_map, current_map, epoch, save_interval, prefix):
    current_map = float(current_map)
    if current_map > best_map[0]:
        logger.info('[Epoch {}] mAP {} higher than current best {} saving to {}'.format(
            epoch, current_map, best_map, '{:s}_best.params'.format(prefix)))
        best_map[0] = current_map
        net.save_parameters('{:s}_best.params'.format(prefix))
        with open(prefix + '_best_map.log', 'a') as f:
            f.write('{:04d}:\t{:.4f}\n'.format(epoch, current_map))
    if save_interval and (epoch + 1) % save_interval == 0:
        logger.info('[Epoch {}] Saving parameters to {}'.format(
            epoch, '{:s}_{:04d}_{:.4f}.params'.format(prefix, epoch, current_map)))
        net.save_parameters('{:s}_{:04d}_{:.4f}.params'.format(prefix, epoch, current_map))


def split_and_load(batch, ctx_list):
    """Split data to 1 batch each device."""
    new_batch = []
    for i, data in enumerate(batch):
        if isinstance(data, (list, tuple)):
            new_data = [x.as_in_context(ctx) for x, ctx in zip(data, ctx_list)]
        else:
            new_data = [data.as_in_context(ctx_list[0])]
        new_batch.append(new_data)
    return new_batch


def validate(net, val_data, ctx, eval_metric, epoch, args):
    """Test on validation dataset."""
    clipper = gcv.nn.bbox.BBoxClipToImage()
    eval_metric.reset()
    # binary cls metric
    from utils.metric.binary_cls import BinaryAccMetric, PrecisionMetric, RecallMetric
    binary_acc = BinaryAccMetric()
    precision = PrecisionMetric()
    recall = RecallMetric()

    if not args.disable_hybridization:
        # input format is differnet than training, thus rehybridization is needed.
        net.hybridize(static_alloc=args.static_alloc)
    for batch in val_data:
        batch = split_and_load(batch, ctx_list=ctx)
        det_bboxes = []
        det_ids = []
        det_scores = []
        gt_bboxes = []
        gt_ids = []
        gt_difficults = []
        for x, y, im_scale in zip(*batch):
            # get prediction results, from sigmoid binary_cls
            ids, scores, bboxes, binary_cls = net(x)
            det_ids.append(ids)
            det_scores.append(scores)
            # clip to image size
            det_bboxes.append(clipper(bboxes, x))
            # rescale to original resolution
            im_scale = im_scale.reshape((-1)).asscalar()
            det_bboxes[-1] *= im_scale
            # split ground truths
            gt_ids.append(y.slice_axis(axis=-1, begin=4, end=5))
            gt_bboxes.append(y.slice_axis(axis=-1, begin=0, end=4))
            gt_bboxes[-1] *= im_scale
            gt_difficults.append(y.slice_axis(axis=-1, begin=5, end=6) if y.shape[-1] > 5 else None)

        # update metric
        for det_bbox, det_id, det_score, gt_bbox, gt_id, gt_diff in zip(det_bboxes, det_ids,
                                                                        det_scores, gt_bboxes,
                                                                        gt_ids, gt_difficults):
            eval_metric.update(det_bbox, det_id, det_score, gt_bbox, gt_id, gt_diff)

        # -----------binary acc s-------------------
        count = y[:, :, 4:5]
        tp = count.squeeze()
        if 1 in tp or 2 in tp or 3 in tp or 4 in tp or 5 in tp:
            lb = mx.nd.array([1.0], ctx=count.context)
        else:
            lb = mx.nd.array([0.0], ctx=count.context)

        binary_cls = binary_cls.squeeze(axis=-1)
        binary_acc.update(lb, binary_cls)
        # -----------binary acc e-------------------

    if not os.path.exists(os.path.join(args.save_prefix, 'json')):
        os.makedirs(os.path.join(args.save_prefix, 'json'))

    root_path = os.path.join(args.save_prefix, 'json', 'binary-cls-L-large-data')
    net.export(path=root_path, epoch=epoch)

    return eval_metric.get(), binary_acc.get(), precision.get(), recall.get()


def get_lr_at_iter(alpha, lr_warmup_factor=1. / 3.):
    return lr_warmup_factor * (1 - alpha) + alpha


class ForwardBackwardTask(Parallelizable):
    def __init__(self, net, optimizer, rpn_cls_loss, rpn_box_loss, rcnn_cls_loss, rcnn_box_loss, binary_cls_loss
                 , mix_ratio):
        super(ForwardBackwardTask, self).__init__()
        self.net = net
        self._optimizer = optimizer
        self.rpn_cls_loss = rpn_cls_loss
        self.rpn_box_loss = rpn_box_loss
        self.rcnn_cls_loss = rcnn_cls_loss
        self.rcnn_box_loss = rcnn_box_loss
        self.binary_cls_loss = binary_cls_loss
        self.mix_ratio = mix_ratio

    def forward_backward(self, x):
        data, label, rpn_cls_targets, rpn_box_targets, rpn_box_masks = x
        with autograd.record():
            gt_label = label[:, :, 4:5]
            gt_box = label[:, :, :4]
            cls_pred, box_pred, roi, samples, matches, rpn_score, rpn_box, anchors, binary_cls = net(
                data, gt_box)
            # losses of rpn
            rpn_score = rpn_score.squeeze(axis=-1)
            num_rpn_pos = (rpn_cls_targets >= 0).sum()
            rpn_loss1 = self.rpn_cls_loss(rpn_score, rpn_cls_targets,
                                          rpn_cls_targets >= 0) * rpn_cls_targets.size / num_rpn_pos
            rpn_loss2 = self.rpn_box_loss(rpn_box, rpn_box_targets,
                                          rpn_box_masks) * rpn_box.size / num_rpn_pos
            # rpn overall loss, use sum rather than average
            rpn_loss = rpn_loss1 + rpn_loss2
            # generate targets for rcnn
            cls_targets, box_targets, box_masks = self.net.target_generator(roi, samples,
                                                                            matches, gt_label,
                                                                            gt_box)
            # losses of rcnn
            num_rcnn_pos = (cls_targets >= 0).sum()
            rcnn_loss1 = self.rcnn_cls_loss(cls_pred, cls_targets,
                                            cls_targets.expand_dims(-1) >= 0) * cls_targets.size / \
                         num_rcnn_pos
            rcnn_loss2 = self.rcnn_box_loss(box_pred, box_targets, box_masks) * box_pred.size / \
                         num_rcnn_pos
            rcnn_loss = rcnn_loss1 + rcnn_loss2

            # =============binary cls loss, TO DO.....===========
            count = label[:, :, 4:5]
            tp = count.squeeze()[0]
            # if 2 in tp or 3 in tp or 4 in tp or 5 in tp or 6 in tp or 7 in tp:

            if 1 in tp or 2 in tp or 3 in tp or 4 in tp or 5 in tp:
                cls_label = mx.nd.array([1.0], ctx=count.context)
            else:
                cls_label = mx.nd.array([0.0], ctx=count.context)

            binary_cls = binary_cls.squeeze(axis=-1)
            binary_loss = self.binary_cls_loss(binary_cls, cls_label)
            # ===========binary cls loss =================

            # overall losses
            total_loss = rpn_loss.sum() * self.mix_ratio + rcnn_loss.sum() * self.mix_ratio
            #total_loss = rpn_loss.sum() * self.mix_ratio + rcnn_loss.sum() * self.mix_ratio + binary_loss * 0.3

            rpn_loss1_metric = rpn_loss1.mean() * self.mix_ratio
            rpn_loss2_metric = rpn_loss2.mean() * self.mix_ratio
            rcnn_loss1_metric = rcnn_loss1.mean() * self.mix_ratio
            rcnn_loss2_metric = rcnn_loss2.mean() * self.mix_ratio
            binary_loss_metric = binary_loss.mean() * 0.2
            rpn_acc_metric = [[rpn_cls_targets, rpn_cls_targets >= 0], [rpn_score]]
            rpn_l1_loss_metric = [[rpn_box_targets, rpn_box_masks], [rpn_box]]
            rcnn_acc_metric = [[cls_targets], [cls_pred]]
            rcnn_l1_loss_metric = [[box_targets, box_masks], [box_pred]]

            if args.amp:
                with amp.scale_loss(total_loss, self._optimizer) as scaled_losses:
                    autograd.backward(scaled_losses)
            else:
                total_loss.backward()

        return rpn_loss1_metric, rpn_loss2_metric, rcnn_loss1_metric, rcnn_loss2_metric, \
               binary_loss_metric, \
               rpn_acc_metric, rpn_l1_loss_metric, rcnn_acc_metric, rcnn_l1_loss_metric


def train(net, train_data, val_data, eval_metric, batch_size, ctx, args):
    """Training pipeline"""
    kv = mx.kvstore.create(args.kv_store)
    net.collect_params().setattr('grad_req', 'null')
    net.collect_train_params().setattr('grad_req', 'write')
    optimizer_params = {'learning_rate': args.lr, 'wd': args.wd, 'momentum': args.momentum}
    if args.horovod:
        hvd.broadcast_parameters(net.collect_params(), root_rank=0)
        trainer = hvd.DistributedTrainer(
            net.collect_train_params(),  # fix batchnorm, fix first stage, etc...
            'sgd',
            optimizer_params)
    else:
        trainer = gluon.Trainer(
            net.collect_train_params(),  # fix batchnorm, fix first stage, etc...
            'sgd',
            optimizer_params,
            update_on_kvstore=(False if args.amp else None), kvstore=kv)

    if args.amp:
        amp.init_trainer(trainer)

    # lr decay policy
    lr_decay = float(args.lr_decay)
    lr_steps = sorted([float(ls) for ls in args.lr_decay_epoch.split(',') if ls.strip()])
    lr_warmup = float(args.lr_warmup)  # avoid int division

    # TODO(zhreshold) losses?
    rpn_cls_loss = mx.gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=False)
    rpn_box_loss = mx.gluon.loss.HuberLoss(rho=1 / 9.)  # == smoothl1
    rcnn_cls_loss = mx.gluon.loss.SoftmaxCrossEntropyLoss()
    rcnn_box_loss = mx.gluon.loss.HuberLoss()  # == smoothl1

    # -------------------YZH binary_cls loss-----------------
    binary_cls_loss = mx.gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=True)
    # -------------------YZH binary_cls loss-----------------

    metrics = [mx.metric.Loss('RPN_Conf'),
               mx.metric.Loss('RPN_SmoothL1'),
               mx.metric.Loss('RCNN_CrossEntropy'),
               mx.metric.Loss('RCNN_SmoothL1'),
               mx.metric.Loss('Binary_Conf')]

    rpn_acc_metric = RPNAccMetric()
    rpn_bbox_metric = RPNL1LossMetric()
    rcnn_acc_metric = RCNNAccMetric()
    rcnn_bbox_metric = RCNNL1LossMetric()
    metrics2 = [rpn_acc_metric, rpn_bbox_metric, rcnn_acc_metric, rcnn_bbox_metric]

    # set up logger
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_file_path = args.save_prefix + '_train.log'
    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    fh = logging.FileHandler(log_file_path)
    logger.addHandler(fh)
    logger.info(args)
    if args.verbose:
        logger.info('Trainable parameters:')
        logger.info(net.collect_train_params().keys())
    logger.info('Start training from [Epoch {}]'.format(args.start_epoch))
    best_map = [0]
    for epoch in range(args.start_epoch, args.epochs):
        mix_ratio = 1.0
        if not args.disable_hybridization:
            net.hybridize(static_alloc=args.static_alloc)
        rcnn_task = ForwardBackwardTask(net, trainer, rpn_cls_loss, rpn_box_loss, rcnn_cls_loss,
                                        rcnn_box_loss, binary_cls_loss, mix_ratio=1.0)
        executor = Parallel(args.executor_threads, rcnn_task) if not args.horovod else None
        if args.mixup:
            # TODO(zhreshold) only support evenly mixup now, target generator needs to be modified otherwise
            train_data._dataset._data.set_mixup(np.random.uniform, 0.5, 0.5)
            mix_ratio = 0.5
            if epoch >= args.epochs - args.no_mixup_epochs:
                train_data._dataset._data.set_mixup(None)
                mix_ratio = 1.0
        while lr_steps and epoch >= lr_steps[0]:
            new_lr = trainer.learning_rate * lr_decay
            lr_steps.pop(0)
            trainer.set_learning_rate(new_lr)
            logger.info("[Epoch {}] Set learning rate to {}".format(epoch, new_lr))
        for metric in metrics:
            metric.reset()
        tic = time.time()
        btic = time.time()
        base_lr = trainer.learning_rate
        rcnn_task.mix_ratio = mix_ratio
        for i, batch in enumerate(train_data):
            if epoch == 0 and i <= lr_warmup:
                # adjust based on real percentage
                new_lr = base_lr * get_lr_at_iter(i / lr_warmup, args.lr_warmup_factor)
                if new_lr != trainer.learning_rate:
                    if i % args.log_interval == 0:
                        logger.info(
                            '[Epoch 0 Iteration {}] Set learning rate to {}'.format(i, new_lr))
                    trainer.set_learning_rate(new_lr)
            batch = split_and_load(batch, ctx_list=ctx)
            metric_losses = [[] for _ in metrics]
            add_losses = [[] for _ in metrics2]
            if executor is not None:
                for data in zip(*batch):
                    executor.put(data)
            for j in range(len(ctx)):
                if executor is not None:
                    result = executor.get()
                else:
                    result = rcnn_task.forward_backward(list(zip(*batch))[0])
                if (not args.horovod) or hvd.rank() == 0:
                    for k in range(len(metric_losses)):
                        metric_losses[k].append(result[k])
                    for k in range(len(add_losses)):
                        add_losses[k].append(result[len(metric_losses) + k])
            # len(metrics) = 5, len(metrics2) =4
            for metric, record in zip(metrics, metric_losses):
                metric.update(0, record)
            for metric, records in zip(metrics2, add_losses):
                for pred in records:
                    metric.update(pred[0], pred[1])
            trainer.step(batch_size)

            # update metrics
            if (not args.horovod or hvd.rank() == 0) and args.log_interval \
                    and not (i + 1) % args.log_interval:
                msg = ','.join(
                    ['{}={:.3f}'.format(*metric.get()) for metric in metrics + metrics2])
                logger.info('[Epoch {}][Batch {}], Speed: {:.3f} samples/sec, {}'.format(
                    epoch, i, args.log_interval * args.batch_size / (time.time() - btic), msg))
                btic = time.time()

        if (not args.horovod) or hvd.rank() == 0:
            msg = ','.join(['{}={:.3f}'.format(*metric.get()) for metric in metrics])
            logger.info('[Epoch {}] Training cost: {:.3f}, {}'.format(
                epoch, (time.time() - tic), msg))
            if not (epoch + 1) % args.val_interval:
                # consider reduce the frequency of validation to save time
                (map_name, mean_ap), binary_acc, precision, recall = validate(net, val_data, ctx, eval_metric, epoch, args)
                val_msg = '\n'.join(['{}={}'.format(k, v) for k, v in zip(map_name, mean_ap)])
                logger.info('[Epoch {}] Validation: \n{}'.format(epoch, val_msg))
                logger.info('binary_acc---->{} \n precision---->{} \n recall---->{}'.format(binary_acc, precision, recall))
                current_map = float(mean_ap[-1])
            else:
                current_map = 0.
            save_params(net, logger, best_map, current_map, epoch, args.save_interval,
                        args.save_prefix)
        executor.__del__()

import copy
from random import randint
from gluoncv.data.transforms import bbox as tbbox
from gluoncv.data.transforms import image as timage
from gluoncv.data.transforms import mask as tmask

class myTrainTransform(FasterRCNNDefaultTrainTransform):
    def __call__(self, src, label):
        """Apply transform to training image/label."""
        # resize shorter side but keep in max_size
        h, w, _ = src.shape
        if self._random_resize:
            short = randint(self._short[0], self._short[1])
        else:
            short = self._short
        img = timage.resize_short_within(src, short, self._max_size, interp=1)
        bbox = tbbox.resize(label, (w, h), (img.shape[1], img.shape[0]))

        # random horizontal flip
        h, w, _ = img.shape
        img, flips = timage.random_flip(img, px=self._flip_p)
        bbox = tbbox.flip(bbox, (w, h), flip_x=flips[0])

        # to tensor
        img = mx.nd.image.to_tensor(img)

        #CHANGE:normalize each img
        img_np = img.asnumpy()
        self._mean = tuple(img_np.mean(axis = (1,2)))
        self._std = tuple(img_np.std(axis = (1,2)))

        img = mx.nd.image.normalize(img, mean=self._mean, std=self._std)

        if self._anchors is None:
            return img, bbox.astype(img.dtype)

        # generate RPN target so cpu workers can help reduce the workload
        # feat_h, feat_w = (img.shape[1] // self._stride, img.shape[2] // self._stride)
        gt_bboxes = mx.nd.array(bbox[:, :4])
        if self._multi_stage:
            anchor_targets = []
            oshapes = []
            cls_targets, box_targets, box_masks = [], [], []
            for anchor, feat_sym in zip(self._anchors, self._feat_sym):
                oshape = feat_sym.infer_shape(data=(1, 3, img.shape[1], img.shape[2]))[1][0]
                anchor = anchor[:, :, :oshape[2], :oshape[3], :]
                oshapes.append(anchor.shape)
                anchor_targets.append(anchor.reshape((-1, 4)))
            anchor_targets = mx.nd.concat(*anchor_targets, dim=0)
            cls_target, box_target, box_mask = self._target_generator(
                gt_bboxes, anchor_targets, img.shape[2], img.shape[1])
            start_ind = 0
            for oshape in oshapes:
                size = oshape[2] * oshape[3] * (oshape[4] // 4)
                lvl_cls_target = cls_target[start_ind:start_ind + size] \
                    .reshape(oshape[2], oshape[3], -1)
                lvl_box_target = box_target[start_ind:start_ind + size] \
                    .reshape(oshape[2], oshape[3], -1)
                lvl_box_mask = box_mask[start_ind:start_ind + size] \
                    .reshape(oshape[2], oshape[3], -1)
                start_ind += size
                cls_targets.append(lvl_cls_target)
                box_targets.append(lvl_box_target)
                box_masks.append(lvl_box_mask)
        else:
            oshape = self._feat_sym.infer_shape(data=(1, 3, img.shape[1], img.shape[2]))[1][0]
            anchor = self._anchors[:, :, :oshape[2], :oshape[3], :]
            oshape = anchor.shape
            cls_target, box_target, box_mask = self._target_generator(
                gt_bboxes, anchor.reshape((-1, 4)), img.shape[2], img.shape[1])
            size = oshape[2] * oshape[3] * (oshape[4] // 4)
            cls_targets = [cls_target[0:size].reshape(oshape[2], oshape[3], -1)]
            box_targets = [box_target[0:size].reshape(oshape[2], oshape[3], -1)]
            box_masks = [box_mask[0:size].reshape(oshape[2], oshape[3], -1)]
        return img, bbox.astype(img.dtype), cls_targets, box_targets, box_masks

class myValTransform(FasterRCNNDefaultValTransform):
    """docstring for ClassName"""

    def __call__(self, src, label):
        """Apply transform to validation image/label."""
        # resize shorter side but keep in max_size
        h, w, _ = src.shape
        img = timage.resize_short_within(src, self._short, self._max_size, interp=1)
        # no scaling ground-truth, return image scaling ratio instead
        bbox = tbbox.resize(label, (w, h), (img.shape[1], img.shape[0]))
        im_scale = h / float(img.shape[0])

        img = mx.nd.image.to_tensor(img)

        #TODO
        img_np = img.asnumpy()
        self._mean = tuple(img_np.mean(axis = (1,2)))
        self._std = tuple(img_np.std(axis = (1,2)))

        img = mx.nd.image.normalize(img, mean=self._mean, std=self._std)
        return img, bbox.astype('float32'), mx.nd.array([im_scale])        

if __name__ == '__main__':
    import sys

    sys.setrecursionlimit(1100)
    args = parse_args()
    # fix seed for mxnet, numpy and python builtin random generator.
    gutils.random.seed(args.seed)

    if args.amp:
        amp.init()

    # training contexts
    if args.horovod:
        ctx = [mx.gpu(hvd.local_rank())]
    else:
        ctx = [mx.gpu(int(i)) for i in args.gpus.split(',') if i.strip()]
        ctx = ctx if ctx else [mx.cpu()]

    # network
    kwargs = {}
    module_list = []
    if args.use_fpn:
        module_list.append('fpn')
    if args.norm_layer is not None:
        module_list.append(args.norm_layer)
        if args.norm_layer == 'bn':
            kwargs['num_devices'] = len(args.gpus.split(','))
    net_name = '_'.join(('faster_rcnn', *module_list, args.network, args.dataset))
    args.save_prefix += net_name

    # from gluoncv.model_zoo import get_model
    # net = get_model(net_name, pretrained_base=True,
    #                 per_device_batch_size=args.batch_size // len(ctx), **kwargs)
    """
    classes_name_to_id = {
        "normal": 1,
        "ascus": 2,
        "asch": 3,
        "lsil": 4,
        "hsil_scc_omn": 5,
        "agc_adenocarcinoma_em": 6,
        "vaginalis": 7,
        "monilia": 8,
        "dysbacteriosis_herpes_act": 9,
        "ec": 10
    }
    """
    if args.details == 'TCT':
        # ascu-->asch
        classes = ['normal', 'ascus', 'asch', 'lsil', 'hsil_scc_omn', 'agc_adenocarcinoma_em',
                   'vaginalis', 'monilia', 'dysbacteriosis_herpes_act', 'ec']
        # classes = ['foreground']
    elif args.details == 'ury_cocostyle':
        classes = ['eryth', 'leuko', 'epith', 'epithn', 'trichomonad', 'cluecell', 'spore', 'hypha']

    elif args.details == 'voc_cocostyle':
        classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
                   'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
                   'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

    else:
        classes = None

    """
    from utils.model.backbone import faster_rcnn_fpn_resnet101_v1d_coco
    net = faster_rcnn_fpn_resnet101_v1d_coco(classes, root='./models', pretrained_base=True,
                                             per_device_batch_size=args.batch_size // len(ctx), **kwargs)
    """
    from utils.model.backbone import faster_rcnn_fpn_resnet50_v1b_coco
    net = faster_rcnn_fpn_resnet50_v1b_coco(classes, root='./models', pretrained_base=True,
                                             per_device_batch_size=args.batch_size // len(ctx), **kwargs)



    if args.resume.strip():
        net.load_parameters(args.resume.strip(), allow_missing=True, ignore_extra=True)
        # load_parameters source code modify

        # from mxnet import ndarray

        # loaded = ndarray.load(args.resume.strip())
        # params = net._collect_params_with_prefix()
        # for name in loaded:
        #     if 'cls_output' not in name and name in params:
        #         params[name]._load_init(loaded[name], ctx=mx.cpu(), cast_dtype=False, dtype_source='current')

    else:
        for param in net.collect_params().values():
            if param._data is not None:
                continue
            param.initialize()

    for param in net.collect_params().values():
        if param._data is not None:
            continue
        param.initialize()

    net.collect_params().reset_ctx(ctx)
    """
    train_patterns = '.*dense(2|3)_'
    print(net.collect_params(select=train_patterns))
    """
    # training data

    train_dataset, val_dataset, eval_metric = get_dataset(args.dataset, args)
    batch_size = args.batch_size // len(ctx) if args.horovod else args.batch_size



    train_data, val_data = get_dataloader(
        net, train_dataset, val_dataset, myTrainTransform,
        myValTransform, batch_size, len(ctx), args)

    # for batch in train_data:
    #     print('batch type is:', type(batch))
    #     print('batch len is :', len(batch))
    #     if isinstance(batch[0], list) or isinstance(batch[0], tuple):
    #         print('batch[0] len is: ', len(batch[0]))
    #         print('batch[1] len is', len(batch[1]))
    #     print(batch[0])
    #     break
    # for batch in train_dataset.transform(FasterRCNNDefaultTrainTransform(800, 1333, net, ashape=net.ashape, multi_stage=True)):
    #     print(type(batch))
    #     print(len(batch))
    #     print(batch[0])
    #     break

    # training
    train(net, train_data, val_data, eval_metric, batch_size, ctx, args)
