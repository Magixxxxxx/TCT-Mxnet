import os
import time
import mxnet as mx

from utils.dataset import ImageTxtDataset
from mxnet import gluon, nd
from mxnet import autograd as ag

from utils.metric import BinaryAccMetric
import logging
from gluoncv.data.transforms import image as timage

def get_dataset(config):
    train_data = gluon.data.DataLoader(
        ImageTxtDataset(config.train_path).transform_first(config.transform_train),
        batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)

    val_data = gluon.data.DataLoader(
        ImageTxtDataset(config.val_path).transform_first(config.transform_test),
        batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    test_data = gluon.data.DataLoader(
        ImageTxtDataset(config.test_path).transform_first(config.transform_test),
        batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    return train_data, test_data, val_data


################################################################################

def validate(net, val_data, ctx, epoch, config):
    # metric = mx.metric.Accuracy()
    metric = BinaryAccMetric(config)
    # net.hybridize()

    for i, batch in enumerate(val_data):
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
        outputs = [net(X) for X in data]
        # tp_outputs = [net(X).asnumpy() for X in data]
        # np.save('./tp_outputs.npy', np.concatenate(tp_outputs, axis=0))
        # print('----------------'+ str(i) + '---------------')
        # print(outputs)
        metric.update(label, outputs)

    path = os.path.join('./OUTPUT', 'bicls-resnet50')
    if not os.path.exists(path):
        os.makedirs(path)

    _, val_acc = metric.get()
    prefix = os.path.join(path, 'model')
    #net.export(path='{}_{:.4f}'.format(prefix, val_acc), epoch=epoch)

    return metric.get()


def train(finetune_net, train_data, val_data, config):
    # set logging
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_file_path = './OUTPUT/output.log'
    log_dir = os.path.dirname(log_file_path)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    fh = logging.FileHandler(log_file_path)
    logger.addHandler(fh)
    
    # training
    trainer = gluon.Trainer(finetune_net.collect_params(select='.*dense2_'), 'sgd', {
        'learning_rate': config.lr, 'momentum': config.momentum, 'wd': config.wd})
    # metric = mx.metric.Accuracy()
    metric = BinaryAccMetric(config)
    L = gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=True)

    # L = gluon.loss.SoftmaxCrossEntropyLoss()

    ################################################################################
    # Training Loop

    lr_counter = 0
    num_batch = len(train_data)

    for epoch in range(config.epochs):
        if epoch == config.lr_steps[lr_counter]:
            trainer.set_learning_rate(trainer.learning_rate * config.lr_factor)
            logger.info('set learning rate to: {}'.format(trainer.learning_rate))
            lr_counter += 1

        tic = time.time()
        train_loss = 0
        metric.reset()

        for i, batch in enumerate(train_data):
            data = gluon.utils.split_and_load(batch[0], ctx_list=config.ctx, batch_axis=0, even_split=False)
            label = gluon.utils.split_and_load(batch[1], ctx_list=config.ctx, batch_axis=0, even_split=False)
            with ag.record():
                outputs = [finetune_net(X) for X in data]
                loss = [L(yhat, y) for yhat, y in zip(outputs, label)]
            for l in loss:
                l.backward()

            trainer.step(config.batch_size, ignore_stale_grad=True)
            tp_loss = sum([l.mean().asscalar() for l in loss]) / len(loss)
            train_loss += tp_loss

            metric.update(label, outputs)

            if i % 50 == 0:
                logger.info('{}-th batch/Epoch:{} ing--metric:{}--train_loss:{}'.format(i, epoch, metric.get()[1], tp_loss))

        _, train_acc = metric.get()
        train_loss /= num_batch

        finetune_net.save_parameters('./OUTPUT/{:04d}.params'.format(epoch))
        _, val_acc = validate(finetune_net, val_data, config.ctx, epoch, config)

        logger.info('[Epoch %d] Train-acc: %.3f, loss: %.3f | Val-acc: %.3f | time: %.1f' %
                    (epoch, train_acc, train_loss, val_acc, time.time() - tic))


if __name__ == '__main__':
    from utils.config import Config

    config = Config()
    # create model
    from utils.backbone import faster_rcnn_fpn_resnet50_v1b_coco

    CLASSES = ['normal', 'ascus', 'asch', 'lsil', 'hsil_scc_omn', 'agc_adenocarcinoma_em',
               'vaginalis', 'monilia', 'dysbacteriosis_herpes_act', 'ec']

    # create model
    finetune_net = faster_rcnn_fpn_resnet50_v1b_coco(classes=CLASSES, root='./models', pretrained_base=False,
                                                     per_device_batch_size=1)
    # ==================init====================

    finetune_net.load_parameters('./OUTPUT/bclsfaster_rcnn_fpn_resnet50_v1b_coco_best.params', allow_missing=True,
                                 ignore_extra=True)
    # train_patterns = '.*dense'
    # print(finetune_net.collect_params()['fasterrcnn0_dense2_weight'].data())
    # print(finetune_net.collect_params()['fasterrcnn0_dense2_bias'].data())
    finetune_net.collect_params().reset_ctx(config.ctx)
    finetune_net.hybridize()

    print(finetune_net)
    # load data
    train_data, test_data, val_data = get_dataset(config=config)

    # train
    train(finetune_net, train_data=train_data, val_data=val_data, config=config)
