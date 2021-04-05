import os

os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
import mxnet as mx
from mxnet import autograd
from gluoncv.model_zoo.faster_rcnn.faster_rcnn import FasterRCNN
from gluoncv.nn.feature import FPNFeatureExpander
from mxnet.gluon import nn


class FasterRCNN_Addcls(FasterRCNN):
    def __init__(self, features, top_features, classes, train_patterns, box_features, roi_size, max_num_gt, **kwargs):
        super(FasterRCNN_Addcls, self).__init__(features=features, top_features=top_features, classes=classes,
                                                train_patterns=train_patterns, roi_size=roi_size,
                                                max_num_gt=max_num_gt,
                                                box_features=box_features, prefix='fasterrcnn0_', **kwargs)

        with self.name_scope():
            self.cls_output = nn.HybridSequential(prefix='')
            self.cls_output.add(nn.GlobalAvgPool2D(),
                                # nn.Dense(256, weight_initializer=mx.init.Normal(0.01)),
                                # nn.Activation('relu'),
                                nn.Dense(1, weight_initializer=mx.init.Normal(0.01), activation='sigmoid'))

    def hybrid_forward(self, F, x, gt_box=None):
        """Forward Faster-RCNN network.

        The behavior during training and inference is different.

        Parameters
        ----------
        x : mxnet.nd.NDArray or mxnet.symbol
            The network input tensor.
        gt_box : type, only required during training
            The ground-truth bbox tensor with shape (1, N, 4).

        Returns
        -------
        (ids, scores, bboxes)
            During inference, returns final class id, confidence scores, bounding
            boxes.

        """

        def _split(x, axis, num_outputs, squeeze_axis):
            x = F.split(x, axis=axis, num_outputs=num_outputs, squeeze_axis=squeeze_axis)
            if isinstance(x, list):
                return x
            else:
                return [x]

        feat = self.features(x)
        # binary classify, P5 layer
        binary_cls = self.cls_output(feat[3])
        
        if not isinstance(feat, (list, tuple)):
            feat = [feat]

        # RPN proposals
        if autograd.is_training():
            rpn_score, rpn_box, raw_rpn_score, raw_rpn_box, anchors = \
                self.rpn(F.zeros_like(x), *feat)
            rpn_box, samples, matches = self.sampler(rpn_box, rpn_score, gt_box)
        else:
            _, rpn_box = self.rpn(F.zeros_like(x), *feat)

        # create batchid for roi
        num_roi = self._num_sample if autograd.is_training() else self._rpn_test_post_nms
        batch_size = self._batch_size if autograd.is_training() else 1
        with autograd.pause():
            roi_batchid = F.arange(0, batch_size)
            roi_batchid = F.repeat(roi_batchid, num_roi)
            # remove batch dim because ROIPooling require 2d input
            rpn_roi = F.concat(*[roi_batchid.reshape((-1, 1)), rpn_box.reshape((-1, 4))], dim=-1)
            rpn_roi = F.stop_gradient(rpn_roi)

        if self.num_stages > 1:
            # using FPN
            pooled_feat = self._pyramid_roi_feats(F, feat, rpn_roi, self._roi_size,
                                                  self._strides, roi_mode=self._roi_mode)
        else:
            # ROI features
            if self._roi_mode == 'pool':
                pooled_feat = F.ROIPooling(feat[0], rpn_roi, self._roi_size, 1. / self._strides)
            elif self._roi_mode == 'align':
                pooled_feat = F.contrib.ROIAlign(feat[0], rpn_roi, self._roi_size,
                                                 1. / self._strides, sample_ratio=2)
            else:
                raise ValueError("Invalid roi mode: {}".format(self._roi_mode))

        # RCNN prediction
        if self.top_features is not None:
            top_feat = self.top_features(pooled_feat)
        else:
            top_feat = pooled_feat
            # mask_attention = None
            # mask_attention = self.mask_attention(top_feat)
            # top_feat = top_feat * mask_attention

        if self.box_features is None:
            box_feat = F.contrib.AdaptiveAvgPooling2D(top_feat, output_size=1)
        else:
            box_feat = self.box_features(top_feat)
        cls_pred = self.class_predictor(box_feat)
        box_pred = self.box_predictor(box_feat)
        # cls_pred (B * N, C) -> (B, N, C)
        cls_pred = cls_pred.reshape((batch_size, num_roi, self.num_class + 1))
        # box_pred (B * N, C * 4) -> (B, N, C, 4)
        box_pred = box_pred.reshape((batch_size, num_roi, self.num_class, 4))

        # no need to convert bounding boxes in training, just return
        if autograd.is_training():
            if self._additional_output:
                return (cls_pred, box_pred, rpn_box, samples, matches,
                        raw_rpn_score, raw_rpn_box, anchors, top_feat, binary_cls)
            return (cls_pred, box_pred, rpn_box, samples, matches,
                    raw_rpn_score, raw_rpn_box, anchors, binary_cls)

        # cls_ids (B, N, C), scores (B, N, C)
        cls_ids, scores = self.cls_decoder(F.softmax(cls_pred, axis=-1))
        # cls_ids, scores (B, N, C) -> (B, C, N) -> (B, C, N, 1)
        cls_ids = cls_ids.transpose((0, 2, 1)).reshape((0, 0, 0, 1))
        scores = scores.transpose((0, 2, 1)).reshape((0, 0, 0, 1))
        # box_pred (B, N, C, 4) -> (B, C, N, 4)
        box_pred = box_pred.transpose((0, 2, 1, 3))

        # rpn_boxes (B, N, 4) -> B * (1, N, 4)
        rpn_boxes = _split(rpn_box, axis=0, num_outputs=batch_size, squeeze_axis=False)
        # cls_ids, scores (B, C, N, 1) -> B * (C, N, 1)
        cls_ids = _split(cls_ids, axis=0, num_outputs=batch_size, squeeze_axis=True)
        scores = _split(scores, axis=0, num_outputs=batch_size, squeeze_axis=True)
        # box_preds (B, C, N, 4) -> B * (C, N, 4)
        box_preds = _split(box_pred, axis=0, num_outputs=batch_size, squeeze_axis=True)

        # per batch predict, nms, each class has topk outputs
        results = []
        for rpn_box, cls_id, score, box_pred in zip(rpn_boxes, cls_ids, scores, box_preds):
            # box_pred (C, N, 4) rpn_box (1, N, 4) -> bbox (C, N, 4)
            bbox = self.box_decoder(box_pred, self.box_to_center(rpn_box))
            # res (C, N, 6)
            res = F.concat(*[cls_id, score, bbox], dim=-1)
            if self.force_nms:
                # res (1, C*N, 6), to allow cross-catogory suppression
                res = res.reshape((1, -1, 0))
            # res (C, self.nms_topk, 6)
            res = F.contrib.box_nms(
                res, overlap_thresh=self.nms_thresh, topk=self.nms_topk, valid_thresh=0.0001,
                id_index=0, score_index=1, coord_start=2, force_suppress=self.force_nms)
            # res (C * self.nms_topk, 6)
            res = res.reshape((-3, 0))
            results.append(res)

        # result B * (C * topk, 6) -> (B, C * topk, 6)
        result = F.stack(*results, axis=0)
        ids = F.slice_axis(result, axis=-1, begin=0, end=1)
        scores = F.slice_axis(result, axis=-1, begin=1, end=2)
        bboxes = F.slice_axis(result, axis=-1, begin=2, end=6)
        if self._additional_output:
            return ids, scores, bboxes, feat, binary_cls
        return ids, scores, bboxes, binary_cls


def get_faster_rcnn_attention(name, dataset, pretrained=False, ctx=mx.cpu(),
                              root='./models', **kwargs):
    r"""Utility function to return faster rcnn networks.

    Parameters
    ----------
    name : str
        Model name.
    dataset : str
        The name of dataset.
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : mxnet.Context
        Context such as mx.cpu(), mx.gpu(0).
    root : str
        Model weights storing path.

    Returns
    -------
    mxnet.gluon.HybridBlock
        The Faster-RCNN network.

    """
    net = FasterRCNN_Addcls(**kwargs)

    if pretrained:
        from gluoncv.model_zoo.model_store import get_model_file
        full_name = '_'.join(('faster_rcnn', name, dataset))
        net.load_parameters(get_model_file(full_name, tag=pretrained, root=root), ctx=ctx)
    else:
        for v in net.collect_params().values():
            try:
                v.reset_ctx(ctx)
            except ValueError:
                pass
    return net


def faster_rcnn_fpn_resnet50_v1b_coco(classes, pretrained=False, pretrained_base=True, **kwargs):
    r"""Faster RCNN model with FPN from the paper
    "Ren, S., He, K., Girshick, R., & Sun, J. (2015). Faster r-cnn: Towards
    real-time object detection with region proposal networks"
    "Lin, T., Dollar, P., Girshick, R., He, K., Hariharan, B., Belongie, S. (2016).
    Feature Pyramid Networks for Object Detection"

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    pretrained_base : bool or str, optional, default is True
        Load pretrained base network, the extra layers are randomized. Note that
        if pretrained is `Ture`, this has no effect.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Examples
    --------
    >>> model = get_faster_rcnn_fpn_resnet101_v1d_coco(pretrained=True)
    >>> print(model)
    """
    from gluoncv.model_zoo.resnetv1b import resnet50_v1b

    # classes = TCTDetection.CLASSES
    pretrained_base = False if pretrained else pretrained_base
    base_network = resnet50_v1b(pretrained=pretrained_base, dilated=False,
                                use_global_stats=True, **kwargs)
    features = FPNFeatureExpander(
        network=base_network,
        outputs=['layers1_relu8_fwd', 'layers2_relu11_fwd', 'layers3_relu17_fwd',
                 'layers4_relu8_fwd'], num_filters=[256, 256, 256, 256], use_1x1=True,
        use_upsample=True, use_elewadd=True, use_p6=True, no_bias=False, pretrained=pretrained_base)

    # op on the feature after roipooled/roialign
    top_features = None
    # 2 FC layer before RCNN cls and reg
    box_features = nn.HybridSequential()
    # according to the function get_faster_rcnn parameter roi_size
    # box_features.add(nn.AvgPool2D(pool_size=(3, 3), strides=2, padding=1))  # 14x14 reduce to 7x7
    for _ in range(2):
        box_features.add(nn.Dense(1024, weight_initializer=mx.init.Normal(0.01)))
        box_features.add(nn.Activation('relu'))

    """
    train_patterns = '|'.join(
        ['.*dense', '.*rpn', '.*down(2|3|4)_conv', '.*layers(2|3|4)_conv', 'P'])
    """

    # binary cls layers
    train_patterns = '.*dense2_'

    # target_roi_scale
    return get_faster_rcnn_attention(
        name='fpn_resnet50_v1b', dataset='coco', pretrained=pretrained, features=features,
        top_features=top_features, classes=classes, box_features=box_features,
        short=800, max_size=1333, min_stage=2, max_stage=6, train_patterns=train_patterns,
        nms_thresh=0.5, nms_topk=-1, post_nms=-1, roi_mode='align', roi_size=(7, 7),
        strides=(4, 8, 16, 32, 64), clip=4.14, rpn_channel=1024, base_size=16,
        scales=(1, 2, 4, 8, 16), ratios=(0.8, 1, 1.25), alloc_size=(384, 384),
        rpn_nms_thresh=0.7, rpn_train_pre_nms=12000, rpn_train_post_nms=2000,
        rpn_test_pre_nms=6000, rpn_test_post_nms=1000, rpn_min_size=1, num_sample=512,
        pos_iou_thresh=0.5, pos_ratio=0.25, max_num_gt=120, **kwargs)


def faster_rcnn_fpn_resnet101_v1d_coco(classes, pretrained=False, pretrained_base=True, **kwargs):
    r"""Faster RCNN model with FPN from the paper
    "Ren, S., He, K., Girshick, R., & Sun, J. (2015). Faster r-cnn: Towards
    real-time object detection with region proposal networks"
    "Lin, T., Dollar, P., Girshick, R., He, K., Hariharan, B., Belongie, S. (2016).
    Feature Pyramid Networks for Object Detection"

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    pretrained_base : bool or str, optional, default is True
        Load pretrained base network, the extra layers are randomized. Note that
        if pretrained is `Ture`, this has no effect.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Examples
    --------
    >>> model = get_faster_rcnn_fpn_resnet101_v1d_coco(pretrained=True)
    >>> print(model)
    """
    from gluoncv.model_zoo.resnetv1b import resnet101_v1d

    # classes = TCTDetection.CLASSES
    pretrained_base = False if pretrained else pretrained_base
    base_network = resnet101_v1d(pretrained=pretrained_base, dilated=False,
                                 use_global_stats=True, **kwargs)
    features = FPNFeatureExpander(
        network=base_network,
        outputs=['layers1_relu8_fwd', 'layers2_relu11_fwd', 'layers3_relu68_fwd',
                 'layers4_relu8_fwd'], num_filters=[256, 256, 256, 256], use_1x1=True,
        use_upsample=True, use_elewadd=True, use_p6=True, no_bias=False, pretrained=pretrained_base)

    # op on the feature after roipooled/roialign
    top_features = None
    # 2 FC layer before RCNN cls and reg
    box_features = nn.HybridSequential()
    # according to the function get_faster_rcnn parameter roi_size
    # box_features.add(nn.AvgPool2D(pool_size=(3, 3), strides=2, padding=1))  # 14x14 reduce to 7x7
    for _ in range(2):
        box_features.add(nn.Dense(1024, weight_initializer=mx.init.Normal(0.01)))
        box_features.add(nn.Activation('relu'))

    train_patterns = '|'.join(
        ['.*dense', '.*rpn', '.*down(2|3|4)_conv', '.*layers(2|3|4)_conv', 'P'])
    """
    # binary cls layers
    train_patterns = '.*dense(2|3)_'
    """
    # target_roi_scale

    return get_faster_rcnn_attention(
        name='fpn_resnet101_v1d', dataset='coco', pretrained=pretrained, features=features,
        top_features=top_features, classes=classes, box_features=box_features,
        short=800, max_size=1333, min_stage=2, max_stage=6, train_patterns=train_patterns,
        nms_thresh=0.5, nms_topk=-1, post_nms=-1, roi_mode='align', roi_size=(7, 7),
        strides=(4, 8, 16, 32, 64), clip=4.14, rpn_channel=1024, base_size=16,
        scales=(1, 2, 4, 8, 16), ratios=(0.8, 1, 1.25), alloc_size=(384, 384),
        rpn_nms_thresh=0.7, rpn_train_pre_nms=12000, rpn_train_post_nms=2000,
        rpn_test_pre_nms=6000, rpn_test_post_nms=1000, rpn_min_size=1, num_sample=512,
        pos_iou_thresh=0.5, pos_ratio=0.25, max_num_gt=120, **kwargs)


