from mxnet.metric import EvalMetric
from mxnet import nd


class BinaryAccMetric(EvalMetric):
    def __init__(self):
        super(BinaryAccMetric, self).__init__('BinaryAcc')

    def update(self, labels, preds):
        # label: [binary_label]
        # preds: [binary_cls_logits]
        binary_label = labels
        binary_cls_logits = preds[0]
        pred_label = binary_cls_logits >= 0.5
        num_acc = nd.sum(pred_label == binary_label)

        self.sum_metric += num_acc.asscalar()
        self.num_inst += 1


class PrecisionMetric(EvalMetric):
    def __init__(self):
        super(BinaryAccMetric, self).__init__('Precision')

    def update(self, labels, preds):

        binary_label = labels
        binary_cls_logits = preds[0]
        pred_label = binary_cls_logits >= 0.5

        TP = nd.sum([p for p,g in (pred_label, binary_label) if (p==g and g==1)])
        FP = nd.sum([p for p,g in (pred_label, binary_label) if (p!=g and g==1)])
        FN = nd.sum([p for p,g in (pred_label, binary_label) if (p!=g and g==0)])

        precision = TP / (TP + FN)
        self.sum_metric += precision.asscalar()
        self.num_inst += 1

class RecallMetric(EvalMetric):
    def __init__(self):
        super(BinaryAccMetric, self).__init__('Recall')

    def update(self, labels, preds):

        binary_label = labels
        binary_cls_logits = preds[0]
        pred_label = binary_cls_logits >= 0.5

        TP = nd.sum([p for p,g in (pred_label, binary_label) if (p==g and g==1)])
        FP = nd.sum([p for p,g in (pred_label, binary_label) if (p!=g and g==1)])
        FN = nd.sum([p for p,g in (pred_label, binary_label) if (p!=g and g==0)])

        recall = TP / (TP + FP)
        self.sum_metric += recall.asscalar()
        self.num_inst += 1