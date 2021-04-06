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
