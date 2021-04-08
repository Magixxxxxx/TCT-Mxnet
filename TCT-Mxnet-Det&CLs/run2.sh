python3 fasterrcnn_cls_bi_norm.py --gpus 0 --num-workers 4 \
--batch-size 1 --dataset coco --network resnet50_v1b --epochs 6 --lr-decay-epoch 3 --lr 0.000001 \
--val-interval 1 --use-fpn --details TCT \
--resume faster_rcnn_fpn_resnet50_v1b_coco_0003_36.7000.params --save-prefix bicls-
