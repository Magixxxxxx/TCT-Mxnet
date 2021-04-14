python3 fasterrcnn_cls_bi_norm.py --gpus 0,1,2,3 --num-workers 4 \
--batch-size 4 --dataset coco --network resnet50_v1b --epochs 6 --lr-decay-epoch 3 --lr 0.00005 \
--val-interval 1 --use-fpn --details TCT --lr-warmup -1 \
--resume faster_rcnn_fpn_resnet50_v1b_coco_0014_28.8000.params --save-prefix bicls-