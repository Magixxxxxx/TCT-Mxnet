<<<<<<< HEAD
python3 fasterrcnn_cls_bi_norm.py --gpus 0 --num-workers 4 \
--batch-size 1 --dataset coco --network resnet50_v1b --epochs 6 --lr-decay-epoch 3 --lr 0.000001 \
--val-interval 1 --use-fpn --details TCT \
--resume faster_rcnn_fpn_resnet50_v1b_coco_0003_36.7000.params --save-prefix bicls-
=======
python3 fasterrcnn_cls_bi_norm.py --gpus 4,5,6,7 --num-workers 4 --batch-size 4 --dataset coco --network resnet50_v1b --epochs 4 --lr-decay-epoch 3 --lr 0.0001 --val-interval 1 --use-fpn --details TCT --lr-warmup -1 --resume 0722-0009.params --save-prefix bi-
>>>>>>> 76c6bc30b1718a866a79b0d17e85473ae1ab8b76
