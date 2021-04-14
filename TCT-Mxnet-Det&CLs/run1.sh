python3 fasterrcnn_cls_norm.py --num-workers 4 --gpus 0,1,2,3 --batch-size 4 \
--epochs 20 --lr 0.0002 --lr-decay-epoch 13,17 \
--dataset coco --network resnet50_v1b --val-interval 1 --use-fpn --details TCT \
--resume TCT0331_29.4000.params --lr-warmup -1
