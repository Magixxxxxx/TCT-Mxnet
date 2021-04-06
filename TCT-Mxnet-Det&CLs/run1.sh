python3 fasterrcnn_cls_norm.py --num-workers 4 --gpus 0,1,2,3 --batch-size 4 \
--epochs 20 --lr 0.002 --lr-decay-epoch 10,15 \
--dataset coco --network resnet50_v1b --val-interval 1 --use-fpn --details TCT