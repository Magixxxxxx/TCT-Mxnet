# TCT-Mxnet-Det-CLs

## 环境：
mxnet-cu100==1.5.1.post0
gluoncv            0.5.0
pycocotools        2.0.0

export LANG=“en_US.UTF-8”（解决大部分乱码问题）

cd fasterrcnn
### 1.数据预处理
    cd datasets/TCT/CocoTool/ 
    在deal.sh更改数据集路径，生成coco格式数据
    移动到 datasets/TCT/annotation下：mv json/* ../annoation/

### 2.训练fasterrcnn
    训练设置在run1.sh，大约十几个epoch，0.001学习率（batchsize=gpu数，不明崩溃试着调低num_workers，测试rtx无限制下可以到6，否则要使用2-甚至0）
    训练fasterrcnn，得到best1.params：sh run1.sh

### 3.训练二分类层
    run2.sh，大约3-5个epoch，0.0001学习率，读取best1.params
    得到best2.params（需要用export出的参数，在同名文件夹下）：sh run1.sh
    cp best2.params ../bi-inference/MODEL/
    cp best2.params ../bicls_aug/OUTPUT/

cd ../bi-inference
### 4.二分类推断
    find -name '*.jpg'>path.txt
    python inference.py path.txt best2.params 0,1,...
    得到二分类结果result-path.txt
    python TOOLS/getAugData.py result-path.txt posSamples.txt 0.45 （posSamples可用TOOLS/getposSamples.py生成)
    得到all.txt，train.txt，test.txt：mv *.txt ../bicls_aug/datasets

cd ../bicls_aug    
### 5.finetune二分类层
    设置在utils/config.py更改
    python train.py，得到best3.params

### 6.导出模型
    python model_produce.py best-bi-aug.params，输出-0999.params即最终结果
