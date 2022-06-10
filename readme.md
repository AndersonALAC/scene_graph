# 基于混合注意力网络与动态标签预测的场景图生成
本repo基于https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch

## 安装

参见./INSTALL.md

## 数据集

参见./DATASET.md

## 训练

如果你使用google colab，可以直接使用我写好的notebook进行训练，环境配置等都包含在内，只需上传到colab之后run all即可。参见./SGG_dev.ipynb

### motif net和vctree

motif net和vctree在master分支下，sha_dlfe则在dev分支下，使用git命令checkout分支

```
! cd scene_graph; git checkout master
```

训练示例如下：
VCTREE SGDet 
```
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port 10025 --nproc_per_node=1 tools/relation_train_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR VCTreePredictor SOLVER.IMS_PER_BATCH 8 TEST.IMS_PER_BATCH 1 DTYPE "float16" SOLVER.MAX_ITER 24000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 GLOVE_DIR /content/glove MODEL.PRETRAINED_DETECTOR_CKPT /content/checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR /content/checkpoints/VCTREE_SGDet_2
```

--nproc_per_node参数为当前机器最多的GPU数量，单卡训练使用1即可

MODEL.ROI_RELATION_HEAD.USE_GT_BOX 以及 MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL控制训练时是否使用GT和label，两者均为True时为PredCls子任务，为(True, False)时为SGCls子任务，为(False, False)时为SGDet子任务。

SOLVER.IMS_PER_BATCH 训练batch size
TEST.IMS_PER_BATCH 测试batch size
注意：训练SGCls和SGDet时TEST.IMS_PER_BATCH必须等于--nproc_per_node的值，即每个GPU核心一个batch中只能测试一张图片

DTYPE 数据精度，由于采用了nvidia apex混合精度加速训练，一般为float16

ROI_RELATION_HEAD.PREDICTOR Motif为MotifPredictor，VCTREE为VCTreePredictor，当前使用的predictor，predictor可以在./maskrcnn-benchmark/modeling/roi_heads/relation_head/roi_relation_predictors.py中找到

SOLVER.MAX_ITER 最大训练步数
SOLVER.CHECKPOINT_PERIOD 保存checkpoint的周期
SOLVER.VAL_PERIOD validation的周期，通常和checkpoint周期一致

GLOVE_DIR glove语义模型路径，找不到时会自己下载
MODEL.PRETRAINED_DETECTOR_CKPT 预训练的faster r-cnn模型位置
OUTPUT_DIR 输出路径

在dev分支下的训练命令如下：
```
! cd scene_graph; git checkout dev
```

```
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port $RANDOM --nproc_per_node=1 tools/relation_train_net.py --config-file "configs/SHA_GCL_e2e_relation_X_101_32_8_FPN_1x.yaml" GLOBAL_SETTING.DATASET_CHOICE 'VG' GLOBAL_SETTING.RELATION_PREDICTOR 'TransLikePredictor' GLOBAL_SETTING.BASIC_ENCODER 'Hybrid-Attention' GLOBAL_SETTING.GCL_SETTING.GROUP_SPLIT_MODE 'divide4' GLOBAL_SETTING.GCL_SETTING.KNOWLEDGE_TRANSFER_MODE 'KL_logit_TopDown' MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False SOLVER.IMS_PER_BATCH 8 TEST.IMS_PER_BATCH 1 DTYPE "float16" SOLVER.MAX_ITER 24000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 GLOVE_DIR /content/glove OUTPUT_DIR /content/checkpoints/SHA_DLFE_SGDet_2 MODEL.BALANCED_NORM True
```

参数基本同上，MODEL.BALANCED_NORM 控制DLFE的开关