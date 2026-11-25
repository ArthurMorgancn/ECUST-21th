Config.cfg 参数详解
[train] 训练配置
参数	说明	建议值
voc_folder	数据集根目录路径	当前：D:\21SmartCar\yolo_zf\dataset\0326
train_data	训练数据列表文件	train_data.txt
eva_data	评估/验证数据列表文件	eva_data.txt
cluster_anchor	锚点文件路径	yolo3_anchors.txt
batch_size	批次大小	当前：16，建议：8-32
total_epochs	总训练轮数	当前：150，建议：200-500
[model] 模型配置
参数	说明	当前值	建议
width	输入图像宽度	112	建议：224或更大
height	输入图像高度	112	建议：224或更大
num_heads	检测头数量	1	建议：2（多尺度检测）
cluster_number	聚类锚点数量	3	保持不变
divider	特征图下采样倍数	8	单头时保持8，双头时改为[16,32]
iou_type	IoU损失类型	iou	建议：siou（更精确）
body_type	骨干网络类型	mbv2	保持不变
obj_scale	目标损失权重	8	建议：5-10
noobj_scale	非目标损失权重	1	建议：1-2
iou_threshold	训练时IoU阈值	0.5	建议：0.7
num_classes	类别数量	1	根据实际类别设置
class_names	类别名称	object	根据实际类别设置
[inference] 推理配置（解决重叠的关键！）
参数	说明	当前值	解决重叠的建议值
nms_iou_threshold	NMS IoU阈值（最重要！）	0.5	建议：0.3-0.4
nms_score_threshold	置信度阈值	0.3	建议：0.4-0.5
max_detections	最大检测数量	10	根据实际场景调整
解决物体框重叠的核心方案
1. 降低 NMS IoU 阈值（最直接有效）

/d:/deep_learning/yolo/yolo3_smartcar/config.cfg
[inference]
nms_iou_threshold = 0.3  # 从0.5降到0.3，更严格地去除重叠框
nms_score_threshold = 0.4  # 提高置信度阈值，过滤低质量检测
max_detections = 10
原理： NMS（非极大值抑制）会去除重叠度超过阈值的检测框。阈值越低，去除重叠越严格。

2. 提高训练时的 IoU 阈值

/d:/deep_learning/yolo/yolo3_smartcar/config.cfg
[model]
iou_threshold = 0.7  # 从0.5提高到0.7
原理： 训练时使用更高的IoU阈值可以让模型学习更精确的边界框定位。

3. 使用更好的 IoU 损失函数

/d:/deep_learning/yolo/yolo3_smartcar/config.cfg
[model]
iou_type = siou  # 从iou改为siou
原理： SIoU（ScIoU）考虑了边界框的方向和距离，比普通IoU对边界框回归更有效。

4. 增加输入图像尺寸（提升检测精度）

/d:/deep_learning/yolo/yolo3_smartcar/config.cfg
[model]
width = 224
height = 224
原理： 更大的输入尺寸可以捕获更多细节，提高检测精度，减少误检。

5. 使用双检测头（多尺度检测）

/d:/deep_learning/yolo/yolo3_smartcar/config.cfg
[model]
num_heads = 2
divider = [16,32]
原理： 双检测头可以在不同尺度上检测目标，提高对不同大小目标的检测能力。

参数影响总结
直接影响物体框重叠的参数（按重要性排序）：
nms_iou_threshold ⭐⭐⭐⭐⭐

最直接控制重叠的参数
值越小，去除重叠越严格
建议范围：0.2-0.4
nms_score_threshold ⭐⭐⭐⭐

过滤低置信度检测
值越大，保留的检测越少
建议范围：0.4-0.6
iou_threshold ⭐⭐⭐⭐

训练时的IoU阈值
影响模型学习精度
建议范围：0.6-0.8
iou_type ⭐⭐⭐


输入尺寸
越大精度越高，但计算量也越大