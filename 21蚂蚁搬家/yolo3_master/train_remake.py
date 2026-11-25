import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras import backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, LearningRateScheduler
import cv2

from utils import get_random_data, yolo_cfg
from utils import get_lr_scheduler

import model_resnet
from tflite_add_post_processing import add_post_node

def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]                                              
    return class_names

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)

def create_res_tiny_model(input_shape, anchors, num_classes, load_pretrained=False, freeze_body=2,
            weights_path='model_data/tiny_yolo_weights.h5',
            iou_threshhold=0.5, obj_scale=1, noobj_scale=1, num_heads=2, divider=32, iou_type='siou'):
    '''create the training model, for Tiny YOLOv3'''
    K.clear_session()
    h, w = input_shape
    image_input = Input(shape=(h, w, 3))
    
    num_anchors = len(anchors)
    if num_heads == 2:
        y_true = [Input(shape=(h//{0:divider[0], 1:divider[1]}[l], w//{0:divider[0], 1:divider[1]}[l], \
            num_anchors//2, num_classes+5)) for l in range(2)]
        model_body = model_resnet.tiny_yolo_res_body((h, w, 3), num_anchors//num_heads, num_classes, num_heads, divider)
        print('Create Tiny YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

        if iou_type == 'iou':
            model_loss = Lambda(model_resnet.yolo_loss, output_shape=(1,), name='yolo_loss',
            arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': iou_threshhold, 'obj_scale': obj_scale, 'noobj_scale': noobj_scale})(
            [*model_body.output, *y_true])
        else:
            model_loss = Lambda(model_resnet.yolo_loss_iou, output_shape=(1,), name='yolo_loss',
                arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': iou_threshhold, 'obj_scale': obj_scale, 'noobj_scale': noobj_scale, 'iou_type': iou_type})(
                [*model_body.output, *y_true])
        model = Model([model_body.input, *y_true], model_loss)
    elif num_heads == 1:
        
        y_true = [Input(shape=(h//divider[0], w//divider[0], num_anchors, num_classes+5))]
        model_body = model_resnet.tiny_yolo_res_body((h, w, 3), num_anchors//num_heads, num_classes, num_heads, divider)
        print('Create Tiny YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

        if iou_type == 'iou':
            model_loss = Lambda(model_resnet.yolo_loss_one, output_shape=(1,), name='yolo_loss',
                arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': iou_threshhold, 'obj_scale': obj_scale, 'noobj_scale': noobj_scale, 'divider': divider[0]})(
                [model_body.output, *y_true])
        else:
            model_loss = Lambda(model_resnet.yolo_loss_iou_one, output_shape=(1,), name='yolo_loss',
                arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': iou_threshhold, 'obj_scale': obj_scale, 'noobj_scale': noobj_scale, 'divider': divider[0], 'iou_type': iou_type})(
                [model_body.output, *y_true])
        model = Model([model_body.input, *y_true], model_loss)

    return model, model_body

def preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes, divider):
    '''Preprocess true boxes to training input format'''
    # 处理空的情况
    if len(true_boxes) == 0:
        return None
    
    assert (true_boxes[..., 4]<num_classes).all(), 'class id must be less than num_classes'
    num_layers = len(anchors)//3
    if (num_layers > 1):
        anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5],[0,1,2]]
    elif num_layers == 1: 
        anchor_mask = [[0,1,2]]

    true_boxes = np.array(true_boxes, dtype='float32')
    input_shape = np.array(input_shape, dtype='int32')
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
    true_boxes[..., 0:2] = boxes_xy/input_shape[::-1]
    true_boxes[..., 2:4] = boxes_wh/input_shape[::-1]

    m = true_boxes.shape[0]
    if (num_layers == 3):
        grid_shapes = [input_shape//{0:32, 1:16, 2:8}[l] for l in range(num_layers)]
        y_true = [np.zeros((m, grid_shapes[l][0], grid_shapes[l][1], len(anchor_mask[l]), 5+num_classes),
            dtype='float32') for l in range(num_layers)]
    elif num_layers == 2:
        grid_shapes = [input_shape//{0:divider[0], 1:divider[1]}[l] for l in range(num_layers)]
        y_true = [np.zeros((m, grid_shapes[l][0], grid_shapes[l][1], len(anchor_mask[l]), 5+num_classes),
            dtype='float32') for l in range(num_layers)]
    elif num_layers == 1:
        grid_shapes = [input_shape//divider[0]]
        y_true = [np.zeros((m, grid_shapes[0][0], grid_shapes[0][1], len(anchor_mask[0]), 5+num_classes),
        dtype='float32')]

    anchors = np.expand_dims(anchors, 0)
    anchor_maxes = anchors / 2.
    anchor_mins = -anchor_maxes
    valid_mask = boxes_wh[..., 0]>0

    for b in range(m):
        wh = boxes_wh[b, valid_mask[b]]
        if len(wh)==0: continue
        wh = np.expand_dims(wh, -2)
        box_maxes = wh / 2.
        box_mins = -box_maxes

        intersect_mins = np.maximum(box_mins, anchor_mins)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)

        best_anchor = np.argmax(iou, axis=-1)

        for t, n in enumerate(best_anchor):
            for l in range(num_layers):
                if n in anchor_mask[l]:
                    i = np.floor(true_boxes[b,t,0]*grid_shapes[l][1]).astype('int32')
                    j = np.floor(true_boxes[b,t,1]*grid_shapes[l][0]).astype('int32')
                    k = anchor_mask[l].index(n)
                    c = true_boxes[b,t, 4].astype('int32')
                    y_true[l][b, j, i, k, 0:4] = true_boxes[b,t, 0:4]
                    y_true[l][b, j, i, k, 4] = 1
                    y_true[l][b, j, i, k, 5+c] = 1

    return y_true

def apply_basic_augmentation(image, boxes, input_shape):
    """使用OpenCV实现基本的数据增强"""
    # 确保图像是uint8类型
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    
    h, w = input_shape
    img_h, img_w = image.shape[:2]
    
    # 复制图像和框
    aug_image = image.copy()
    aug_boxes = boxes.copy() if len(boxes) > 0 else boxes
    
    # 1. 随机水平翻转
    '''
    if np.random.random() < 0.3 and len(aug_boxes) > 0:
        aug_image = cv2.flip(aug_image, 1)
        aug_boxes[:, [0, 2]] = img_w - aug_boxes[:, [2, 0]]
    # 3. 随机高斯噪声
    if np.random.random() < 0.2:
        noise = np.random.normal(0, 15, aug_image.shape).astype(np.uint8)
        aug_image = cv2.add(aug_image, noise)
        
    # 4. 随机模糊
    if np.random.random() < 0.15:
        kernel_size = np.random.choice([3, 5])
        aug_image = cv2.GaussianBlur(aug_image, (kernel_size, kernel_size), 0)
    
    # 5. 随机HSV变换
    if np.random.random() < 0.25:
        try:
            hsv = cv2.cvtColor(aug_image, cv2.COLOR_BGR2HSV)
            hsv = hsv.astype(np.float32)
            hsv[:, :, 0] = np.clip(hsv[:, :, 0] + np.random.randint(-10, 10), 0, 179)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * np.random.uniform(0.8, 1.2), 0, 255)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * np.random.uniform(0.8, 1.2), 0, 255)
            aug_image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        except:
            pass
    
    # 6. 随机锐化
    if np.random.random() < 0.15:
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        aug_image = cv2.filter2D(aug_image, -1, kernel)
        
    '''
    
    
    # 2. 随机亮度对比度调整
    #if np.random.random() < 0.3:
    alpha = np.random.uniform(0.9, 2.0)
    beta = np.random.uniform(-30, 30)
    aug_image = cv2.convertScaleAbs(aug_image, alpha=alpha, beta=beta)
    
    
    
    
    return aug_image, aug_boxes

def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes, divider, use_augmentation=True):
    '''数据生成器，支持数据增强'''
    n = len(annotation_lines)
    i = 0
    
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i == 0:
                np.random.shuffle(annotation_lines)
            
            # 读取原始数据
            image, box = get_random_data(annotation_lines[i], input_shape, random=True)
            
            # 应用数据增强
            if use_augmentation and len(box) > 0:
                # 应用增强
                image, box = apply_basic_augmentation(image, box, input_shape)
                
                # 确保边界框有效
                if len(box) > 0:
                    # 裁剪边界框到图像范围内
                    box[:, [0, 2]] = np.clip(box[:, [0, 2]], 0, input_shape[1] - 1)
                    box[:, [1, 3]] = np.clip(box[:, [1, 3]], 0, input_shape[0] - 1)
                    # 过滤无效框
                    valid = (box[:, 2] > box[:, 0]) & (box[:, 3] > box[:, 1])
                    box = box[valid]
            
            image_data.append(image)
            box_data.append(box)
            i = (i + 1) % n
        
        # 将image_data转换为numpy数组
        image_data = np.array(image_data)
        
        # 处理box_data - 将其填充为相同形状
        max_boxes = 0
        for boxes in box_data:
            if len(boxes) > max_boxes:
                max_boxes = len(boxes)
        
        # 如果最大边界框数为0，创建一个空的batch
        if max_boxes == 0:
            # 使用一个小的默认值，preprocess_true_boxes会处理
            box_data_padded = np.zeros((batch_size, 1, 5), dtype='float32')
        else:
            box_data_padded = np.zeros((batch_size, max_boxes, 5), dtype='float32')
            for b, boxes in enumerate(box_data):
                if len(boxes) > 0:
                    box_data_padded[b, :len(boxes)] = boxes
        
        y_true = preprocess_true_boxes(box_data_padded, input_shape, anchors, num_classes, divider)
        
        # 如果y_true为None，创建一个默认的y_true
        if y_true is None:
            # 创建一个空的y_true
            num_layers = len(anchors)//3
            if num_layers == 2:
                y_true = [np.zeros((batch_size, input_shape[0]//divider[0], input_shape[1]//divider[0], 
                                   len(anchors)//2, num_classes+5), dtype='float32'),
                         np.zeros((batch_size, input_shape[0]//divider[1], input_shape[1]//divider[1], 
                                   len(anchors)//2, num_classes+5), dtype='float32')]
            else:
                y_true = [np.zeros((batch_size, input_shape[0]//divider[0], input_shape[1]//divider[0], 
                                   len(anchors), num_classes+5), dtype='float32')]
        
        yield [image_data, *y_true], np.zeros(batch_size)

def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes, divider, use_augmentation=True):
    n = len(annotation_lines)
    if n == 0 or batch_size <= 0: 
        return None
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes, divider, use_augmentation)

import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser() 
    parser.add_argument('-body', help='body type', default=r'mbv2', type=str)
    parser.add_argument('--no_aug', help='disable data augmentation', action='store_true')
    args, unknown = parser.parse_known_args()

    cfg = yolo_cfg()
    annotation_path = cfg.train_data
    log_dir = 'logs/000/'
   
    anchors_path = cfg.cluster_anchor
    class_names = cfg.class_names
    num_classes = cfg.num_classes
    anchors = get_anchors(anchors_path)
    divider = cfg.divider
    input_shape = (cfg.width, cfg.height)

    if args.body == 'mbv2':
        print("head num:%d"%cfg.num_heads)
        model, infer_model = create_res_tiny_model(input_shape, anchors, num_classes, cfg.iou_threshold,
        cfg.obj_scale, cfg.noobj_scale, num_heads=cfg.num_heads, divider=divider, iou_type=cfg.iou_type)
        model_name = 'mbv2_%dhead_ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'%cfg.num_heads
    else:
        print('unsupported body type')
        exit(-1)

    model.compile(optimizer=Adam(lr=1e-3), loss={
            'yolo_loss': lambda y_true, y_pred: y_pred})

    model.summary()
    
    print('Build Model')

    # 修复ModelCheckpoint的period参数警告
    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(
        log_dir + model_name,
        monitor='val_loss', 
        save_weights_only=True, 
        save_best_only=True, 
        save_freq='epoch'  # 使用save_freq替代period
    )
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=1)

    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val
    
    lr_decay_type = 'cos'
    lr_start = 1e-2
    lr_min = 1e-6
    batch_size = cfg.batch_size 
    nbs = 32
    lr_limit_max = 1e-3 
    lr_limit_min = 3e-6 
    epochs = cfg.total_epochs
    Init_lr_fit = min(max(batch_size / nbs * lr_start, lr_limit_min), lr_limit_max)
    Min_lr_fit = min(max(batch_size / nbs * lr_min, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

    print('lr_start:%f, Min_lr_fit:%f'%(lr_start, Min_lr_fit))
    lr_scheduler_func = get_lr_scheduler(lr_decay_type, lr_start, Min_lr_fit, epochs)
    lr_scheduler = LearningRateScheduler(lr_scheduler_func, verbose=1)

    use_augmentation = not args.no_aug
    print(f'Data Augmentation: {"Enabled" if use_augmentation else "Disabled"}')
    
    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
    
    # 使用 model.fit 替代 model.fit_generator
    model.fit(
        data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes, divider, use_augmentation=use_augmentation),
        steps_per_epoch=max(1, num_train//batch_size),
        validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes, divider, use_augmentation=False),
        validation_steps=max(1, num_val//batch_size),
        epochs=cfg.total_epochs,
        initial_epoch=0,
        callbacks=[logging, checkpoint, lr_scheduler])
    
    model.save_weights(log_dir + '%s_trained_weights_final.h5'%cfg.body_type)

    infer_model.load_weights(log_dir + '%s_trained_weights_final.h5'%cfg.body_type)
    converter = tf.lite.TFLiteConverter.from_keras_model(infer_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supportes_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    def representative_data_gen(): 
        n = len(lines)
        i = 0
        input_shape = (cfg.width, cfg.height)
        
        image_data = []
        box_data = []
        for b in range(100):
            if i == 0:
                np.random.shuffle(lines)
            image, box = get_random_data(lines[i], input_shape, random=True)
            image = (image).astype('float32')
            image_data.append(image)
            box_data.append(box)
            i = (i + 1) % n

        image_data = np.array(image_data)
        image_data = image_data.reshape(100, cfg.width, cfg.height, 3)
        for input_value in image_data:
            input_value = input_value.reshape(1, cfg.width, cfg.height, 3)
            yield [input_value]

    converter.representative_dataset = representative_data_gen
    tflite_model_quant = converter.convert() 
    m_path = 'yolo3_%s.tflite'%(cfg.iou_type)
    with open(m_path,'wb') as f:
        f.write(tflite_model_quant)
        f.close()
    add_post_node(m_path)
    print('Training Complete')