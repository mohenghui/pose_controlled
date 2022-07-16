import warnings
#消除警告
import os
import cv2
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings(action='ignore')
from keras.callbacks import CSVLogger,ModelCheckpoint,EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from load_and_process import data_reader
from load_and_process import dense_to_one_hot
from models.cnn import mini_XCEPTION
from load_and_process import proprecess_input
from sklearn.model_selection import  train_test_split

# 参数
batch_size=32
# 每次训练投喂量
num_epochs=50
# 训练次数
input_shape=(300,300,1)
# 输入的尺寸和通道数
validation_split=.2
# 划分验证集合训练集
verbose =1
# 输出精度条记录(默认)
num_classes=10
# 三个分类
patience=50
# 模型存放位置
base_path='models/'
# 训练集存放的位置
data_path='dataset_train/'
#
num_class_number=10
# img=cv2.imread("dataset_train\\0\\000.png")
# print(img.shape)

# 构建模型
model=mini_XCEPTION(input_shape,num_classes)
# 输入图片大小和类型
# 优化器选adam
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              # 多分类的对数损失函数
              metrics=['accuracy']
              )
model.summary()


# 定义回调函数 Callbacks 用于训练过程
log_file_path =base_path+'shoushi_training.log'
csv_logger = CSVLogger(log_file_path,append=False)
early_stop =EarlyStopping('val_loss',patience=patience)
reduce_lr = ReduceLROnPlateau('val_loss',factor=0.1,
                              patience=int(patience/4),
                              verbose=1)

# 模型位置及命名
trained_models_path=base_path+'_mini_XCEPTION'
model_names=trained_models_path+'.{epoch:03d}-{val_acc:.2f}.hdf5'

# 定义模型权重位置、命名等
model_checkpoint=ModelCheckpoint(model_names,
                                 'val_loss',verbose=1,
                                 save_best_only=True)

callbacks=[model_checkpoint,csv_logger,early_stop,reduce_lr]

# 载入数据集
train_data,train_label=data_reader(data_path)
# 将train_label转换成one_hot

train_label=dense_to_one_hot(train_label,num_class_number)
train_data=proprecess_input(train_data)
num_samples,num_classes=train_label.shape

# 划分训练、测试集
xtrain,xtest,ytrain,ytest=train_test_split(train_data,train_label,test_size=0.2,shuffle=True)

# 图片生成器，在批量中对数据进行增强，扩充数据集大小
data_generator=ImageDataGenerator(
    featurewise_center=False,
    featurewise_std_normalization=False,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=.1,
    horizontal_flip=True)

# 利用数据增强进行训练
model.fit_generator(data_generator.flow(xtrain,ytrain,batch_size),
                    steps_per_epoch=len(xtrain)/batch_size,
                    epochs=num_epochs,
                    verbose=1,callbacks=callbacks,
                    validation_data=(xtest,ytest))

