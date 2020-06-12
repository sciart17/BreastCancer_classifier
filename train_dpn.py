import  matplotlib
matplotlib.use('Agg')
from    matplotlib import pyplot as plt
matplotlib.rcParams['font.size'] = 18
matplotlib.rcParams['figure.titlesize'] = 18
matplotlib.rcParams['figure.figsize'] = [9, 7]
matplotlib.rcParams['axes.unicode_minus'] = False

import  os
import  tensorflow as tf
import  numpy as np
import time
from    tensorflow import keras
from    tensorflow.keras import layers, optimizers, losses, datasets
from    tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import load_model

tf.random.set_seed(1234)
np.random.seed(1234)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

from  dual_path_network import DPN92, DPN98, DPN107, DPN137
from gen_csv import  load_train, load_test, load_val


def preprocess(x,y):
    x = tf.io.read_file(x)
    x = tf.image.decode_png(x, channels=1)  # RGBA
    x = tf.image.resize(x, [224, 224])  # resize image into 224x224
    x = tf.image.grayscale_to_rgb(x)  # convert gray to rgb
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)
    x = tf.image.random_crop(x, [224, 224, 3])

    # x: [0,255]=> 0~1
    x = tf.cast(x, dtype=tf.float32) / 255.

    y = tf.cast(y, dtype=tf.int32)  # format convert
    y = tf.convert_to_tensor(y)
    y = tf.one_hot(y, depth=3)  # correspond to classes 3

    return x, y


batchsz = 32

images1, labels1, table1 = load_train('train', mode='train')
print('trainimages:', len(images1))
print('traintable:', table1)
db_train = tf.data.Dataset.from_tensor_slices((images1, labels1))
db_train = db_train.shuffle(2000).map(preprocess).batch(batchsz)

images2, labels2, table2 = load_val('val', mode='val')
print('valimages:', len(images2))
print('valtable:', table2)
db_val = tf.data.Dataset.from_tensor_slices((images2, labels2))
db_val = db_val.shuffle(500).map(preprocess).batch(batchsz)

images3, labels3, table3 = load_test('test', mode='test')
print('testlabels:', len(labels3))
print('testtable:', table3)
db_test = tf.data.Dataset.from_tensor_slices((images3, labels3))
db_test = db_test.shuffle(500).map(preprocess).batch(batchsz)

net = DPN92(input_shape=(224, 224, 3), weights='imagenet', include_top=False, pooling='avg')
newnet = keras.Sequential([
    net,  
    layers.Dense(2668, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(rate=0.5),
    layers.Dense(3, activation='softmax')
])
newnet.build(input_shape=(None, 224, 224, 3))
newnet.summary()

early_stopping = EarlyStopping(
    monitor='val_accuracy',
    min_delta=0.001,
    patience=5
)

# Learning Rate Reducer
learn_control = ReduceLROnPlateau(
    monitor='val_accuracy',
    patience=3,
    verbose=1,
    factor=0.5,
    min_lr=1e-7
)


tic = time.time()
# stage 1
newnet.compile(optimizer=optimizers.Adam(lr=1e-4),
               loss=losses.CategoricalCrossentropy(from_logits=True),
               metrics=['accuracy'])

history = newnet.fit(db_train, validation_data=db_val, validation_freq=1, verbose=2, epochs=5,
             callbacks=[learn_control, early_stopping])

# stage 2
newnet.compile(optimizer=optimizers.Adam(lr=1e-5),
               loss=losses.CategoricalCrossentropy(from_logits=True),
               metrics=['accuracy'])

history1 = newnet.fit(db_train,validation_data=db_val, validation_freq=1, verbose=2, epochs=30,
             callbacks=[learn_control, early_stopping])

save_name = 'ddsm_dpn_3cls'

toc = time.time()
print('Time for training is---'+str(toc-tic)+'seconds---------')

history = history.history
history1 = history1.history
print(history.keys())
print('val_accuracy', history['val_accuracy'], history1['val_accuracy'])
print('accuracy', history['accuracy'], history1['accuracy'])
print('val_loss', history['val_loss'], history1['val_loss'])
print('loss', history['loss'], history1['loss'])

test_loss, test_acc = newnet.evaluate(db_test)
print(test_loss, test_acc)
newnet.save(save_name+'.h5')

plt.figure()
returns = history['val_accuracy']
returns1 = history1['val_accuracy']
returns = np.append(returns, returns1)
plt.plot(np.arange(len(returns)), returns, label='val-acc')
returns = history['accuracy']
returns1 = history1['accuracy']
returns = np.append(returns, returns1)
plt.plot(np.arange(len(returns)), returns, label='train-acc')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('acc')
plt.savefig(save_name+'_acc.svg')

plt.figure()
returns = history['val_loss']
returns1 = history1['val_loss']
returns = np.append(returns, returns1)
plt.plot(np.arange(len(returns)), returns, label='val-loss')
returns = history['loss']
returns1 = history1['loss']
returns = np.append(returns, returns1)
plt.plot(np.arange(len(returns)), returns, label='train-loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.savefig(save_name+'_loss.svg')

