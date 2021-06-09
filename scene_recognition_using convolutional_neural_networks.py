import os
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

#Function to keep next 200 images from each class for training data
def training_data():
    label = []
    cntr = 0
    # os.walk will access all the sub-folders itself from scene_categories
    path = '...scene_categories' #add data folder path here
    for root, dirs, files in os.walk(path, topdown=True):
        x = 0
        label.append(cntr)
        cntr += 1
        for names in files:
            x += 1
            if x <= 200:
                pass
            elif x >= 200:
                print(os.path.join(root, names))
                os.remove(os.path.join(root, names))


#Function to keep next 20 images from each class as testing data
def testing_data():
    label = []
    cntr = 0
    # os.walk will access all the sub-folders itself from scene_categories
    path = '...scene_categories' #add data folder path here
    for root, dirs, files in os.walk(path, topdown=True):
        x = 0
        label.append(cntr)
        cntr += 1
        for names in files:
            x += 1
            if x < 200 or x >= 220:
                print(os.path.join(root, names))
                os.remove(os.path.join(root, names))
            elif x >= 200 or x < 220:
                pass


# generating data using Image Generator
datagen = ImageDataGenerator(rescale=1./255)
# prepare an iterators for each dataset
train_it = datagen.flow_from_directory('...scene_categories',
                                       target_size = (256, 256), batch_size= 50, class_mode='binary')
test_it = datagen.flow_from_directory('...scene_categories',
                                      target_size = (256, 256), class_mode='binary')
# confirm the iterator works
x_train, y_train = train_it.next()
x_test, y_test = test_it.next()
print('Batch shape=%s, min=%.3f, max=%.3f' % (x_train.shape, x_train.min(), x_train.max()))


#Callback to stop iterating when accuracy of training data reaches 70% to avoid overfitting
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('acc')>0.7):
            print("\nReached 70% accuracy so cancelling training!")
            self.model.stop_training = True


callbacks = myCallback()

# Creating a CNN with 5 convolution layers followed by a DNN with 1 hidden layer with 512 neurons and 15 neurons as output for 15 categories
# All parameters are chosen by hit and trial method
model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 256x256 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(256, 256, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fourth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fifth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    # 15 output neuron since we have 15 categories
    tf.keras.layers.Dense(15, activation='softmax')
])

model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#Training Model on 3000 training images
history = model.fit(train_it, steps_per_epoch=4, epochs=100, callbacks = [callbacks])

#Evaluating Performance on Test Data
print("\nDone Training. Evaluating Performance on Testing Data")
model.evaluate(test_it)