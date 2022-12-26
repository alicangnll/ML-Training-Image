import tensorflow as tf

print("Tensorflow Version : " + tf.version.VERSION)

#Global Veriler
image_genislik = int(150)
image_uzunluk = int(150)
image_renkanal = int(3)
rotasyon_acisi = int(40)
width_shift_range = float(0.2)
height_shift_range = float(0.2)
shear_range = float(0.2)
zoom_range = float(0.2)
batch_size = int(64)
epochs = int(5000)
horizontal_flip = True
max_acc = float(0.96)
max_val_acc = float(0.80)

fill_mode = 'nearest'
egitim_klasor = './tf_files/train'
dogrulama_klasor = './tf_files/validation'
model_dosyasi = 'model.h5'

def model_olustur():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='tanh', input_shape=(image_genislik, image_uzunluk, image_renkanal)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='tanh'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='tanh'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='tanh'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512, activation='tanh', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.RMSprop(lr=0.0001), metrics=['accuracy'])
    print(model.summary())
    return model

def topla_data():
    egit_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
    rotation_range=rotasyon_acisi,
    width_shift_range=width_shift_range,
    height_shift_range=height_shift_range,
    shear_range=shear_range,
    zoom_range=zoom_range,
    horizontal_flip=horizontal_flip,
    fill_mode=fill_mode)
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    egitim_olusturucu = egit_datagen.flow_from_directory(
        egitim_klasor,
        target_size=(image_genislik, image_uzunluk),
        batch_size=batch_size,
        class_mode='sparse')
    dogrulama_olusturucu = test_datagen.flow_from_directory(
        dogrulama_klasor,
        target_size=(image_genislik, image_uzunluk),
        batch_size=batch_size,
        class_mode='sparse')
    return egitim_olusturucu, dogrulama_olusturucu

def omurga():
    model = model_olustur()
    model_olusturucu, model_dogrulayici = topla_data()

    # Hazır Kod geliyor...
    class MyCustomCallback(tf.keras.callbacks.Callback):
      def on_epoch_end(self, epoch, logs=None):
          print(' === epoch {}, acc{:7.4f}, val_acc{:7.4f}.'.format(epoch, logs['accuracy'], logs['val_accuracy']))
          if((epoch % 100) == 0):
            f_name = model_dosyasi
            name_pos = f_name.find(".h5")
            f_name = f_name[0:name_pos]
            keras_file = f_name + "." + str(epoch) + ".h5"
            tf.keras.models.save_model(model, keras_file)
          # Set criteria to stop training.
          if ((logs['val_accuracy'] > max_val_acc) or (logs['accuracy'] > max_acc)):
            self.stopped_epoch = epoch
            self.model.stop_training = True

    model.fit(
        model_olusturucu,
        epochs=int(epochs),
        validation_data=model_dogrulayici,
        callbacks=[MyCustomCallback()],
        verbose=1)

    tf.keras.models.save_model(model, model_dosyasi)

omurga()