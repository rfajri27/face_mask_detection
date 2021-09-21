import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input
import numpy as np
import matplotlib.pyplot as plt

physical_device = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_device[0], True)

def model():
    # Data Preprocessing
    TRAINING_DIR = "dataset/training/"
    VALIDATION_DIR = "dataset/validation/"
    batch_size = 32

    train_datagen = ImageDataGenerator(
        rescale=1 / 255,
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    train_generator = train_datagen.flow_from_directory(TRAINING_DIR,
                                                        batch_size=batch_size,
                                                        class_mode='categorical',
                                                        target_size=(224, 224))

    validation_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR,
                                                                  batch_size=10,
                                                                  class_mode='categorical',
                                                                  target_size=(224, 224))

    # Create Model
    pre_trained_model = MobileNetV2(weights="imagenet", include_top=False,
                                    input_tensor=Input(shape=(224, 224, 3)))

    for layer in pre_trained_model.layers:
        layer.trainable = False

    last_output = pre_trained_model.output

    x = tf.keras.layers.AveragePooling2D(pool_size=(7, 7))(last_output)
    x = tf.keras.layers.Flatten(name="flatten")(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(2, activation="softmax")(x)

    model = tf.keras.models.Model(pre_trained_model.input, x)

    int_lr = 1e-4
    num_epochs = 20

    optimizer = tf.optimizers.Adam(lr=int_lr, decay=int_lr / num_epochs)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Training

    H = model.fit(train_generator,
                  steps_per_epoch=20,
                  epochs=num_epochs,
                  validation_data=validation_generator,
                  validation_steps=3)

    return model, H, num_epochs

if __name__ == '__main__':
    model, H, epoch = model()
    model.save("model.h5")

# Plot Loss
plt.style.use("ggplot")
plt.figure(figsize=(10, 5))
plt.plot(np.arange(0, epoch), H.history["loss"], label="training")
plt.plot(np.arange(0, epoch), H.history["val_loss"], label="validation")
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("images/plot.png")

# Plot Accuracy
plt.figure(figsize=(10, 5))
plt.plot(np.arange(0, epoch), H.history["accuracy"], label="training")
plt.plot(np.arange(0, epoch), H.history["val_accuracy"], label="validation")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("images/plot_acc.png")
