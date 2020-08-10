from tensorflow.keras import layers
from tensorflow.keras import models
import tensorflow as tf

def get_resnet_backbone():
    base_model = tf.keras.applications.ResNet50(include_top=False,
    	weights=None, input_shape=(None, None, 3))
    base_model.trainabe = True

    inputs = layers.Input((None, None, 3))
    h = base_model(inputs, training=True)
    h = layers.GlobalAveragePooling2D()(h)

    backbone = models.Model(inputs, h)

    return backbone

def get_projection_prototype(prototype_dimension):
	inputs = layers.Input((2048, ))
	projection_1 = layers.Dense(2048)(inputs)
	projection_1 = layers.BatchNormalization()(projection_1)
	projection_1 = layers.Activation("relu")(projection_1)

	projection_2 = layers.Dense(128)(projection_1)

	projection_2_normalize = tf.math.l2_normalize(projection_2, axis=1, name='projection')

	prototype = layers.Dense(prototype_dimension, use_bias=False, name='prototype')(projection_2_normalize)

	return models.Model(inputs=inputs,
		outputs=[projection_2_normalize, prototype])
