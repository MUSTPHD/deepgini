import numpy as np
import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


model = tf.keras.Sequential(
    [
        tf.keras.layers.InputLayer(input_shape=(2, )),
        tf.keras.layers.Dense(2),
    ]
)
model.compile(loss='mse')

W = model.trainable_variables[0]
W.assign(np.array([[1.0, 0.0], [0.0, 1.0]]).T)

input = np.array([[1.0, 2.0], [3.0, 4.0], ], dtype=np.float32)

print("__call__:")
print(model(input))

print("Call:")
print(model.call(tf.convert_to_tensor(input)))

print("Predict:")
print(model.predict(input))
