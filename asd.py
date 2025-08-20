import tensorflow as tf
import numpy as np

x=np.arange(1,11,1).reshape(5,2)
y=np.arange(3,20,4).reshape(5,1)

model=tf.keras.Sequential([
    tf.keras.layers.Dense(1,activation="linear",input_shape=(2,))
])

model.compile(optimizer="sgd",loss="mse")
model.fit(x,y,epochs=100,verbose=0)
print(model.predict(np.array([[10,5]])))