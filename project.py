import cv2
import numpy as np

import tensorflow as tf

model = tf.keras.models.load_model('keras_model.h5')

camera = cv2.VideoCapture(0)

while True:

    status,frame = camera.read()

    # Modify the input data by:

    if status:
        frame = cv2.flip(frame,1)
        resized_frame = cv2.resize(frame,(224,224))
        resized_frame = np.expand_dims(resized_frame,axis = 0)
        resized_frame = resized_frame/255
        predictions = model.predict(resized_frame)
        rock = int(predictions[0][0]*100)
        paper = int(predictions[0][1]*100)
        scissor = int(predictions[0][2]*100)
        print(f'Rock:{rock}%, Paper:{paper}%, Scissors:{scissor}%')
        cv2.imshow('feed',frame)
        code = cv2.waitKey(1)
        if code==32:
            break
camera.release()
cv2.destroyAllWindows()            