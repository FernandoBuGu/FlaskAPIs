import tensorflow as tf
import numpy as np
import json
import requests

SIZE = 128
MODEL_URI = 'http://localhost:8502/v1/models/pets:predict'
CLASSES = ['Cat', 'Dog']

def get_prediction_F(image_path_S):

	image = tf.keras.preprocessing.image.load_img(
		image_path_S, target_size=(SIZE, SIZE)
	)
	image = tf.keras.preprocessing.image.img_to_array(image)
	image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
	image = np.expand_dims(image, axis=0)#1 x 128 x 128 x 3


	data = json.dumps({
		'instances': image.tolist()
	})
	response = requests.post(MODEL_URI, data=data.encode())#default UTF-8
	result = json.loads(response.text)#returns 0to1 value 
	prediction = np.squeeze(result['predictions'][0])
	class_name = CLASSES[int(prediction > 0.5)]
	return class_name

