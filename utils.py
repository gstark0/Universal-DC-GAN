from PIL import Image
import numpy as np
import os
import re

from config import data_folder
data_path = data_folder

def smooth(min, max):
	smoothing_value = np.random.uniform(low=min, high=max)
	return smoothing_value

def image_to_array(image_path):
	with Image.open(image_path) as image:    
		image_array = np.fromstring(image.tobytes(), dtype=np.uint8)
		image_array = image_array.reshape((image.size[1], image.size[0], 3))
		image_array = image_array.astype(np.float)
		image_array = image_array/255
	return image_array

def get_random_batch(dataset_name, batch_size):
	# Currently for JPG/JPEG only
	#files = [f.lower() for f in os.listdir('./%s%s' % (data_path, dataset_name)) if re.match(r'.*\.(jpg|jpeg|png|gif|tiff)', f)]
	dataset_path = './' + data_path + dataset_name
	image_names = [f.lower() for f in os.listdir(dataset_path) if re.match(r'.*\.(jpg|jpeg)', f)]
	random_image_names = np.random.choice(image_names, batch_size)

	images = []
	for image_name in random_image_names:
		images.append(image_to_array('./%s/%s' % (dataset_path, image_name)))

	return images