from PIL import Image
import numpy as np
import sys
import os
import re
import glob
import matplotlib.pyplot as plt
import pickle

def img_to_array(image_path, channels):
	with Image.open(image_path) as image:  
		#print(image_path)       
		im_arr = np.fromstring(image.tobytes(), dtype=np.uint8)
		try:
			im_arr = im_arr.reshape((image.size[1], image.size[0], channels))
		except:
			os.remove(image_path)
			return None
		im_arr = im_arr[:, :, :-1].astype(np.float)
		im_arr = im_arr.astype(np.float)
		im_arr = 2*(im_arr/255) - 1
	return im_arr

def main():
	if len(sys.argv) > 2:
		folder_name = sys.argv[1]
		channels = int(sys.argv[2])
	else:
		print('No folder or channels number name specified!')
		sys.exit()

	files = [f.lower() for f in os.listdir('./%s' % folder_name) if re.match(r'.*\.(jpg|jpeg|png|gif|tiff)', f)]
	print(files)
	images = []
	for image_name in files:
		img = img_to_array('./%s/%s' % (folder_name, image_name), channels)
		if img is None:
			pass
			print(image_name, 'Image removed')
		else:
			images.append(img)

	#with open('dataset2.pickle', 'wb') as p:
		#pickle.dump(images, p)

if __name__ == '__main__':
	main()