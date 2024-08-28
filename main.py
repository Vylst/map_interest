import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


#ffmpeg -start_number 0 -i %d.png -t 15 -vcodec mpeg4 -b 6400k test.mkv
#ffmpeg -i in.ffconcat -vcodec mpeg4 -b 6400k out.mkv


import cv2
import numpy as np
import time
import imutils
import matplotlib.pyplot as plt

from keras.models import load_model
from keras.preprocessing import image


protoPath = 'deploy.prototxt'
modelPath = 'res10_300x300_ssd_iter_140000.caffemodel'
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
embedder = cv2.dnn.readNetFromTorch('openface_nn4.small2.v1.t7')

true_labels = [[0,0,1,1,0,1,0,0,0,1,0,0],  
		[1,1,1,1,0,1,0,0,1,0,1,0], 
		[0,0,0,0,1,0,1,0,0,0,0,0],  
		[1,0,1,0,0,0,0,1,0,0,0,0],  
		[1,1,0,1,0,0,0,0,0,0,1,0],  
		[0,0,0,0,0,0,0,0,0,0,0,1]]

emotion_model_path = 'VGG19_features_48.h5'
name_labels=["angry","fear","happy","sad","surprise","neutral"]
color_labels=[np.array([1, 0, 0], dtype='float16'), np.array([0, 1, 0], dtype='float16'), np.array([1, 1, 0], dtype='float16'), np.array([0, 0, 1], dtype='float16'), np.array([0, 1, 1], dtype='float16'), np.array([1, 1, 1], dtype='float16')]
emotion_classifier = load_model(emotion_model_path, compile=False)
emotion_classifier.summary()


outer = np.array([0, 0, 0])[None, None, :]


#vs = cv2.VideoCapture(0)
vs = cv2.VideoCapture('video.mov')
print(int(vs.get(cv2.CAP_PROP_FRAME_COUNT)))
vs.set(cv2.CAP_PROP_POS_FRAMES, 117000);



face_counter = 0
five_frame_saver = np.zeros((5,749,1200,3))

idx = 0
while idx < 901*5 +1:
	
	# grab the frame from the threaded video stream
	frame = vs.read()[1]

	#print(frame)
	#print(np.shape(frame))

	# resize and then grab the image dimensions
	frame = imutils.resize(frame, width=1200)

	(h, w) = frame.shape[:2]

	# construct a blob from the image and detect faces
	imageBlob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)
	detector.setInput(imageBlob)
	detections = detector.forward()


	plt.cla()
	plt.clf()
	plt.close()

	total = np.zeros((h,w,3))
	face_counter = 0
	fig, ax = plt.subplots()

	# loop over the face detections
	for i in range(0, detections.shape[2]):

		face_detection_confidence = detections[0, 0, i, 2]

		#Pode-se meter aqui 0.4 para apanhar caras mais viradas ou deturpadas 
		if face_detection_confidence > 0.12:

			# compute the (x, y)-coordinates of the bounding box for the face
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

	
			startX = startX - 10
			endX = endX + 10
			startY = startY - 10
			endY = endY + 10

			# extract the face ROI
			face = frame[startY:endY, startX:endX]
			(fH, fW) = face.shape[:2]



			# ensure the face width and height are sufficiently large
			if fW < 20 or fH < 20:
				continue

			color = (0,255,0)
			if(np.shape(face)[0] != 0 and np.shape(face)[1] != 0):

				img2 = cv2.resize(face, (48, 48))
				img2 = image.img_to_array(img2)
				img2 = np.expand_dims(img2,axis=0)

				emotion_prediction = emotion_classifier.predict(img2)


				average_positive_facs_confidence = np.mean(emotion_prediction[emotion_prediction >= 0.5])


				emotion_prediction[emotion_prediction >= 0.5] = 1
				emotion_prediction[emotion_prediction < 0.5] = 0
				emotion_prediction=np.array(emotion_prediction)
				for i in range(len(emotion_prediction)):
					for j in range(len(true_labels)):
						match=True
						for k in range(len(true_labels[j])):
							if true_labels[j][k]!=emotion_prediction[i][k]:
								match=False
								break

						if match:
							#As the average_facs_conf increases, the color must be stronger. So the non-255 values in RGB must be closer to 255
							factor = 1 - average_positive_facs_confidence
							inner = color_labels[j]
							inner[inner == 0] = factor
							#print(inner)
							inner = inner[None, None, :]
							emotion_text=name_labels[j]
							break
						else:

							#white (neutral) does not fade, it's always white, duh..
							inner = color_labels[5]
							inner = inner[None, None, :]
							emotion_text="neutral"



				x_axis = np.linspace(0, 1, 1200)
				y_axis = np.linspace(0, 1, 749)


				centerX = (startX + (endX - startX)) / w
				centerY = (startY + (endY - startY)) / h


				xx, yy = np.meshgrid(x_axis, y_axis)
				#Quanto maior o factor inicial, menor é o círculo. Logo se a face_detect_conf aumentar, (1-face_detect_conf) vai aumentar arr também
				elongate_factor = (12*(1-face_detection_confidence))
				if(elongate_factor < 3):
					elongate_factor = 3
				arr = elongate_factor*np.sqrt((xx - centerX) ** 2 + (yy - centerY) ** 2)

				

				arr[arr>1] = 1
				arr = arr[:, :, None]
				arr = arr * outer + (1 - arr) * inner
				arr = cv2.cvtColor(arr.astype('float32'), cv2.COLOR_BGR2RGB)
				cv2.circle(arr, (int(centerX*w), int(centerY*h)), 5, (0,0,0), -1)
			
				#cv2.imshow("arr", aa)
				#cv2.waitKey()


				#plt.imshow(total/face_counter, cmap='gray')
				#plt.show()
				
				total = total + arr
				

				#c1 = plt.Circle((endX-startX,endY-startY), 15, color='black')
				#ax.add_artist(c1)

				# Draw
				cv2.putText(frame, emotion_text, (startX,startY), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
				cv2.rectangle(frame, (startX,startY), (endX,endY), color, 2)
						

				face_counter += 1


	#print(np.shape((total - np.min(total))/np.ptp(total)))
	five_frame_saver[idx%5] = (total - np.min(total))/np.ptp(total)

	idx += 1

	if(idx%5 == 0):
		
		b = sum(five_frame_saver/5)
		cv2.imwrite('maps/' + str(idx/5) + '.png', b*255)

		#idx = 0
	
	#plt.imsave('maps/' + str(idx) + '.png', b, cmap='gray', vmin=0, vmax=255)
	#plt.imshow(total/face_counter, cmap='gray')
	#plt.show()



	# show the output frame
	#frame = cv2.flip(frame, +1)
	#cv2.imshow("Frame", frame)




	# if the `q` key was pressed, break from the loop
	key = cv2.waitKey(1) & 0xFF	
	if key == ord("q"):
		break

	

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()






