
import numpy as np
import matplotlib.pyplot as plt
import cv2

from PIL import Image

centerX = 600
centerY = 374
h = 749
w = 1200

fig, ax = plt.subplots()


x_axis = np.linspace(0, 1, 600)
y_axis = np.linspace(0, 1, 300)

xx, yy = np.meshgrid(x_axis, y_axis)
#arr = 1.5*np.sqrt( ((xx - centerX/w) ** 2)/20 + ((yy - centerY/h) ** 2)/20  )
arr = 3*np.sqrt( (xx -0.5) ** 2 + (yy-0.5) ** 2  )


inner = np.array([0, 0, 1])[None, None, :]
outer = np.array([1, 0, 0])[None, None, :]


arr[arr>1] = 1
#arr /= arr.max()
arr = arr[:, :, None]
arr = arr * outer + (1 - arr) * inner




#c1 = plt.Circle((600,374.5), 15, color='black')
#ax.add_artist(c1)

#plt.imsave('teste.png', arr, cmap='gray')

arr = cv2.cvtColor(arr.astype('float32'), cv2.COLOR_BGR2RGB)
#aa = arr.copy()
#cv2.circle(aa, (centerX, centerY), 10, (0,0,0), -1)

#b = ((aa+arr) - np.min((aa+arr)))/np.ptp((aa+arr))
cv2.imshow('arr',arr)
cv2.waitKey()
#cv2.imwrite('white.png', arr*255)

#plt.imshow(arr, cmap='gray')
#fig.savefig('teste.png')
#plt.show()

