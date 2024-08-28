import numpy as np
import matplotlib.pyplot as plt

arr = np.zeros((749,1200,3), dtype=np.uint8)
imgsize = arr.shape[:2]
print(imgsize)

centerX = 110
centerY = 231

innerColor = (0, 255, 0)
outerColor = (255, 255, 255)
for y in range(imgsize[1]):
    for x in range(imgsize[0]):
        #Find the distance to the center
        distanceToCenter = np.sqrt((x - centerX) ** 2 + (y - centerY) ** 2)

        #Make it on a scale from 0 to 1innerColor
        distanceToCenter = distanceToCenter / (np.sqrt(2) * imgsize[0]/2)

        #Calculate r, g, and b values
        r = outerColor[0] * distanceToCenter + innerColor[0] * (1 - distanceToCenter)
        g = outerColor[1] * distanceToCenter + innerColor[1] * (1 - distanceToCenter)
        b = outerColor[2] * distanceToCenter + innerColor[2] * (1 - distanceToCenter)
        # print r, g, b
        arr[x, y] = (int(r), int(g), int(b))

plt.imshow(arr, cmap='gray')
plt.show()

#346 456 387 518
#110 131
