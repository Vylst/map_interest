from PIL import Image

img = Image.open('white.png')
img = img.convert("RGBA")
datas = img.getdata()

threshold = 200

newData = []
for item in datas:
	#if item[0] >= threshold and item[1] >= threshold and item[2] >= threshold:
		newData.append((item[0], item[1], item[2], 210))
	#else:
	#	newData.append(item)

img.putdata(newData)
img.save("white_c.png", "PNG")
