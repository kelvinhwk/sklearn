import PIL
from PIL import Image

def rectimg(img):
	width, height = img.size
	
	left, top, right, low = 0, 0, width, height
	
	if (width > height):
		delta = width - height
		left += delta/2
		right -= delta/2
	elif (width < height):
		delta = height - width
		top += delta/2
		low -= delta/2
	
	return img.crop((left, top, right, low))
	

	


#help(Image.Image.transform)

#help(Image.open)

myImage = Image.open('test.jpg').convert('RGBA')

data = myImage.getdata()
#print(list(data)[0:100])

# 
bands = myImage.getbands()
print(bands)


trans = myImage.transform((300,300), Image.EXTENT, [-125,0,875,1000], fillcolor=(0,0,0,0))
trans.save('aaa.png')
#trans.show()
data = trans.getdata()
#print(list(data)[0:1000])
	
gray = myImage.convert('L')
data = gray.getdata()
print(list(data)[0:1000])

#rect = rectimg(myImage)
#rect.show()

'''
# thumbnail
clone = myImage.copy()
clone.thumbnail(size=(100,100))
clone.show()

# resize
resizeImage = myImage.resize((300,300), Image.ANTIALIAS)
resizeImage.show()
'''


#print(myImage.getbbox())



grayImage = myImage.convert('L')

#myImage.show()

#grayImage.show()
