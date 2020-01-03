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