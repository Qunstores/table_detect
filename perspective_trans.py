#!/usr/bin/env python

# import the necessary packages
#from pyimagesearch.transform import four_point_transform
#from pyimagesearch import imutils
from skimage.filters import threshold_adaptive #NOTE: rimossa s da filters
import numpy as np
import argparse
import cv2

def print_dimension(img):
	print("image shape: " \
		+ "h=" + str(img.shape[0]) \
		+ ", w=" + str(img.shape[1]) \
		+ ", d=" + str(img.shape[2]))

def resize(img, ratio):
	""" height is the reference 
        ratio have to be float """
	dimension=(int(img.shape[1]/ratio),int(img.shape[0]/ratio)) #(w,h)
	print("resizing at: " + str(dimension))
	print(" with ratio: " + str(ratio))
	resized=cv2.resize(img, dimension, interpolation = cv2.INTER_AREA)
	return resized

def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")
 
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
 
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
 
	# return the ordered coordinates
	return rect

def four_point_transform(image, pts):
    
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	print('test')
	#print(rect)
	
	(tl, tr, br, bl) = rect
	tl[0] -= 6
	tl[1] -= 6
	tr[0] += 6
	tr[1] -= 6
	br[0] += 6
	br[1] += 6
	bl[0] -= 6
	bl[1] += 6
	rect[0] = tl
	rect[1] = tr
	rect[2] = br
	rect[3] = bl
	print(rect)
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
 
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
 
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
 
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
 
	# return the warped image
	return warped, rect, dst
   
def adverse_transform( src_img, warped_img, rect, dst ):
	map_matrix = cv2.getPerspectiveTransform(dst, rect)
	#recovered_img = cv2.warpPerspective(warped_img, map_matrix,(2610, 1507))
	#cv2.namedWindow('recovered_img', cv2.WINDOW_NORMAL)
	#cv2.imshow('recovered_img', recovered_img)
	#cv2.waitKey(0)
	#recovered_image = cv2.warpPerspective(warped_img, map_matrix, (src_img.shape[1], src_img.shape[0]))
	return map_matrix
    
def transfer_img( image, debug_model):
    #image = input_img.copy()
    orig = image.copy()
    no_transfer = False
    print('the transfer_img size is: ')
    print_dimension(orig)
    print('the transfer copy size is: ')
    print_dimension(image)
    ratio=float(image.shape[0])/500
    image=resize(image, ratio)
    print('after resize')
    print_dimension(orig)
    print_dimension(image)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 30, 100)
    if debug_model:
        cv2.namedWindow('edged', cv2.WINDOW_NORMAL)
        cv2.imshow("edged", edged)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    #closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT,(10,10)))
    im, cnts, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:1]
    #blank_img = np.zeros(image.shape, np.uint8)
    #cv2.drawContours(blank_img, cnts, -1, (255, 0, 0), 2)
    #cv2.namedWindow('Outline_img', cv2.WINDOW_NORMAL)
    #cv2.imshow("Outline_img", blank_img)
    #global screen_cnt
    print('len cnts:',len(cnts))
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True) 
        screen_cnt = approx
        print('approx out ok:', approx.shape)
        if len(approx) == 4:
            #screen_cnt = approx
            print('screen_cnt out ok:', screen_cnt.shape)
            break
    if len(screen_cnt) != 4:
        map_like = 0;
        no_transfer = True
        return orig, ratio, map_like, no_transfer
    #cv2.drawContours(image, [screenCnt], -1, (255, 0, 0), 2)
    print("STEP 2: Find contours of paper")
    warped, rect_points, dst_points = four_point_transform(orig, screen_cnt.reshape(4, 2) * ratio)
    map_matrix = adverse_transform(orig, warped, rect_points, dst_points)

    '''
    cv2.namedWindow('out', cv2.WINDOW_NORMAL)
    cv2.imshow("out", warped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    return warped, ratio, map_matrix, no_transfer
    
        
def main( args ):
    image = cv2.imread(args["image"])
    debug_model= args["debug"]
    print('the original image size is: ')
    print_dimension(image)
    warped, ratio, map_matrix= transfer_img(image, debug_model)
    # show the original and scanned images
    print("STEP 3: Apply perspective transform")
    #cv2.imshow("Original", image)
    if debug_model:
        cv2.namedWindow('Scanned', cv2.WINDOW_NORMAL)
        cv2.imshow("Scanned", resize(warped, ratio))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required = True, help = "Path to the image to be scanned")
    ap.add_argument("--debug", action = 'store_true')
    args = vars(ap.parse_args())
    main(args)



