import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

#import model
modelPath = 'E:/mythesis/03-Classification/Models/TSModelV3' #<-- replace with your model directory
model = keras.models.load_model(modelPath)

def returnComp(img):
    b, g, r = cv2.split(img)
    return r, g, b

def thresholdNew(img, lower_thresh, upper_thresh):
    _, lower_mask = cv2.threshold(img, lower_thresh, 255, cv2.THRESH_BINARY)
    _, upper_mask = cv2.threshold(img, upper_thresh, 255, cv2.THRESH_BINARY_INV)
    result = cv2.bitwise_and(lower_mask, upper_mask)
    return result

def findContour(img, min_area_threshold):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area_threshold]
    return filtered_contours

def boundaryBox(img,contours):
	x, y, w, h = cv2.boundingRect(np.vstack(contours))
	img = cv2.rectangle(img, (x-5,y-5), (x+w+5,y+h+5), (0,255,0), 2)
	sign = img[y-10:y+h+10, x-10:x+w+10]
	return img, sign

def preprocessingImageToClassifier(image=None,imageSize=40,mu=127.70122641509434,std=74.58174034218196):
    image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    image = cv2.resize(image,(imageSize,imageSize))
    image = (image - mu) / std
    image = image.reshape(1,imageSize,imageSize,1)
    return image

def predict(sign):
	img = preprocessingImageToClassifier(sign,imageSize=40)
	return np.argmax(model.predict(img))

def prob(sign):
    img = preprocessingImageToClassifier(sign, imageSize=40)
    round_prob = round(np.max(model.predict(img)) * 100, 2)
    return round_prob

#--------------------------------------------------------------------------
labelToText = { 0:"Stop",
    			1:"Slow",
    			2:"Crosswalk"}

#cap=cv2.VideoCapture(0) # capture video from camera
file1='E:/mythesis//Test Video/FrontSchool.mp4' #<-- replace with your video directory
file2='E:/mythesis//Test Video/InSchool.mp4' #<-- replace with your video directory
file3='E:/mythesis/Test Video/joko.mp4' #<-- replace with your video directory
cap=cv2.VideoCapture(file3)

while(True):
    #_, frame = cap.read() # read captured video from camera
    ret, frame = cap.read()
    if not ret:
        break  # If there are no more frames to read, break the loop

    Red, Green, Blue = returnComp(frame) #specify the RGB of the image
    thresholded_r = thresholdNew(Red, 185, 220)  #<-- redness thresholds
    thresholded_g = thresholdNew(Green, 3, 50)  #<-- greenness thresholds
    thresholded_b = thresholdNew(Blue, 30, 80)   #<-- blueness thresholds

    common = cv2.bitwise_and(thresholded_r, cv2.bitwise_and(thresholded_g, thresholded_b))
    common_contours = findContour(common, min_area_threshold=90)

    try:
        if  140 < cv2.contourArea(np.vstack(common_contours)) < 1200:
            #print(cv2.contourArea(Range))
            img,sign = boundaryBox(frame,common_contours)
            x, y, w, _ = cv2.boundingRect(np.vstack(common_contours))
            img = cv2.putText(img, labelToText[predict(sign)] + str(prob(sign)) + '%', (x+w+10, y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36, 255, 12), 2)
            cv2.imshow('frame',img)
            #cv2.imwrite('E:/mythesis/result/joko result/frame_{:04d}.png'.format(int(cap.get(cv2.CAP_PROP_POS_FRAMES))), img)
            print("Now,I see:",labelToText[predict(sign)])
        else:
            cv2.imshow('frame',frame)
            #cv2.imwrite('E:/mythesis/result/joko result/frame_{:04d}.png'.format(int(cap.get(cv2.CAP_PROP_POS_FRAMES))), frame)
    except Exception as e:
        print("Exception:", e)
        cv2.imshow('frame', frame)
        #cv2.imwrite('E:/mythesis/result/joko result/frame_{:04d}.png'.format(int(cap.get(cv2.CAP_PROP_POS_FRAMES))), frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
