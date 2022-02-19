from lanes import *
import cv2
import numpy as np
import matplotlib.pyplot as plt

imagename = "testimage7.jpeg"
inputimage = cv2.imread(imagename)

canny,color,combined = thresholding(inputimage)

# plt.imshow(combined,cmap='gray')
# plt.show()

verts = np.array([[[552,443],[309,671],[1216,671],[648,443]]],dtype=np.int32) # testimage7
verts2 = np.array([[[338,252],[590,392],[117,392],[293,252]]],dtype=np.int32) # testimage 5
verts3 = np.array([[[87,219],[196,132],[220,132],[376,219]]],dtype=np.int32) # testimage 6
masked = mask(combined,verts=verts)

# plt.imshow(masked,cmap='gray')
# plt.show()

plt.figure()

topdown,tmatrix = perspective_transform(masked)#,verts=verts)
img,curves,ploty = sliding_window(topdown,nwindows=10,margin=50,minpix=1,draw_windows=True)
final_image = draw_lane(inputimage, curves[0],curves[1])
plt.imshow(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))
plt.show()