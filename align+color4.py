import numpy as np
import cv2
from matplotlib import pyplot as plt
import math


def deskew():
    im_out = cv2.warpPerspective(skewed_image, np.linalg.inv(M), (orig_image.shape[1], orig_image.shape[0]))
    #this performs a perspective transformation - needs input image which gets altered, needs a matrix which says which pixels should go where, and size of output required in pixels
    plt.imshow(im_out, 'gray')
    plt.show()
    return im_out

def deskew_b():
    im_out_b = cv2.warpPerspective(skewed_image_a, np.linalg.inv(M), (orig_image.shape[1], orig_image.shape[0]))
    #this performs a perspective transformation - needs input image which gets altered, needs a matrix which says which pixels should go where, and size of output required in pixels
    plt.imshow(im_out_b, 'gray')
    plt.show()
    return im_out_b

def deskew_c():
    im_out_c = cv2.warpPerspective(orig_image, np.linalg.inv(M), (orig_image.shape[1], orig_image.shape[0]))
    #this performs a perspective transformation - needs input image which gets altered, needs a matrix which says which pixels should go where, and size of output required in pixels
    plt.imshow(im_out_c, 'gray')
    plt.show()
    return im_out_c


orig_image = cv2.imread(r'ictas_r.jpg' , 1) #red
#orig_image2 = orig_image.copy()
#orig_image = cv2.flip(orig_image2,1);
skewed_image = cv2.imread(r'ictas_b.jpg', 1) #green
skewed_image_a = cv2.imread(r'ictas_g.jpg', 1) #blue

skewed_image_flipped = skewed_image.copy()
skewed_image = cv2.flip(skewed_image_flipped,1)

surf = cv2.xfeatures2d.SURF_create(300) #threshold for hessian detector, won't solve our issue probably
kp1, des1 = surf.detectAndCompute(orig_image, None)
kp2, des2 = surf.detectAndCompute(skewed_image, None)
kp3, des3 = surf.detectAndCompute(skewed_image_a, None)

orig_image2 = cv2.drawKeypoints(orig_image, kp1, None)
skewed_image2= cv2.drawKeypoints(skewed_image, kp2, None)
skewed_image_a2= cv2.drawKeypoints(skewed_image_a, kp3, None)

cv2.imshow("orig",orig_image2)
cv2.imshow("skew",skewed_image2)
cv2.imshow("skew_a",skewed_image_a2)
cv2.imwrite('skew_kp2.jpg',skewed_image2)
cv2.imwrite('skew_kp3.jpg',skewed_image_a2)
cv2.waitKey()
cv2.destroyAllWindows()

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=5000)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)
matches_a = flann.knnMatch(des1, des3, k=2)


# store all the good matches as per Lowe's ratio test.

#images 1 and 2
good = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good.append(m)

MIN_MATCH_COUNT = 20
if len(good) >= MIN_MATCH_COUNT:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good
                          ]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good
                          ]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # see https://ch.mathworks.com/help/images/examples/find-image-rotation-and-scale-using-automated-feature-matching.html for details
    ss = M[0, 1]
    sc = M[0, 0]
    scaleRecovered = math.sqrt(ss * ss + sc * sc)
    thetaRecovered = math.atan2(ss, sc) * 180 / math.pi
    print("Calculated scale difference: %.2f\nCalculated rotation difference: %.2f" % (scaleRecovered, thetaRecovered))
    res = deskew()

    kp4, des4 = surf.detectAndCompute(res, None)
    matches_b = flann.knnMatch(des4, des3, k=2)

# for images res and 3
good = []
for m, n in matches_b:
    if m.distance < 0.7 * n.distance:
        good.append(m)

MIN_MATCH_COUNT = 10
if len(good) >= MIN_MATCH_COUNT:
    src_pts = np.float32([kp4[m.queryIdx].pt for m in good
                          ]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp3[m.trainIdx].pt for m in good
                          ]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # see https://ch.mathworks.com/help/images/examples/find-image-rotation-and-scale-using-automated-feature-matching.html for details
    ss = M[0, 1]
    sc = M[0, 0]
    scaleRecovered = math.sqrt(ss * ss + sc * sc)
    thetaRecovered = math.atan2(ss, sc) * 180 / math.pi
    print("Calculated scale difference: %.2f\nCalculated rotation difference: %.2f" % (
        scaleRecovered, thetaRecovered))

    res_b = deskew_b()

    matches_c = flann.knnMatch(des4, des1, k=2)

    # fit the original image to first result!
# for images res and 3
good = []
for m, n in matches_c:
    if m.distance < 0.7 * n.distance:
        good.append(m)

MIN_MATCH_COUNT = 10
if len(good) >= MIN_MATCH_COUNT:
    src_pts = np.float32([kp4[m.queryIdx].pt for m in good
                          ]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp1[m.trainIdx].pt for m in good
                          ]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # see https://ch.mathworks.com/help/images/examples/find-image-rotation-and-scale-using-automated-feature-matching.html for details
    ss = M[0, 1]
    sc = M[0, 0]
    scaleRecovered = math.sqrt(ss * ss + sc * sc)
    thetaRecovered = math.atan2(ss, sc) * 180 / math.pi
    print("Calculated scale difference: %.2f\nCalculated rotation difference: %.2f" % (
        scaleRecovered, thetaRecovered))

    res_c = deskew_c()
    # orig_image[:, :, 0] = 0
    # orig_image[:, :, 2] = 0
    # cv2.imshow("image2", orig_image)
    # cv2.moveWindow("image2",0,0)


    res_c[:, :, 0] = res[:, :, 0] #set red channel 0
    res_c[:, :, 1] = res_b[:, :, 1] #set green channel same as original image but reoriented



    cv2.imshow("image3", res_c)
    cv2.moveWindow("image3",1,0)

    # g = orig_image[:, :, 1]
    # r = np.zeros((100,100))
    # b = res[:, :, 1]

    # final_out = cv2.merge((b,g,r))
    # cv2.imshow("image4", final_out)
    # cv2.moveWindow("image4", 1, 0)


    cv2.waitKey()
    cv2.destroyAllWindows()



else:
    print("Not  enough  matches are found   -   %d/%d" % (len(good), MIN_MATCH_COUNT))
    matchesMask = None