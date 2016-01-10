import numpy as np
import cv2
import copy
import os
# from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 10
FLANN_INDEX_KDTREE = 0
MAX_BEER = 10

#surf = cv2.xfeatures2d.SURF_create(100, 8, 10, False, False)
detector = cv2.xfeatures2d.SIFT_create()
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

def beer(filename):
    img2 = cv2.imread(filename,0)
    # img2 = cv2.resize(img2, (0,0), fx=0.5, fy=0.5)

    print img2.shape

    rects = []

    for i in range(0, MAX_BEER):
        mx = 0
        mimg = None
        mkp = None
        mdes = None
        mgood = []

        kp2, des2 = detector.detectAndCompute(img2,None)

        for root, dirs, files in os.walk('Beer/'):
            for name in files:
                print "Process ("+name+")"
                img1 = cv2.imread('Beer/'+name,0)
                kp1, des1 = detector.detectAndCompute(img1,None)
                matches = flann.knnMatch(des1,des2,k=2)

                good = []
                for m,n in matches:
                    if m.distance < 0.65*n.distance:
                        good.append(m)

                if len(good)>mx:
                    mx = len(good)
                    mimg = img1
                    mkp = kp1
                    mdes = des1
                    mgood = good

                del img1
                del kp1
                del des1
                del matches
                del good

            if mx>MIN_MATCH_COUNT:

                print "Found Beer by "+str(mx)

                src_pts = np.float32([ mkp[m.queryIdx].pt for m in mgood ]).reshape(-1,1,2)
                dst_pts = np.float32([ kp2[m.trainIdx].pt for m in mgood ]).reshape(-1,1,2)

                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
                if mask==None:
                    return copy.deepcopy(rects)

                matchesMask = mask.ravel().tolist()

                h,w = mimg.shape
                pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
                dst = cv2.perspectiveTransform(pts,M)

                min_x = 9999999
                max_x = 0
                min_y = 9999999
                max_y = 0

                for i in range(0, 4):
                    min_x = max(0, min(min_x, dst[i][0][0]))
                    max_x = min(img2.shape[1]-1, max(max_x, dst[i][0][0]))
                    min_y = max(0, min(min_y, dst[i][0][1]))
                    max_y = min(img2.shape[0]-1, max(max_y, dst[i][0][1]))

                rect = {"x":int(min_x), "y":int(min_y), "w":int(max_x-min_x), "h":int(max_y-min_y)}
                rects.append(rect)
                print rects
                img2[rect["y"]:rect["y"]+rect["h"], rect["x"]:rect["x"]+rect["w"]] = 0
                cv2.imwrite("out.jpg", img2)
                img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

            else:
                cv2.imwrite("out.jpg", img2)
                print "Not enough matches are found - %d/%d" % (len(mgood),MIN_MATCH_COUNT)
                matchesMask = None
                return rects

            draw_params = dict(matchColor = (0,255,0),
                               singlePointColor = None,
                               matchesMask = matchesMask,
                               flags = 2)

            # img3 = cv2.drawMatches(mimg, mkp, img2, kp2, mgood, None,**draw_params)

            # plt.imshow(img3, 'gray'),plt.show()

    return rects
