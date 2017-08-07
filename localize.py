import sys
sys.path.append('/usr/local/lib/python2.7/dist-packages')
import cv2

def traffic_sign_locate(img, classifier, downscale):

    frame = img.copy()
    frame = cv2.morphologyEx(frame, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
                             )

    frame = cv2.morphologyEx(frame, cv2.MORPH_CLOSE,
                    cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
                    )

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.equalizeHist(frame, frame)

    scaledsize = (frame.shape[1]/downscale, frame.shape[0]/downscale)
    scaledframe = cv2.resize(frame, scaledsize)

    # Detect signs in downscaled frame
    signs = classifier.detectMultiScale(scaledframe,
                                        1.1,
                                        5,
                                        0,
                                        (10, 10),
                                        (200, 200))
    locations = []
    for sign in signs:
        locations.append([i * downscale for i in sign])

    return locations