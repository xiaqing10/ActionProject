import cv2


class DrawObjects(object):
    
    def __init__(self, topology):
        self.topology = topology
        
    def __call__(self, image, object_counts, objects, normalized_peaks, _type= "pose"):

        new_image = image.copy()
        topology = self.topology
        height = image.shape[0]
        width = image.shape[1]
        
        K = topology.shape[0]
        count = int(object_counts[0])
        K = topology.shape[0]
    
        s_l = list()

        if _type == "hand":
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)

        for i in range(count):
            obj = objects[0][i]
            C = obj.shape[0]
            for j in range(C):

                k = int(obj[j])
                if k >= 0:
                    peak = normalized_peaks[0][j][k]
                    x = round(float(peak[1]) * width)
                    y = round(float(peak[0]) * height)
                    cv2.circle(image, (x, y), 3, color, 2)

                    roi_dis = 60
                    if j == 9 or j == 10:
                        img = new_image[y-roi_dis:y+roi_dis, x-roi_dis:x+roi_dis]
                        s_l.append((img, x, y, roi_dis))

            for k in range(K):
                c_a = topology[k][2]
                c_b = topology[k][3]
                if obj[c_a] >= 0 and obj[c_b] >= 0:
                    peak0 = normalized_peaks[0][c_a][obj[c_a]]
                    peak1 = normalized_peaks[0][c_b][obj[c_b]]
                    x0 = round(float(peak0[1]) * width)
                    y0 = round(float(peak0[0]) * height)
                    x1 = round(float(peak1[1]) * width)
                    y1 = round(float(peak1[0]) * height)

                    if  _type == "hand":
                        print("line")
                    cv2.line(image, (x0, y0), (x1, y1), color, 2)
            import random
            if _type == "hand":
                cv2.imwrite("imgs/" + str(random.random()) + "hand.jpg", image)
        return s_l