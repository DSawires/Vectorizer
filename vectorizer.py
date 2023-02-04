from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import scipy
import time
import random
import collections
import svgwrite
import sys

start_time = time.time()
dic = collections.defaultdict(int)

def leftOf(a, b, c):
        signedArea = (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
        return (signedArea > 0)


def get_edge_pts(color, threshold, img, pts, p):
    global dic
    width = img.size[0]
    height = img.size[1]

    matrix = np.zeros((height, width)).astype(int)

    for row in range(0, height):
        for col in range(0, width):
            matrix[row][col] = img.getpixel((col, row))[color]

    for row in range(0, height-1):
        for col in range(0, width-1):
            if (abs(matrix[row][col] - matrix[row + 1][col]) >= threshold) or (abs(matrix[row][col] - matrix[row][col+1]) >= threshold):
                if random.random() <= p:
                    if dic[col, row] == 0:
                        pts.append([col, row])
                        dic[col,  row] = 1

    return pts

def vectorize(img, n, p):
    im = Image.open(img)

    width = im.size[0]
    height = im.size[1]

    threshold = 20
    print(str(time.time() - start_time) + " - Started edge recognition")

    pts = [0 for i in range(n)]
    for i in range(n):
        pts[i] = [random.randint(0,width),random.randint(0,height)]

    pts = get_edge_pts(0, threshold, im, pts, p)
    print(str(time.time() - start_time) + " - Red Done!")

    pts = get_edge_pts(1,threshold,im, pts, p)
    print(str(time.time() - start_time) + " - Green Done!")

    pts = get_edge_pts(2,threshold,im, pts, p)
    print(str(time.time() - start_time) + " - Blue Done!")

    x = np.array(pts)
    tri = scipy.spatial.Delaunay(x)
    print(str(time.time() - start_time) + " - Triangulated!")
    print(str(len(x[tri.simplices])) + " Triangles")

    dwg = svgwrite.Drawing('test.svg')

    for triangle in x[tri.simplices]:
        x_cor = (triangle[0][0] + triangle[1][0] + triangle[2][0])/3
        y_cor = (triangle[0][1] + triangle[1][1] + triangle[2][1])/3
        x_cords = [ triangle[0][0], triangle[1][0], triangle[2][0] ]
        y_cords = [ triangle[0][1], triangle[1][1], triangle[2][1] ]
        pixel = im.getpixel((x_cor, y_cor))

        a = (str(triangle[0][0]), str(triangle[0][1]))
        b = (str(triangle[1][0]), str(triangle[1][1]))
        c = (str(triangle[2][0]), str(triangle[2][1]))
        dwg.add(svgwrite.shapes.Polygon(points=[a, b, c], stroke=svgwrite.rgb(pixel[0], pixel[1], pixel[2], "RGB"), fill=svgwrite.rgb(pixel[0], pixel[1], pixel[2], "RGB")))
    print(str(time.time() - start_time) + " - Colored Triangles!")

    dwg.save()


if len(sys.argv) != 4:
    print("Usage: python vectorizer.py FileName \"No of Random Dots\" \"Edge Weights out of 1\"")
else:
    vectorize(sys.argv[1], int(sys.argv[2]), float(sys.argv[3]))