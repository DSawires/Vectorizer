import time
import random
import collections
import sys
from PIL import Image
import numpy as np
import scipy
import svgwrite

start_time = time.time()
def get_edge_pts(color, threshold, img, pts, probability):
    dictionary =  collections.defaultdict(int)
    width = img.size[0]
    height = img.size[1]

    matrix = np.zeros((height, width)).astype(int)

    for row in range(0, height):
        for col in range(0, width):
            matrix[row][col] = img.getpixel((col, row))[color]

    for row in range(0, height-1):
        for col in range(0, width-1):
            if (abs(matrix[row][col] - matrix[row + 1][col]) >= threshold) or (abs(matrix[row][col] - matrix[row][col+1]) >= threshold):
                if random.random() <= probability:
                    if dictionary[col, row] == 0:
                        pts.append([col, row])
                        dictionary[col,  row] = 1

    return pts

def vectorize(img, number_of_points, probability):
    image = Image.open(img)

    width = image.size[0]
    height = image.size[1]

    threshold = 20
    print(str(time.time() - start_time) + " - Started edge recognition")

    pts = [0 for i in range(number_of_points)]
    for i in range(number_of_points):
        pts[i] = [random.randint(0,width),random.randint(0,height)]

    pts = get_edge_pts(0, threshold, image, pts, probability)
    print(str(time.time() - start_time) + " - Red Done!")

    pts = get_edge_pts(1,threshold,image, pts, probability)
    print(str(time.time() - start_time) + " - Green Done!")

    pts = get_edge_pts(2,threshold,image, pts, probability)
    print(str(time.time() - start_time) + " - Blue Done!")

    triangles = np.array(pts)
    tri = scipy.spatial.Delaunay(triangles)
    print(str(time.time() - start_time) + " - Triangulated!")
    print(str(len(triangles[tri.simplices])) + " Triangles")

    dwg = svgwrite.Drawing('test.svg')

    for triangle in triangles[tri.simplices]:
        x_cor = (triangle[0][0] + triangle[1][0] + triangle[2][0])/3
        y_cor = (triangle[0][1] + triangle[1][1] + triangle[2][1])/3
        pixel = image.getpixel((x_cor, y_cor))

        points = []*3
        point_a = (str(triangle[0][0]), str(triangle[0][1]))
        point_b = (str(triangle[1][0]), str(triangle[1][1]))
        point_c = (str(triangle[2][0]), str(triangle[2][1]))
        dwg.add(svgwrite.shapes.Polygon(points=[point_a, point_b, point_c],
                                        stroke=svgwrite.rgb(pixel[0], pixel[1], pixel[2], "RGB"),
                                        fill=svgwrite.rgb(pixel[0], pixel[1], pixel[2], "RGB")))
    print(str(time.time() - start_time) + " - Colored Triangles!")

    dwg.save()


if len(sys.argv) != 4:
    print("Usage: python vectorizer.py FileName \"No of Random Dots\" \"Edge Weights out of 1\"\n")
else:
    vectorize(sys.argv[1], int(sys.argv[2]), float(sys.argv[3]))
