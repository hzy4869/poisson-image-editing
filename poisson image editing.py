import cv2
import numpy as np
from scipy.sparse import csc_matrix, linalg

#read the image
#poisson_final is the background, poisson_1 are the two boys
img_source = cv2.imread("poisson_final.jpg")
img_target = cv2.imread("poisson_1.png")

mask = img_target[30:330, 50:550]
source = img_source[50:350, 200:700]

'''find the laplacian of the mask'''
b_s, g_s, r_s = cv2.split(mask)
mask_b = cv2.Laplacian(b_s, cv2.CV_64F)
mask_b = mask_b.reshape(-1)

mask_g = cv2.Laplacian(g_s, cv2.CV_64F)
mask_g = mask_g.reshape(-1)

mask_r = cv2.Laplacian(r_s, cv2.CV_64F)
mask_r = mask_r.reshape(-1)


def add_matrix(ROW, COL, DATA, x, y, tar, i): #col represents the number of the current element
    '''add into the sparse matrix and balance the equation'''
    global row, col, data, count
    if ROW%500 == 0 and ROW == (COL + 1):
        tar[count] -= img_source[50+x, 199][i]
        return
    elif (ROW+1)%500 == 0 and ROW == (COL - 1):
        tar[count] -= img_source[50+x, 700][i]
        return
    elif COL < 0:
        tar[count] -= img_source[49, 200+y][i]
        return
    elif COL > 149999:
        tar[count] -= img_source[350, 200+y][i]
        return
    row.append(ROW)
    col.append(COL)
    data.append(DATA)
    return

def solving(tar, i):
    '''solve the equation by the function splu'''
    global row, col, data, count
    row = []
    col = []
    data = []
    count = 0
    for x in range(300):
        for y in range(500):
            add_matrix(count, count, -4, x, y, tar, i)
            add_matrix(count, count-1, 1, x, y, tar, i)
            add_matrix(count, count+1, 1, x, y, tar, i)
            add_matrix(count, count-500, 1, x, y, tar, i)
            add_matrix(count, count+500, 1, x, y, tar, i)
            count += 1

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)

    matrix = csc_matrix((data, (row, col)), shape= (count, count))
    answer = linalg.splu(matrix).solve(tar)
    return answer

answer_b = solving(mask_b, i = 0)
answer_g = solving(mask_g, i = 1)
answer_r = solving(mask_r, i = 2)

answer_b = cv2.convertScaleAbs(answer_b)
answer_g = cv2.convertScaleAbs(answer_g)
answer_r = cv2.convertScaleAbs(answer_r)

'''change the color of each pixel'''
final_count = 0
for x in source:
    for i in x:
        i[0] = answer_b[final_count]
        i[1] = answer_g[final_count]
        i[2] = answer_r[final_count]
        final_count += 1

'''show the result'''
cv2.imshow("result", img_source)
cv2.waitKey(0)
