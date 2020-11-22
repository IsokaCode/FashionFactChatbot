""" import numpy as np 

#using numpy and matrix's
arr1 = np.array([
                [2,4,5],
                [3,4,5],
                [4,5,3],
                [6,4,2]
])

id_arr1 = np.array([
                [1,0,0],
                [0,1,0],
                [0,0,1]
]) 



matrix1 = np.matrix(arr1)
# matrix2 = np.matrix(id_arr1)

print(arr1.flatten())
print(matrix2)

result_matrix = matrix1 * matrix2

print(result_matrix)

print(matrix1.max(),matrix1.min())

matrix1 < 2

arr3d = np.zeros((4,3,2))

print(arr3d)  """

import numpy as np
import skimage
from skimage import data
import matplotlib.pyplot as plt

camera = data.camera()
camera[:10] = 0
mask = camera < 87
camera[mask] = 255
inds_x = np.arange(len(camera))
inds_y = (4 * inds_x) % len(camera)
camera[inds_x, inds_y] = 0

l_x, l_y = camera.shape[0], camera.shape[1]
X, Y = np.ogrid[:l_x, :l_y]
outer_disk_mask = (X - l_x / 2)**2 + (Y - l_y / 2)**2 > (l_x / 2)**2
camera[outer_disk_mask] = 0

plt.figure(figsize=(4, 4))
plt.imshow(camera, cmap='gray')
plt.axis('off')
plt.show()