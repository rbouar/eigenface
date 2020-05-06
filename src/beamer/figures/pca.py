import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv



# NUAGE DE POINTS
n = 40

x0, y0 = 2.5, 2.5

x = np.random.normal(x0, 0.8, n)
y = np.random.normal(y0, 0.4, n)

plt.axis('scaled')
plt.axis([0, 5, 0, 5])

plt.scatter(x, y, marker='.', color='b')
plt.savefig('fig_pca_1.png', bbox_inches='tight')

# COV
points = np.column_stack((x,y))
print("points : ",points.shape)

cov = np.cov(points.T)
print(cov)

# ACP
mean, eigenvectors, eigenvalues = cv.PCACompute2(points, np.mean(points, axis=0).reshape(1,-1))

print(eigenvectors)

print(eigenvalues)

plt.quiver( 2*[mean[0][0]], 2*[mean[0][1]], eigenvectors[:,0], eigenvectors[:,1], angles='xy', scale_units='xy', scale=2.5, color='r')
plt.savefig('fig_pca_4.png', bbox_inches='tight')



# DROITE AFFINE
vec_dir = [10, 1]
a = mean[0][1] - vec_dir[1]/vec_dir[0]*mean[0][0]

plt.clf()
plt.axis('scaled')
plt.axis([0, 5, 0, 5])
plt.scatter(x, y, marker='.', color='b')

plt.plot([0, vec_dir[0]*10+a], [0+a, vec_dir[1]*10+a],'#bbb')
plt.savefig('fig_pca_2.png', bbox_inches='tight')



# PROJECTION SUR CETTTE DROITE
vec_dir_norm = vec_dir / np.linalg.norm(vec_dir)
weight = np.dot(points, vec_dir_norm)

proj = np.dot(weight.reshape(-1,1), vec_dir_norm.reshape(1,2))

vec_dir_orth = [ -vec_dir_norm[1], vec_dir_norm[0] ]

s = np.dot([0, a], vec_dir_orth)

proj[:,0] += s*vec_dir_orth[0]
proj[:,1] += s*vec_dir_orth[1]


plt.scatter(x, y, marker='.', color='b')
for point,point_proj in zip(points, proj):
    plt.plot([point[0],point_proj[0]], [point[1],point_proj[1]], ',:k')
plt.savefig('fig_pca_3.png', bbox_inches='tight')
