import cv2
import os
import numpy as np
dir = "dataset/"
image_list = []
for file in os.listdir(dir):
    path = dir + file
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    image_list.append(image)

print("Nombres d'images: ", len(image_list))
size = image_list[0].shape

# Transformation des matrices en vecteurs
vectored_image_list = np.zeros((len(image_list), size[0] * size[1]), dtype=np.uint8)
for i in range(len(image_list)):
    vectored_image_list[i] = image_list[i].flatten()
print(vectored_image_list[0].reshape(size))

mean_vector = np.mean(vectored_image_list, axis=0).astype(np.uint8)

# print(mean_vector)
A = np.subtract(vectored_image_list, mean_vector)
# print(type(A[0][0]))
# print(numpy.cov(A).shape)
AT = A.T
print("transpose OK")
COV = np.dot(AT, A)
# COV = np.load('cov.npy')
print('laod:doone')
print("AT * A: DONE")
np.save('cov.npy', COV)
print("save: DONE")
# print(COV.shape)

# print(type(A))
# print(A.shape)
# AT = numpy.transpose(A)
# print(len(A.dot(AT)))
# print(len(AT))
# cov = numpy.matmul(AT, A)
# print(cov.shape)

# Calcul des vecteurs propre avec l'analyse en composante principal
eigen_vectors = np.linalg.eigh(COV)
# np.linalg.svd(COV, full_matrices=True, compute_uv=True, hermitian=True)
print("Eigenvector:DONE")

# Convertion des vecteurs en images de la taille original
eigen_faces = []
# avg_face = avg_face_vector.reshape(size)
# for ev in eigen_vectors:
    # eigen_faces.append(ev.reshape(size))
#
# cv2.imshow('average face', avg_face)
# i = 0
# for ef in eigen_faces:
    # cv2.imshow("eigen" + str(i), ef)
    # i +=1
# cv2.waitKey(0)
# cv2.destroyAllWindows()
