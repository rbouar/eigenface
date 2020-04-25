import cv2
import os
import numpy as np


def load_images(dir):
    image_list = []
    i = 0
    for file in os.listdir(dir):
        image = cv2.imread(dir + file, cv2.IMREAD_GRAYSCALE)
        image_list.append(image)
    print("Nombres d'images: ", len(image_list))
    return image_list

def images_to_vectored_images(image_list):
    vectored_image_list = np.zeros((len(image_list), image_list[0].shape[0] * image_list[0].shape[1]), dtype=np.uint8)
    for i in range(len(image_list)):
        vectored_image_list[i] = image_list[i].flatten()
    return vectored_image_list

def create_output_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

def save_eigenfaces(eigenfaces, size, dir):
    for i in range(len(eigenfaces)):
        ei = eigenfaces[i]
        im = cv2.normalize(ei.reshape(size), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.COLOR_BGR2GRAY)
        cv2.imwrite(dir + "eigenface" + str(i) + ".jpg", im)

def main():
    threshold_is_face = 0.5
    threshold_is_known_face = 0.1
    datadir = "dataset/"
    out_dir = "output/"
    create_output_dir(out_dir)
    image_list = load_images(datadir)
    size = image_list[0].shape

    vectored_image_list = images_to_vectored_images(image_list)

    mean_vector = np.mean(vectored_image_list, axis=0)
    cv2.imwrite(out_dir + "mean.jpg", mean_vector.astype(np.uint8).reshape(size))
    # im_to_test = cv2.imread(, cv2.IMREAD_GRAYSCALE).flatten()
    A = np.subtract(vectored_image_list, mean_vector)
    U, S, V = np.linalg.svd(A.T, full_matrices=False)
    eigenfaces = U.T

    save_eigenfaces(eigenfaces, size, out_dir)
    weights = np.dot(A, eigenfaces.T)
    proj = np.dot(im_to_test - mean_vector, eigenfaces.T)
    proj_face = np.dot(proj, eigenfaces)
    cv2.imwrite(out_dir + "proj.png", cv2.normalize(proj_face.reshape(size), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.COLOR_BGR2GRAY))
    dist = np.min((proj - weights) ** 2, axis=1)

    indiceImg = np.argmin(dist)
    mindist = np.sqrt(dist[indiceImg])
    print(mindist)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__== '__main__':
    main()
