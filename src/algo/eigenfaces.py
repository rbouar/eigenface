import cv2
import os
import numpy as np

datadir = "dataset/"
out_dir = "output/"

def load_images(dir):
    image_list = []
    for file in os.listdir(dir):
        image = cv2.imread(dir + file, cv2.COLOR_BGR2GRAY)
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

def save_eigenfaces(eigenfaces, size):
    for i in range(len(eigenfaces)):
        ei = eigenfaces[i]
        im = cv2.normalize(ei.reshape(size), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.COLOR_BGR2GRAY)
        cv2.imwrite(out_dir + "eigenface" + str(i) + ".png", im)

def main():
    print("a")
    create_output_dir(out_dir)
    image_list = load_images(datadir)
    size = image_list[0].shape

    vectored_image_list = images_to_vectored_images(image_list)

    mean_vector = np.mean(vectored_image_list, axis=0).astype(np.uint8)
    cv2.imwrite(out_dir + "mean.png", mean_vector.reshape(size))

    A = np.subtract(vectored_image_list, mean_vector)

    U, S, V = np.linalg.svd(A.T, full_matrices=False)
    eigenfaces = U.T

    save_eigenfaces(eigenfaces, size)

if __name__== '__main__':
    main()
