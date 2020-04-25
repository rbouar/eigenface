import cv2
import os
import numpy as np


def load_images(dir):
    image_list = []
    subjects = {}
    dim = None
    for subdir in os.listdir(dir):
        name = subdir
        subdir = dir + subdir + "/"
        if (os.path.isdir(subdir)):
            subject = []
            for file in os.listdir(subdir):
                image = cv2.imread(subdir + "/" + file, cv2.IMREAD_GRAYSCALE)
                if dim == None:
                    dim = image.shape
                flatten_image = image.reshape(dim[0] * dim[1])
                image_list.append(flatten_image)
                subject.append(flatten_image)
            subjects[name] = np.array(subject)
    print("Nombres d'images: ", len(image_list))
    print("Nombre d'individus: ", len(subjects))
    return np.array(image_list), subjects, dim

def create_output_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)
    if not os.path.exists(dir + "eigenfaces/"):
        os.mkdir(dir + "eigenfaces/")

def save_eigenfaces(eigenfaces, dim, dir):
    for i in range(len(eigenfaces)):
        ei = eigenfaces[i]
        im = cv2.normalize(ei.reshape(dim), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.COLOR_BGR2GRAY)
        cv2.imwrite(dir + "eigenfaces/eigenface" + str(i) + ".jpg", im)

def init_face_class(subjects, eigenfaces, mean, dim):
    face_class = {}
    for subject, list in subjects.items():
        proj_list = []
        for face in list:
            proj_list.append(projection(eigenfaces, face, mean, dim))
        face_class[subject] = np.mean(np.array(proj_list), axis=0)
    return face_class
def projection(eigenfaces, vector, mean, dim, file=None):
    mean_adjusted_face = vector - mean
    proj = np.dot(eigenfaces.T, mean_adjusted_face)
    proj_face = (mean + np.dot(eigenfaces, proj))
    if (file != None):
        cv2.imwrite(file, proj_face.reshape(dim))
    return proj_face

def nearest(proj_face, face_class):
    min = nearest = None
    for name, face in face_class.items():
        norm = np.linalg.norm(proj_face - face)
        if min == None or norm < min:
            nearest = name
            min = norm

    return nearest, min
def main():
    threshold_is_face = 0.5
    threshold_is_known_face = 0.1
    datadir = "att_faces/"
    out_dir = "output/"
    create_output_dir(out_dir)
    image_list, subjects, dim = load_images(datadir)
    mean = np.mean(image_list, axis=0)
    cv2.imwrite(out_dir + "mean.jpg", mean.astype(np.uint8).reshape(dim))
    im_to_test = cv2.imread("att_faces/s1/1.pgm", cv2.IMREAD_GRAYSCALE).flatten().astype(np.float64)
    A = np.subtract(image_list, mean)
    U, S, V = np.linalg.svd(A.T, full_matrices=False)
    eigenfaces = U

    save_eigenfaces(eigenfaces.T, dim, out_dir)

    face_class = init_face_class(subjects, eigenfaces, mean, dim)
    proj_face = projection(eigenfaces, im_to_test, mean, dim, out_dir + "proj.jpg")
    facespace_dist = np.linalg.norm(im_to_test - proj_face)

    nearest_class, norm = nearest(proj_face, face_class)
    print(nearest_class)
    print(norm)
    print(facespace_dist)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__== '__main__':
    main()
