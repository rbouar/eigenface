import cv2
import os
import numpy as np
import sys

threshold_is_face = 0.5
threshold_is_known_face = 0.1
datadir = "yale_dataset/"
out_dir = "output/"

def input_image():
    if len(sys.argv) < 2:
        print("Un arguments nécessaire")
        return None
    else:
        im = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
        if im is None:
            print(sys.argv[1] + ": image inconnue")
            return None
        return im.flatten()
def load_images():
    image_list = []
    subjects = {}
    dim = None
    for subdir in os.listdir(datadir):
        name = subdir
        subdir = datadir + subdir + "/"
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

def create_output_dir():
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    if not os.path.exists(out_dir + "eigenfaces/"):
        os.mkdir(out_dir + "eigenfaces/")
    if not os.path.exists(out_dir + "faceclass/"):
        os.mkdir(out_dir + "faceclass/")

def save_eigenfaces(eigenfaces, dim):
    for i in range(len(eigenfaces)):
        ei = eigenfaces[i]
        im = cv2.normalize(ei.reshape(dim), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.COLOR_BGR2GRAY)
        cv2.imwrite(out_dir + "eigenfaces/eigenface" + str(i+1) + ".jpg", im)


def projection(eigenfaces, vector, mean, dim, file=None):
    mean_adjusted_face = vector - mean
    proj = np.dot(eigenfaces.T, mean_adjusted_face)
    proj_face = (mean + np.dot(eigenfaces, proj))
    if (file != None):
        cv2.imwrite(file, proj_face.reshape(dim))
    return proj_face



def init_face_class(subjects, eigenfaces, mean, dim):
    face_class = {}
    for subject, list in subjects.items():
        proj_list = []
        for face in list:
            proj_list.append(projection(eigenfaces, face, mean, dim))
        face_class[subject] = np.mean(np.array(proj_list), axis=0)
        cv2.imwrite(out_dir + "faceclass/" + subject + ".jpg", face_class[subject].reshape(dim))
    return face_class


def nearest(proj_face, face_class):
    min = nearest = None
    for name, face in face_class.items():
        norm = np.linalg.norm(proj_face - face)
        if min == None or norm < min:
            nearest = name
            min = norm

    return nearest, min

def result(proj_face, face_class, facespace_dist):
    nearest_class, norm = nearest(proj_face, face_class)
    print("Distance par rapport à l'espace des images: " + str(facespace_dist))
    print("Personne la plus proche: " + str(nearest_class) + " avec une distance de " + str(norm))


def main():
    create_output_dir()
    image_list, subjects, dim = load_images()
    im_to_test = input_image()
    if im_to_test is None:
        return
    mean = np.mean(image_list, axis=0)
    cv2.imwrite(out_dir + "mean.jpg", mean.astype(np.uint8).reshape(dim))
    A = np.subtract(image_list, mean)
    U, S, V = np.linalg.svd(A.T, full_matrices=False)
    eigenfaces = U

    save_eigenfaces(eigenfaces.T, dim)

    face_class = init_face_class(subjects, eigenfaces, mean, dim)
    proj_face = projection(eigenfaces, im_to_test, mean, dim, out_dir + "input_proj.jpg")
    facespace_dist = np.linalg.norm(im_to_test - proj_face)
    result(proj_face, face_class, facespace_dist)




if __name__== '__main__':
    main()
