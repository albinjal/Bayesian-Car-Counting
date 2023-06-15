import glob
import cv2
import numpy as np

def draw_bboxes(jpg_file, npy_file):
    # get image and annotation
    image = cv2.imread(jpg_file)
    annotations = np.load(npy_file)
    print(annotations.shape)
    print(image.shape)

    # draw bounding boxes
    for annotation in annotations:
        try:
            x, y, _ = annotation
        except:
            x, y = annotation
        x, y = int(x), int(y)  # coordinates should be integers
        cv2.rectangle(image,
                      (x - 8, y - 8),
                      (x + 8, y + 8),
                      (0, 255, 0), 2)

    # show image
    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    dataset = "out"

    # get random jpg and its corresponding npy file from dataset
    jpg_files = glob.glob(dataset + "/*.jpg")

    # define second file
    jpg_file2 = "crowd1.jpg"
    npy_file2 = jpg_file2.replace(".jpg", ".npy")

    # draw bounding boxes
    draw_bboxes(jpg_file2, npy_file2)

    for file in jpg_files:
        npy_file = file.replace(".jpg", ".npy")
        draw_bboxes(file, npy_file)
