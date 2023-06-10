import glob
import os
import numpy as np

if __name__ == "__main__":
    in_folder = "cowc_pre"

    # get all npy files in folder and subfolders
    npy_files = glob.glob(in_folder + "/**/*.npy", recursive=True)

    for npy_file in npy_files:
        # load the npy file
        try:
            annotations = np.load(npy_file)
        except Exception as e:
            print(f"Error loading file {npy_file}: {e}")
            # delete the files
            os.remove(npy_file)
            os.remove(npy_file.replace(".npy", ".jpg"))


        # flip the x and y coordinates
        annotations[:, [0, 1]] = annotations[:, [1, 0]]

        # overwrite the npy file
        np.save(npy_file, annotations)

        print("flipped", npy_file)

    print("done")
