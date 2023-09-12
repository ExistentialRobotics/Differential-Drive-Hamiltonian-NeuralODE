import os
import shutil


def rename_files_in_directory(directory_path):
    """
    rename point cloud files to have format 00xxxx.npy
    """
    if not os.path.exists(directory_path):
        return
    files = os.listdir(directory_path)
    files.sort(key=lambda x: int(x.split("_")[-1].split(".npy")[0]))
    traj = int(directory_path[-1])
    print(f"currently renaming directory : {traj}")

    for i, file in enumerate(files):
        new_filename = "point_cloud_{:06}.npy".format(i)
        old_filepath = os.path.join(directory_path, file)
        new_filepath = os.path.join(directory_path, new_filename)
        try:
            os.rename(old_filepath, new_filepath)
            print(f"{old_filepath} renamed to {new_filepath}")
        except Exception as e:
            print(e)


if __name__ == '__main__':
    for i in range(len(os.listdir("RealJackalData"))):
        directory_path = "RealJackalData/traj_{:02}".format(i)
        rename_files_in_directory(directory_path)
