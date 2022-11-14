
import os

PATH = '/run/media/zephy_manjaro/Crucial X6/AIC2022/data/scene_frames/frames'


for folder in sorted(os.listdir(PATH)):
    path_folder = os.path.join(PATH, folder)

    list_deleted_files = []

    # if len(os.listdir(path_folder)) % 3 != 0:
    #     print(folder)


    if folder == 'C01_V0100':
        for file in sorted(os.listdir(path_folder)):
            print(file)


    
