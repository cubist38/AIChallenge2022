
import os
import fiftyone as fo


PATH_KEYFRAME = 'data/demo/KeyFramesC00_V00/'

def get_neighbor_frames(video, frameid, delta = 100):

    def search(frameid, frameid_list):
        found_index = -1
        l, r = 0, len(frameid_list) - 1
        while l <= r:
            mid = (l + r)//2
            if frameid_list[mid] >= frameid:
                print(frameid_list[mid])
                found_index = mid
                r = mid - 1
            else:
                l = mid + 1

        return found_index


    def get_neighbor_list(frameid, frameid_list, delta):
        index= search(frameid, frameid_list)
        left, right = max(0, index - delta), min(len(frameid_list), index + delta)
        neighbor_frameid_list = frameid_list[left:right]
        return neighbor_frameid_list


    path_frames_video = os.path.join(PATH_KEYFRAME, video)
    frameid_list = sorted(os.listdir(path_frames_video))

    neighbor_frameid_list = get_neighbor_list(frameid, frameid_list, delta)
    neighbor_frameid_list = [os.path.join(path_frames_video, file) for file in neighbor_frameid_list]


    return neighbor_frameid_list


dataset = fo.Dataset.from_images(
    neighbor_frameid_list
)


session = fo.launch_app(dataset, auto=False)
session.open_tab()






    



