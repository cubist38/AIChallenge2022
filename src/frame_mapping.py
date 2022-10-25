from src.utils import get_frame_id_mapping

class FrameMapping:
    def __init__(self, folder_path):
        self.dict = get_frame_id_mapping(folder_path)

    def get_id(self, video, frame_name):
        return self.dict[(self.dict['video'] == video) & (self.dict['frame_name'] == frame_name)].values[0][1]
