import pandas
import os

def get_frame_id_mapping(folder_path):
    csv_files = os.listdir(folder_path)
    csv_files = [os.path.join(folder_path, csv_file).replace('\\', '/') for csv_file in csv_files]
    frame_id_mapping = []
    for file in csv_files:
        df = pandas.read_csv(file, names = ['frame_name', 'frame_id'])
        df['video'] = file.split('/')[-1].split('.')[0]
        frame_id_mapping.append(df)
    frame_id_mapping = pandas.concat(frame_id_mapping)
    return frame_id_mapping