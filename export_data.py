import glob
import pandas as pd


file_names = []

# path to search file
path = 'data/demo/KeyFramesC00_V00/*/*.jpg'
for file in glob.glob(path, recursive=True):
    file_names.append(file)

data = {'id': range(len(file_names)),
        'path': file_names}

df = pd.DataFrame(data)


df.to_csv('data/towhee/reverse_image_search.csv',
          index=False, header=True)


