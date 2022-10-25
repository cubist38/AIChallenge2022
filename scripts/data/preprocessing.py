import os
import glob
import json

thresh_hold = 0.5
DATA_PATH = './data' #Your AIC22_Objects dataset path 
DES_PATH = './unique_objects'

def refine_data(path = DATA_PATH, thr = thresh_hold):
    fold = glob.glob(path + '/*')
    for fold_path in fold:
        file = glob.glob(fold_path + '/*.json')
        for file_path in file:
            with open(file_path, 'r') as f:
                data = json.load(f)
            values = data['detection_scores']
            idxs = []
            for idx, value in enumerate(values):
                if float(value) > thr:
                    idxs.append(idx)
            
            ls = {}

            for key, value in data.items():
                if key not in ls:
                    ls[key] = []
                for i in idxs:
                    ls[key].append(value[i])
                
            with open(file_path, 'w') as f:
                json.dump(ls, f)



def unique_objects(path = DATA_PATH,  des_path = DES_PATH):
    if not os.path.exists(des_path):
        os.mkdir(des_path)

    fold = glob.glob(path + '/*')
    for fold_path in fold:
        fold_name = fold_path.replace('\\', '/').split('/')[-1]
        file = glob.glob(fold_path + '/*.json')
        for file_path in file:
            with open(file_path, 'r') as f:
                data = json.load(f)

            distinct_objects = []
            distinct_labels = []
            
            for i in range(len(data['detection_class_names'])):
                if data['detection_class_entities'][i] not in distinct_objects:
                    distinct_objects.append(data['detection_class_entities'][i])
                    distinct_labels.append(data['detection_class_labels'][i])

            ls = {'unique_objects': distinct_objects, 'class_labels': distinct_labels}
            

            des_fold_path = os.path.join(des_path, fold_name)
            if not os.path.exists(des_fold_path):
                os.mkdir(des_fold_path)

            file_name = file_path.replace('\\', '/').split('/')[-1]

            with open(os.path.join(des_fold_path, file_name), 'w') as f:
                json.dump(ls, f)


if __name__ == '__main__':
    #refine_data(DATA_PATH, thresh_hold)
    unique_objects(DATA_PATH, DES_PATH)