import os
from sklearn.model_selection import train_test_split

def load_annotations(list_file_path, data_dir):
    data = []
    label_map = {}
    with open(list_file_path, 'r') as f:
        for line in f:
            if line.startswith('#'): continue
            name, label = line.split()[:2]
            label = int(label) - 1  # zero-based
            label_map.setdefault(label, name.split('_')[0])
            path = os.path.join(data_dir, name + '.jpg')
            data.append((path, label))
    return data, label_map

def split_data(data, test_size=0.2, val_size=0.1):
    train, test = train_test_split(data, test_size=test_size, random_state=42, stratify=[d[1] for d in data])
    train, val = train_test_split(train, test_size=val_size, random_state=42, stratify=[d[1] for d in train])
    return train, val, test
