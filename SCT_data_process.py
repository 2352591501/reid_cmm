import os
import shutil
import random
import glob
source_path = 'D:\\360MoveData\\Users\\TmT\\Desktop\\reid-data\\SYSU-MM01'
target = 'D:\\360MoveData\\Users\\TmT\\Desktop\\reid-data\\SYSU_MM01_valid'

Infrad_ID = []
RGB_ID = []

train_ids = open(os.path.join(source_path, 'exp', 'train_id.txt')).readline()
val_ids = open(os.path.join(source_path, 'exp', 'val_id.txt')).readline()
train_ids = train_ids.strip('\n').split(',')
val_ids = val_ids.strip('\n').split(',')

valida_ids = random.sample(val_ids, 10)

val_ids = list(set(val_ids) - set(valida_ids))

selected_ids = train_ids + val_ids
selected_ids_int = []
for selected_id in selected_ids:
    selected_ids_int.append(int(selected_id))

test_ids = open(os.path.join(source_path, 'exp', 'test_id.txt')).readline()
selected_ids_test = test_ids.strip('\n').split(',')
selected_ids_test = [int(i) for i in selected_ids_test]

Infrad_ID_train = set(Infrad_ID) & set(selected_ids_int)

RGB_ID_train = set(RGB_ID) & set(selected_ids_int)

Infrad_ID = random.sample(selected_ids_int, len(selected_ids_int) // 2)
RGB_ID = list(set(selected_ids_int) - set(Infrad_ID))
valida_ids = [int(i) for i in valida_ids]

Infrad_paths = []
RGB_paths = []
validation_paths = []

Infrad_paths_reverse = []
RGB_paths_reverse = []

query_paths = []
gallery_paths = []

for root, dirs, _ in os.walk(source_path):
    for dir in dirs:
        if dir == 'cam3' or dir == 'cam6':
            for _ , Id_dirs, _ in os.walk(os.path.join(root, dir)):
                for id_dir in Id_dirs:
                    if int(id_dir) in Infrad_ID:
                        img_paths = glob.glob(os.path.join(root, dir, id_dir, '*.jpg'))
                        Infrad_paths.extend(img_paths)
                    if int(id_dir) in valida_ids:
                        img_paths = glob.glob(os.path.join(root, dir, id_dir, '*.jpg'))
                        validation_paths.extend(img_paths)

                    if int(id_dir) in RGB_ID:
                        img_paths = glob.glob(os.path.join(root, dir, id_dir, '*.jpg'))
                        RGB_paths_reverse.extend(img_paths)
        elif dir == 'cam1' or dir == 'cam2' or dir == 'cam4' or dir == 'cam5':
            for _ , rgb_Id_dirs, _ in os.walk(os.path.join(root, dir)):
                for id_dir in rgb_Id_dirs:
                    if int(id_dir) in RGB_ID:
                        img_paths = glob.glob(os.path.join(root, dir, id_dir, '*.jpg'))
                        RGB_paths.extend(img_paths)
                    if int(id_dir) in valida_ids:
                        img_paths = glob.glob(os.path.join(root, dir, id_dir, '*.jpg'))
                        validation_paths.extend(img_paths)
                    if int(id_dir) in Infrad_ID:
                        img_paths = glob.glob(os.path.join(root, dir, id_dir, '*.jpg'))
                        Infrad_paths_reverse.extend(img_paths)

print(len(Infrad_paths), len(RGB_paths))


for ir_path in Infrad_paths:
    seq = ir_path.split('\\')[-1]
    id = ir_path.split('\\')[-2]
    cid = 'c' + ir_path.split('\\')[-3][-1]
    shutil.copy(ir_path, os.path.join(target, 'Infrad', id+'_'+cid+'_'+seq))

for rgb_path in RGB_paths:
    seq = rgb_path.split('\\')[-1]
    id = rgb_path.split('\\')[-2]
    cid = 'c' + rgb_path.split('\\')[-3][-1]
    shutil.copy(rgb_path, os.path.join(target, 'RGB', id+'_'+cid+'_'+seq))

for path in validation_paths:
    seq = path.split('\\')[-1]
    id = path.split('\\')[-2]
    cid = 'c' + path.split('\\')[-3][-1]
    shutil.copy(path, os.path.join(target, 'validation', id + '_' + cid + '_' + seq))

'''
for rgb_path in Infrad_paths_reverse:
    seq = rgb_path.split('\\')[-1]
    id = rgb_path.split('\\')[-2]
    cid = 'c' + rgb_path.split('\\')[-3][-1]
    shutil.copy(rgb_path, os.path.join(target, 'reverse', 'Infrad', id + '_' + cid + '_' + seq))


for rgb_path in RGB_paths_reverse:
    seq = rgb_path.split('\\')[-1]
    id = rgb_path.split('\\')[-2]
    cid = 'c' + rgb_path.split('\\')[-3][-1]
    shutil.copy(rgb_path, os.path.join(target, 'reverse', 'RGB', id + '_' + cid + '_' + seq))

'''


