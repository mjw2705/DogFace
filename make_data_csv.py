import csv
import os
import itertools
import random

DataPath = './crop_img/'

txt = open('img4_path_refer.txt', 'r')
paths = {}
while True:
    path = txt.readline()
    if not path:
        break
    path = path.strip().split('G:/crop_img/')[-1].split('/')
    idd, name = path[0], path[-1]
    paths[idd] = name
txt.close()

f = open('./train_data.csv', 'w', encoding='utf-8', newline='')
wr = csv.writer(f)

id_arr = []
for ids in os.listdir(DataPath):
    path = os.path.join(DataPath, ids)
    if len(os.listdir(path)) >= 4:
        id_arr.append(ids)


for ids in os.listdir(DataPath):
    img_id = os.path.join(DataPath, ids)
    images = os.listdir(img_id)
    img_path = []

    if len(images) >= 4:
        genuine_pairs = list(itertools.permutations(images, 2))
        num = len(genuine_pairs)
        pair_num = 1049 if num > 1049 else num

        for genuine in genuine_pairs:
            f, s = genuine
            first = os.path.join(ids, f)
            second = os.path.join(ids, s)
            first = first.replace('\\', '/')
            second = second.replace('\\', '/')
            wr.writerow([first, second, 0])

        # 본인을 뺀 나머지 dir에서 랜덤으로 imposter 추출
        pop_path = paths.pop(ids)
        join_pop_path = os.path.join(ids, pop_path).replace('\\', '/')

        imposter = random.sample(list(paths.values()), pair_num)
        for imp in imposter:
            id = imp.split('.')[0]
            imp = id + '/' + imp
            wr.writerow([join_pop_path, imp, 1])
        paths[ids] = pop_path

        # genuine 개수 1050개 넘어가는 id는 다른 id에서 refer이미지 제외하고 랜덤 추출
        if num > 1049:
            idx = id_arr.index(ids)
            pop_idx = id_arr.pop(idx)
            diff_num = num - pair_num
            rest_ids = random.sample(id_arr, diff_num)
            for rest_id in rest_ids:
                rest = os.path.join(DataPath, rest_id)
                image = os.listdir(rest)
                image.remove(paths[rest_id])
                rand_img = random.choice(image)
                rand_img = rest_id + '/' + rand_img
                wr.writerow([join_pop_path, rand_img, 1])
    else:
        continue


f.close()