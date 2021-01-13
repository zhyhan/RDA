import pickle
import numpy as np
import random
np.set_printoptions(threshold=np.inf)
from sklearn.cluster import KMeans

with open('RDA_V2-refine_webcam-feature_uniform-noisy-0.4-2021-01-05-11-31-57.pkl',"rb") as f:
    prob_matrix = pickle.load(f)
prob_matrix_random = prob_matrix.copy()
# print(prob_matrix.shape)
# #print(np.max(prob_matrix, axis=1))
# random.shuffle(prob_matrix_random)
# km = KMeans(n_clusters=2).fit(prob_matrix_random)
# #km.predict(prob_matrix)
# labels = km.predict(prob_matrix)
# print(len(labels))
# distances = km.transform(prob_matrix)
# preds, new_labels = [], []
# margin = 1.5
# clean_label = np.sum(labels[:10])/10.
# if clean_label < 0.5:
#     tgt_label = 0
# else:
#     tgt_label = 1
# for i, j in enumerate(labels):
#     disc = abs(distances[i][0] - distances[i][1])
#     if j == tgt_label and disc > margin:
#         new_labels.append(0)
#     else:
#         new_labels.append(1)
#     preds.append([j, distances[i]])
# print(np.array(new_labels))

# #TODO: generate new file.
# source_file =  "/home/ubuntu/nas/projects/RDA/data/Office-31/webcam_feature_uniform_noisy_0.4_false_pred.txt"
# save_file = "/home/ubuntu/nas/projects/RDA/data/Office-31/webcam_feature_uniform_noisy_0.4_false_pred_refine.txt"
# with open(source_file, 'r') as f:
#     file_dir, label = [], []
#     for i in f.read().splitlines():
#         file_dir.append(i.split(' ')[0])
#         label.append(int(i.split(' ')[1]))

# with open(save_file,'w') as f:
#     for i, d in enumerate(new_labels):
#         if d == 0:
#             f.write('{} {}\n'.format(file_dir[i], label[i]))

#use energy to obtain clean data.
source_file =  "/home/ubuntu/nas/projects/RDA/data/Office-31/webcam_feature_uniform_noisy_0.4_false_pred.txt"
save_file = "/home/ubuntu/nas/projects/RDA/data/Office-31/webcam_feature_uniform_noisy_0.4_false_pred_refine.txt"
with open(source_file, 'r') as f:
    file_dir, label = [], []
    for i in f.read().splitlines():
        file_dir.append(i.split(' ')[0])
        label.append(int(i.split(' ')[1]))

with open(save_file,'w') as f:
    for i, d in enumerate(prob_matrix):
        if d < np.sort(prob_matrix)[50]:
            f.write('{} {}\n'.format(file_dir[i], label[i]))