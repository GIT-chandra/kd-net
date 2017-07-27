
from pyntcloud.io import read_off
import numpy as np
import sys
import time

print('Done importing!')
start_t = time.time()

LABELS_DICT = {'bathtub':0, 'bed':1, 'chair':2, 'desk':3, 'dresser':4, 'monitor':5, 'night_stand':6, 'sofa':7, 'table':8, 'toilet':9}

def triangle_area(v1,v2,v3):
    return 0.5 * np.linalg.norm(np.cross(v2-v1, v3-v1),axis = 1)

def create_kd_tree(fname):
    print(fname),
    mesh = read_off(fname)

    mesh_points_xyz = mesh["points"][["x","y","z"]].values

    v1s = mesh_points_xyz[mesh["mesh"]["v1"]]
    v2s = mesh_points_xyz[mesh["mesh"]["v2"]]
    v3s = mesh_points_xyz[mesh["mesh"]["v3"]]

    areas = triangle_area(v1s,v2s,v3s)

    probs = areas/areas.sum()
    n = 2**10
    weighted_rand_inds = np.random.choice(range(len(areas)),size = n, p = probs)


    sel_v1s = v1s[weighted_rand_inds]
    sel_v2s = v2s[weighted_rand_inds]
    sel_v3s = v3s[weighted_rand_inds]

    # barycentric co-ords
    u = np.random.rand(n,1)
    v = np.random.rand(n,1)

    invalids = u + v >1 # u+v+w =1

    u[invalids] = 1 - u[invalids]
    v[invalids] = 1 - v[invalids]

    w = 1-(u+v)

    pt_cld = (sel_v1s * u) + (sel_v2s * v) + (sel_v3s * w)

    # find the kd-tree leafs
    kd_orients = []
    nodes = [pt_cld]
    temp_list = []
    while len(nodes[0]) != 1:
        temp_list.clear()
        for node in nodes: 
            ranges = np.amax(node,axis = 0) - np.amin(node,axis = 0)
            split_dir = np.argmax(ranges)
            kd_orients.append(split_dir)
            sorted_node = node[node[:,split_dir].argsort()]
            num_pts = len(sorted_node)
            temp_list.append(sorted_node[0:np.int(num_pts/2),:])
            temp_list.append(sorted_node[np.int(num_pts/2):num_pts,:])
        nodes.clear()
        nodes.extend(temp_list)

    kd_leaves = np.array(nodes)
    kd_orients = np.array(kd_orients)
    # np.save('kd_tree.npy',kd_tree)
    print(str(time.time()-start_t) + 's elapsed.')
    return (kd_leaves,kd_orients)

if len(sys.argv)>1:
    create_kd_tree(sys.argv[1])
    exit()

loc_of_label  = 9 # change this based on depth of directory where ModelNet data is stored

flist_mn10test = [line.rstrip('\n') for line in open('test10.txt')]
MNET10_test = []
MNET10_test_orients = []
MNET10_test_labels = []
for model in flist_mn10test:
    tree = create_kd_tree(model)
    MNET10_test.append(tree[0])
    MNET10_test_orients.append(tree[1])
    label = model.split('/')[loc_of_label] 
    MNET10_test_labels.append(LABELS_DICT[label])


np.save('mnet10_test.npy',np.array(MNET10_test))
np.save('mnet10_test_orients.npy',np.array(MNET10_test_orients))
np.save('mnet10_test_labels.npy',np.array(MNET10_test_labels))

flist_mn10train = [line.rstrip('\n') for line in open('train10.txt')]
MNET10_train = []
MNET10_train_orients = []
MNET10_train_labels = []
for model in flist_mn10train:
    tree = create_kd_tree(model)
    MNET10_train.append(tree[0])
    MNET10_train_orients.append(tree[1])
    label = model.split('/')[loc_of_label] 
    MNET10_train_labels.append(LABELS_DICT[label])

np.save('mnet10_train.npy',MNET10_train)
np.save('mnet10_train_orients.npy',MNET10_train_orients)
np.save('mnet10_train_labels.npy',np.array(MNET10_train_labels))



            