from segmentation_tools import *
from Bio import Phylo
import matplotlib.pyplot as plt
import os
plt.ion()

myS = labeled_series()
myS.load_from_file('../sahand_test/pred/Yutao1-17_1-c-???.h5_processed.h5', '../sahand_test/Yutao1-17_1-c-???.h5', p_cutoff=0.7) 
plt.figure()
plt.hist(myS.get_object_props_allTimes('area'), bins=100)
plt.draw()

lower_th = int(raw_input('Enter lower cut_off: '))
upper_th = int(raw_input('Enter upper cut_off: '))

#myS.filter_objects('area', lower_th = lower_th, upper_th = upper_th)

myS.filter_objects_multiProp({'area':(lower_th,upper_th)},'OR')
myS.track_objects(match_func=match_mindist)

myS.color_lineages()
qc_dir = '../sahand_test/QC'
if not os.path.isdir(qc_dir):
    os.mkdir(qc_dir)

#myS.save_QC_series('../sahand_test/QC/img_', additional_tps=[-1])

myS.find_trees()
myS.color_trees('eccentricity')
for tree in myS.trees:
    Phylo.draw(tree[1], label_func = lambda x:'', show_confidence=False)

