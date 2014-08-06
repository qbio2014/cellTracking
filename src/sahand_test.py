from segmentation_tools import *
from Bio import Phylo
import matplotlib.pyplot as plt
import os

myS = labeled_series()
myS.load_from_file('../sahand_test/pred/Yutao1-17_1-c-???.h5_processed.h5') 
myS.filter_objects('area', lower_th = 10, upper_th = 1000)
myS.track_objects(match_func=match_mindist)

myS.color_lineages()
qc_dir = '../sahand_test/QC'
if not os.path.isdir(qc_dir):
    os.mkdir(qc_dir)

#myS.save_QC_series('../sahand_test/QC/img_', additional_tps=[-1])

myS.find_trees()
myS.color_trees('area')
for tree in myS.trees:
    Phylo.draw(tree[1], label_func = lambda x:'', show_confidence=False)

