import segmentation_tools as st
from Bio import Phylo
import matplotlib.pyplot as plt
import os



#####
# Import images from disc and threshold the prediction map
#####

# activate plotting in interactive mode
plt.ion()
# generate a labeled_series object
myS = st.labeled_series()

# load ilastik prediction and original images from disc. Original images can be h5. 
# p_cutoff is a float between 0 and 1, denoting the cutoff on the probability landscape
# as produced by ilastik
myS.load_from_file('../sahand_test/pred/Yutao1-17_1-c-???.h5_processed.h5', '../sahand_test/Yutao1-17_1-c-???.h5', p_cutoff=0.7) 




#####
# Quality filter images based on certain object features
#####

# generate the histogram of a region property (area) to get an estimate of segmentation quality
plt.figure()
plt.hist(myS.get_object_props_allTimes('area'), bins=100)
plt.draw()

# based on the histogram, pick an upper and lower cut off for cell areas
lower_th = int(raw_input('Enter lower cut_off: '))
upper_th = int(raw_input('Enter upper cut_off: '))

# simpler version for object filtering
#myS.filter_objects('area', lower_th = lower_th, upper_th = upper_th)

# filter objects with given area cut offs. Multiple filters in conunction are possible. 
myS.filter_objects_multiProp({'area':(lower_th,upper_th),
                              'eccentricity':(0.3,1)},'OR')


# generate a folder for saving quality estimates 
qc_dir = 'QC'
if not os.path.isdir(qc_dir):
    os.mkdir(qc_dir)
# Save quality inspection for each time point to disc
myS.save_QC_series('QC/img_', additional_tps=[])



#####
# Track cells, make lineages and plot the resutling phylogenies
#####

# track cells using point matching based on minimal distance
myS.track_objects(match_func=st.match_mindist)
# track cells using point matching based on minimal distance enforcing match in both directions to be the same
#myS.track_objects(match_func=st.match_bijective)


# generate lineages 
myS.color_lineages()
# Save quality inspection of the point matching to disc 
myS.save_QC_series('QC/track_', additional_tps=[-1])

# find phylogenetic trees in the tracks
myS.find_trees()
# label them according to a property such as area, eccentricty and plot
myS.color_trees('eccentricity')
for tree in myS.trees:
    Phylo.draw(tree[1], label_func = lambda x:'', show_confidence=False)

