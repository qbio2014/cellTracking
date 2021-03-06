import numpy as np
from scipy import ndimage
from skimage.morphology import remove_small_objects
from skimage.measure import regionprops
from skimage.io import imread
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib import cm
from Bio import Phylo

####################################################################################
# utility functions
####################################################################################
# -------------    
# -------------
def xcorr2fft(image1, image2):
    '''
    phase correlation method based shift estimation between images
    translation of image2 with respect to image1

    return variable shift is a vector that has the dim of the two images    
    '''
    assert image1.shape == image2.shape
    dim = np.asarray(image1.shape)
    F     = np.fft.fftn(image1)
    Fc    = np.conj(np.fft.fftn(image2))
    R     = F*Fc
    c     = np.fft.ifftn(R)

    shift_indices = np.asarray(np.unravel_index(c.argmax(), c.shape))
    shift = np.where(np.abs(shift_indices-1)<np.abs(dim-shift_indices+1), -shift_indices, dim-shift_indices)
    return shift

def read_ilastik_pred(fname, channel, p_cutoff = 0.5):
    '''
    read an image from and ilastik h5 file. assumes prediction is in group /volumes/prediction
    channel specifies the object class to be returned
    p_cutoff is the threshold for the posterior probability
    '''
    import h5py
    f = h5py.File(fname,'r')
    return np.asarray(f['volume']['prediction']).squeeze()[:,:,channel]>p_cutoff

def read_ilastik_data(fname):
    '''
    read an image from and ilastik h5 file. assumes prediction is in group /volumes/prediction
    channel specifies the object class to be returned
    p_cutoff is the threshold for the posterior probability
    '''
    import h5py
    f = h5py.File(fname,'r')
    return np.asarray(f['volume']['data']).squeeze()

def match_mindist(X2, X1, dmax):
    '''
    Match points by euclidean distance with upper limit for distance.
    Briefly: Obtain distance matrix, compute where minima in rows
    
    Return is a vector that assigns each element in  a partner in X2 or -1 (no match case)
    '''
    from scipy.spatial.distance import cdist

    # calculate distance matrix
    D = cdist(X1,X2,'euclidean')
    m1, m2  = np.min(D, axis=1), np.min(D, axis=0)
    m1i, m2i = np.argmin(D, axis = 1), np.argmin(D, axis = 0)
    m1i[m1>=dmax] = -1
    m2i[m2>=dmax] = -1
    return m1i, m2i


def match_bijective(X2,X1,dmax):
    '''
    Match points by euclidean distance with upper limit for distance.
    Briefly: Obtain distance matrix, compute where minima in rows and columns
             agree and remove assigned points from the matrix. Iterate until 
             no entries with distance < dmax exist, or until no unique 
             assignments are possible anymore. 
    
    
    X1 are the points in time 1 and X2 in time 2
    dmax is an upper limit to the distance between matched points
    
    Returns a two vectors assigning each element in X1 a partner in X2
    and vice versa. -1 signifies to no match 
    '''

    np1, np2= X1.shape[0], X2.shape[0]
    ind1, ind2 = np.arange(np1), np.arange(np2)
    ix1, ix2 = -np.ones(np1, dtype = 'int32'), -np.ones(np2, dtype = 'int32')
    
    # calculate distance matrix
    while len(ind1) and len(ind2):
        m12, m21 = match_mindist(X2[ind2], X1[ind1], dmax)
        agree1 = (m21[m12] == np.arange(len(ind1)))*(m12>-1)
        agree2 = (m12[m21] == np.arange(len(ind2)))*(m21>-1)
        if agree1.sum()==0 or agree2.sum()==0:
            break

        ix1[ind1[agree1]]= m12[agree1]       
        ix2[ind2[agree2]]= m21[agree2]       
        ind1 = ind1[-agree1]
        ind2 = ind2[-agree2]

    return ix1, ix2


####################################################################################
# classes
####################################################################################

class segmented_image(object):
    '''
    simple class providing utility functions to work with segmented images
    TODO: add functionality to read images of disk
    '''

    ####################################################################################
    # constructor
    ####################################################################################    
    
    def __init__(self, seg_fname, image_fname=None, save_image=False, p_cutoff = 0.5, channel = 1):
        self.p_cutoff = p_cutoff  # ilastik posterior probability cutoff
        self.channel = channel     # ilastik segmentation channel
        
        self.image_fname = image_fname
        self.seg_fname = seg_fname
        if save_image:
            self.seg = self.get_seg()
            if self.image_fname is not None: self.img = self.get_img()
        else: 
            self.img=None
            self.seg = None
        tmp_seg = self.get_seg()
        self.seg_shape = tmp_seg.shape
        print "segmentation dimensions:", self.seg_shape
        if self.image_fname is not None:
            tmp_img = self.get_img()
            self.img_shape = tmp_img.shape
            print "image dimensions:", self.img_shape
        else:
            self.img_shape = None

    ####################################################################################
    # methods
    ####################################################################################

    def get_seg(self):
        '''
        return segmented image. load from file if not saved
        '''
        if self.seg is not None:
            return self.seg
        else:
            # determine file format and specify loading function
            if self.seg_fname.split('.')[-1].startswith('tif'):
                return imread(self.seg_fname)
            elif self.seg_fname.split('.')[-1].startswith('h5'):
                return read_ilastik_pred(self.seg_fname, self.channel, self.p_cutoff)
            else:
                print 'unsupported image format'            
                return None
            
    def get_img(self):
        '''
        return intensity image. load from file if not saved.
        '''
        if self.seg is not None:
            return self.seg
        elif self.image_fname is not None:
            # determine file format and specify loading function
            if self.image_fname.split('.')[-1].startswith('tif'):
                return imread(self.image_fname)
            elif self.image_fname.split('.')[-1].startswith('h5'):
                return read_ilastik_data(self.image_fname)
            else:
                print 'unsupported image format'            
                return None            
        else:
            return None



class labeled_image(segmented_image):   
    '''
    child of segmented_image
    simple class providing utility functions to work with labeled objects
    '''
    def __init__(self, seg_fname, img_fname, min_size = None, save_image=False, channel = 1, p_cutoff = 0.5):
        segmented_image.__init__(self,seg_fname, img_fname,save_image, channel=channel,p_cutoff=p_cutoff)
        self.parents = defaultdict(list)
        self.children = defaultdict(list)
        self.min_size = min_size
        labeled_img = self.get_labeled_img()

        # make a dictionary that links the label th the slice of the image
        # in which the object is found.  
        self.labeled_obj = {label_i+1:o_slice for label_i, o_slice in 
                            enumerate(ndimage.find_objects(labeled_img))}

        self.region_props = {label_i+1:o_slice for label_i, o_slice in 
                            enumerate(regionprops(labeled_img, intensity_image = self.get_img()))}
        self.obj_list = self.labeled_obj.keys()
        self.obj_list.sort()
    
    def filter_objects(self, prop, lower_th, upper_th):
        '''
        filter the list of objects by a given property using a lower and upper threshold.
        updates self.obj_list with objects passing filter only
        '''
        self.obj_list= [label_i for label_i, obj in self.region_props.iteritems()
                            if obj[prop]>=lower_th and obj[prop]<upper_th]
        self.obj_list.sort()
        # reset the parents and children since assignments have been invalidated after
        # redefining the object set.
        self.parents = defaultdict(list)
        self.children = defaultdict(list)
        return self.obj_list

    def get_labeled_img(self):
        if self.min_size is not None:
            labeled_img, self.n_objects = ndimage.label(remove_small_objects(self.get_seg(), self.min_size))
        else:
            labeled_img, self.n_objects = ndimage.label(self.get_seg())
        return labeled_img

        
    def filter_objects_multiProp(self, criteria, gate = 'AND'):
        '''
        filter the list of objects by a given property using a lower and upper threshold.
        updates self.obj_list with objects passing filter only
        criteria: Dictionary with string, and value tuples. 
        '''
        passing_filter = {} # instantiate dictionary       
        for prop, (low,up) in criteria.iteritems(): # iteritems returns key, and tuple of values
            passing_filter[prop]= set([label_i for label_i, obj in self.region_props.iteritems()
                                    if obj[prop]>=low and obj[prop]<up])
                                
        if gate == 'AND':
            # criteria have to be met in all properties
            self.obj_list = set(self.labeled_obj.keys())
            for prop, val in passing_filter.iteritems():
                self.obj_list.intersection_update(val)
        else:
            # criteria have to be met in one property
            self.obj_list = set()
            for prop, val in passing_filter.iteritems():
                self.obj_list.update(val)
                    
        
        self.obj_list = list(self.obj_list)                                            
        self.obj_list.sort()
        # reset the parents and children since assignments have been invalidated after
        # redefining the object set.
        self.parents = defaultdict(list)
        self.children = defaultdict(list)
        return self.obj_list


#################################################################################

class labeled_series(object):
    '''
    class holding a series of images of objects that can be tracked through the 
    series
    '''
    def __init__(self):
        self.colorlookup = {}  # dictionary of dictionaries assigning colors to obj in time slices
        self.series = []       # list holding the labeled images of the experiment
        self.shifts = None     # two dimensional shifts of an image relative to its predecessor
        self.trees = []


    def load_from_file(self, file_mask_seg, file_mask_intensity=None, min_size=None, channel = 1, p_cutoff = 0.9):
        '''
        loads an image series from file given a search string for segmented images
        optionally takes a corresponding search string for the intensity images
        file list need to be sortable
        '''
        from glob import glob
        # make list of segmentation files to be loaded, sort them
        self.segmentation_files = glob(file_mask_seg)
        self.segmentation_files.sort()

        # if intensity image names are provided, load and sort them too. truncate list if too many images found
        if file_mask_intensity is not None:
            self.image_files = glob(file_mask_intensity)
            self.image_files.sort()
            if len(self.image_files)>len(self.segmentation_files):
                self.image_files=self.image_files[:len(self.segmentation_files)]
        
        # load image and append to self.series
        if len(self.segmentation_files)>0:
            if file_mask_intensity is not None:
                for seg_name, img_name in zip(self.segmentation_files, self.image_files):
                    print "reading", seg_name, img_name
                    self.series.append(labeled_image(seg_name, img_name, min_size, channel = channel, p_cutoff=p_cutoff))
            else:
                for seg_name in self.segmentation_files:
                    print "reading", seg_name
                    self.series.append(labeled_image(seg_name, None, min_size, channel = channel, p_cutoff=p_cutoff))
        
            self.dim = self.series[-1].seg_shape
            [self.color_randomly(ti) for ti in range(len(self.series))]
        else:
            print "no images found at", file_mask_seg

    def get_object_props(self, prop, ti):
        '''
        Per time point in the series, extract the properties of interest as list. 
        prop: String specifiging property name according to reginoprops
        ti  : integer. time point
        
        '''
        return [p[prop] for p in self.series[ti].region_props.values()]
        
    def get_object_props_allTimes(self, prop):
        '''
        For all time points in the series, extract the properties of interest as list. 
        prop: String specifiging property name according to reginoprops
        
        '''
        return_list = []
        for t in range(len(self.series)):
            return_list.extend(self.get_object_props(prop,t))
    
        return return_list

    def filter_objects(self, prop, lower_th, upper_th):
        '''
        loops over all time points and filters the objects at this time point according to 
        a given criterion of regionprops with upper and lower threshold
        TODO add AND and OR operations for multiple conditions
        '''
        for ti,labeled_img in enumerate(self.series):
            labeled_img.filter_objects(prop, lower_th, upper_th)

    def filter_objects_multiProp(self, criteria, gate = 'AND'):
        '''
        loops over all time points and filters the objects at this time point according to 
        a given criterion of regionprops with upper and lower threshold
        TODO add AND and OR operations for multiple conditions
        '''
        for ti,labeled_img in enumerate(self.series):
            labeled_img.filter_objects_multiProp(criteria, gate)

    def calc_image_shifts(self, channel=None):
        '''
        loops over the series of images and calculates the most likely shift
        by which image ti differs from image ti+1
        '''
        self.shifts = np.zeros((len(self.series)-1, len(self.dim)))
        for ti in xrange(len(self.series)-1):
            try:
                if channel is None:
                    if len(self.series[ti].img_shape)==2:
                        img1, img2 = self.series[ti].get_img(), self.series[ti+1].get_img()
                    else:
                        img1, img2 = self.series[ti].get_img().max(axis=-1), self.series[ti+1].get_img().max(axis=-1)
                else:
                    img1, img2 = self.series[ti].get_img()[:,:,channel], self.series[ti+1].get_img()[:,:,channel]
            except:
                img1, img2 = self.series[ti].get_seg()>0, self.series[ti+1].get_seg()>0
            self.shifts[ti] = xcorr2fft(img1, img2)
            print 'Shift', ti, 'to', ti+1, self.shifts[ti]


    def track_objects(self, match_func = match_bijective, dmax = 1000):
        '''
        loops over the image series and applices the match_func to each pair of images
        '''
        if self.shifts is None:
            self.calc_image_shifts()

        for ti in xrange(len(self.series)-1):
            obj1, obj2 = self.series[ti].obj_list, self.series[ti+1].obj_list
            if len(obj1) and len(obj2):
                points1 = np.array([self.series[ti].region_props[obj]['centroid'] for obj in obj1])
                points2 = np.array([self.series[ti+1].region_props[obj]['centroid']-self.shifts[ti] for obj in obj2])
                m12, m21 = match_func(points2, points1, dmax)
                for parent, child in zip(m21, obj2):
                    if parent!=-1:
                        self.series[ti].children[obj1[parent]].append(child)
                        self.series[ti+1].parents[child].append(obj1[parent])
            else:
                print "no objects in image at time",ti,"after filtering"

            print "matched", (m12>-1).sum(), 'objects.', (m12==-1).sum() , (m21==-1).sum(), 'left unmatched in time step', ti, ti+1, 'respectively'

    ####################################################################
    ### Phylogeny
    ####################################################################

    def find_trees(self):
        '''
        loops overall time points and finds objects without parents
        for each, generate a new tree
        '''
        self.trees = []
        for ti,tp in enumerate(self.series):
            for oi in tp.obj_list:
                if oi not in tp.parents: # oi does not have a parent
                    print "new tree found at time",ti, "with object id",oi, "as root"
                    self.trees.append((ti, self.build_tree(ti,oi)))

    def build_tree(self,ti,oi):
        '''
        given a root, construct a BioPython tree and call a function that recursively adds 
        subtrees for all children of the root. The tree object is returned
        '''
        new_tree = Phylo.BaseTree.Tree()
        new_tree.root.name = str((ti,oi))
        self.add_subtree(new_tree.root, ti, oi)
        return new_tree

    def add_subtree(self, clade, ti, oi):
        '''
        recursively add children to the tree.
        '''
        node_children = self.series[ti].children[oi]
        clade.split(len(node_children))
        for ci,child in enumerate(node_children):
            clade.clades[ci].name = str((ti+1,child))
            self.add_subtree(clade.clades[ci], ti+1, child)
    
    def color_trees(self, prop):
        for _, tree in self.trees:
            props = {}
            for node in tree.get_terminals()+tree.get_nonterminals():
                ti, oi = map(int, node.name[1:-1].split(','))
                props[node.name] = self.series[ti].region_props[oi][prop]
            max_prop = max(props.values())
            for node in tree.get_terminals()+tree.get_nonterminals():
                node.color = [int(x) for x in cm.jet(props[node.name]/max_prop, bytes=True)[:-1]]

    ####################################################################
    ### coloring
    ####################################################################
    def color_randomly(self,ti):
        '''
        resets the color lookup of time point ti to random choices of the jet colormap
        '''
        self.colorlookup[ti] = {oi:cm.jet(np.random.randint(256)) for oi in self.series[ti].obj_list}

    def color_as_parent(self, ti):
        '''
        redo the color assignment by assigning each object the same color 
        as that of its parent
        ti -- time slice to operate on
        '''
        assert ti>0
        cur = self.series[ti]
        parent_colors = self.colorlookup[ti-1]
        temp_colors = {}
        for oi in cur.obj_list:
            if oi in cur.parents: # if node has parent, assign parent color
                temp_colors[oi] = parent_colors[cur.parents[oi][0]]
            else: # otherwise assign random color
                temp_colors[oi] = cm.jet(np.random.randint(256))
        self.colorlookup[ti]=temp_colors

    def color_lineages(self, initial_frame = 0):
        '''
        assign random colors to the initial frame, color all subsequent frames sa parent
        '''
        self.color_randomly(initial_frame)
        for ti in xrange(initial_frame+1, len(self.series)):
            self.color_as_parent(ti)


    ####################################################################
    ### plotting
    ####################################################################
    
    def add_centroids(self, ti):
        '''
        adds a colored dot (according to the colorlookup) at the centroid of each
        object passing filter criteria of time point ti.
        uses the current axis
        '''
        for label_i in self.series[ti].obj_list:
            x,y = self.series[ti].region_props[label_i]['centroid']
            plt.plot([y],[x], 'o', c=self.colorlookup[ti][label_i])
        plt.ylim(0,self.series[ti].seg_shape[0])
        plt.xlim(0,self.series[ti].seg_shape[1])

    def add_forward_trajectory(self, ti, oi):
        '''
        starting with object oi at time points oi, looks for one
        descendant in subsequent time slices and constructs a trajectory 
        of the centroids. adds this trajectory to the current axis. 
        NOTE: this always uses child[0]
        '''
        traj = [self.series[ti].region_props[oi]['centroid']]
        next_oi = oi
        next_ti = ti
        while next_oi in self.series[next_ti].children:
            next_oi = self.series[next_ti].children[next_oi][0]
            next_ti += 1
            traj.append(self.series[next_ti].region_props[next_oi]['centroid'])
        traj = np.array(traj)
        plt.plot(traj[:,1], traj[:,0], ls='-', marker = 'o', c=self.colorlookup[ti][oi])     

    def add_backward_trajectory(self, ti, oi):
        '''
        starting with object oi at time points oi, looks for the
        ancestor in previous time slices and constructs a trajectory 
        of the centroids. adds this trajectory to the current axis. 
        NOTE: this always uses child[0]
        '''
        traj = [self.series[ti].region_props[oi]['centroid']]
        next_oi = oi
        next_ti = ti
        while next_oi in self.series[next_ti].parents:
            next_oi = self.series[next_ti].parents[next_oi][0]
            next_ti -= 1
            traj.append(self.series[next_ti].region_props[next_oi]['centroid'])
        traj = np.array(traj)
        plt.plot(traj[:,1], traj[:,0], ls='-', marker = 'o', c=self.colorlookup[ti][oi])        

    def plot_image_and_centroids(self, ti, ax=None):
        '''
        plots the intensity image (if not available the labeled image) and adds
        dots at the centroids of each object passing filters. Constructs a new figure
        if no axis is given
        '''
        if ax is None: # make new figure if necessary
            fig = plt.figure()
            ax = plt.subplot(111)
        try:  # plot intensity image if available
            ax.imshow(self.series[ti].get_img(), interpolation='nearest', cmap = cm.gray)
        except: # fallback to labeled image
            ax.imshow(self.series[ti].get_seg(), interpolation='nearest', cmap = cm.gray)
        self.add_centroids(ti)        
        
    def plot_image_and_trajectories(self, ti, ax=None, bwd = False, fwd = True):
        '''
        plots the intensity image (if not available the labeled image) and adds
        dots at the trajectory of centroids of each object passing filters. Constructs a new figure
        if no axis is given. backward and forward trajectories can be specified.
        '''
        if ax is None:
            fig = plt.figure()
            ax = plt.subplot(111)
        try:
            ax.imshow(self.series[ti].get_img(), interpolation='nearest',cmap = cm.gray)
        except:
            ax.imshow(self.series[ti].get_seg(), interpolation='nearest',cmap = cm.gray)
        for oi in self.series[ti].obj_list:
            if fwd: self.add_forward_trajectory(ti,oi)        
            if bwd: self.add_backward_trajectory(ti,oi)        
        plt.ylim(0,self.series[ti].seg_shape)
        plt.xlim(0,self.series[ti].seg_shape)

    def save_QC_series(self,save_path, additional_tps = [], img_format='png'):
        '''
        saves each time frame to file and adds the centroids of subsequent time slices.
        '''
        for ti in xrange(len(self.series)):
            plt.ioff()
            self.plot_image_and_centroids(ti)
            if len(additional_tps):
                for dt in additional_tps:
                    if ti>=-dt:
                        self.add_centroids(ti+dt)
            plt.savefig(save_path+format(ti,'03d')+'.'+img_format)
            plt.close()



if __name__ == '__main__':

    # test matching
    points1 = np.random.uniform(size = (5,2))
    points2 = np.concatenate((points1+ 0.02* np.random.uniform(size = (5,2)),
                              np.random.uniform(size = (4,2))))[::-1]
    m12, m21 = match_mindist(points2, points1, 0.1)
    print 'minimal distance', m12, m21

    m12, m21 = match_bijective(points2, points1, 0.1)
    print 'minimal bijective', m12, m21

    # test shift detetion
    Y,X = np.meshgrid(np.arange(100), np.arange(100))
    center = (50,50)
    shift = (3,-7)
    test_image =  np.exp(-(X-center[0])**2 - (Y-center[1])**2) + 0.01*np.random.randn(100,100)
    test_image2 = np.exp(-(X-center[0]-shift[0])**2 - (Y-center[1]-shift[1])**2)+ 0.01*np.random.randn(100,100)
    print 'True:', shift, 'Inferred:', xcorr2fft(test_image, test_image2)
