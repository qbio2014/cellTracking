import numpy as np
from scipy import ndimage
from skimage.morphology import remove_small_objects
from skimage.measure import regionprops
from skimage.io import imread
from collections import defaultdict
import glob

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
    '''
    def __init__(self, seg_img, img, copy_image=False):
        if copy_image:
            self.seg_img = seg_img.copy()
            if img is not None: self.img = img.copy()
            else: self.img=None
        else:
            self.seg_img = seg_img
            self.img = img    


class labeled_image(segmented_image):
    '''
    simple class providing utility functions to work with labeled objects
    '''
    def __init__(self, seg_img, img, min_size = None, copy_image=False):
        segmented_image.__init__(self,seg_img, img,copy_image)
        self.parents = defaultdict(list)
        self.children = defaultdict(list)

        if min_size is not None:
            self.labeled_img, self.n_objects = ndimage.label(remove_small_objects(self.seq_img, min_size))
        else:
            self.labeled_img, self.n_objects = ndimage.label(self.seg_img)

        # make a dictionary that links the label th the slice of the image
        # in which the object is found.  
        self.labeled_obj = {label_i+1:o_slice for label_i, o_slice in 
                            enumerate(ndimage.find_objects(self.labeled_img))}

        self.region_props = {label_i+1:o_slice for label_i, o_slice in 
                            enumerate(regionprops(self.labeled_img, intensity_image = self.img))}
        self.obj_list = self.labeled_obj.keys()
        self.obj_list.sort()
    
    def filter_objects(self, prop, lower_th, upper_th):
        '''
        filter the list of objects by a given property using a lower and upper threshold.
        updates self.obj_list with objects passing filter only
        '''
        self.obj_list= [obj for obj in self.region_props
                            if obj[prop]>=lower_th and obj[prop]<upper_th]
        self.obj_list.sort()
        return self.obj_list




class labeled_series(object):
    '''
    class holding a series of images of objects that can be tracked through the 
    series
    '''
    def __init__(self):
        pass


    def load_from_file(self, file_mask_seg, file_mask_intensity=None, min_size=None):
        '''
        loads an image series from file given a search string for segmented images
        optionally takes a corresponding search string for the intensity images
        file list need to be sortable
        '''
        self.segmentation_files = glob.glob(file_mask_seg)
        self.segmentation_files.sort()
        if file_mask_intensity is not None:
            self.image_files = glob.glob(file_mask_intensity)
            self.image_files.sort()
            if len(self.image_files)>len(self.segmentation_files):
                self.image_files=self.image_files[:len(self.segmentation_files)]
        
        if len(self.segmentation_files)>0:
            self.series = []
            if file_mask_intensity is not None:
                for seg_name, img_name in zip(self.segmentation_files, self.image_files):
                    print "reading", seg_name, img_name
                    self.series.append(labeled_image(imread(seg_name), imread(img_name), 
                                                min_size))
            else:
                for seg_name in self.segmentation_files:
                    print "reading", seg_name
                    self.series.append(labeled_image(imread(seg_name), None, min_size))
        
            self.dim = self.series[-1].seg_img.shape
        else:
            print "no images found at", file_mask_seg

    def calc_image_shifts(self):
        '''
        loops over the series of images and calculates the most likely shift
        by which image ii differs from image ii+1
        '''
        self.shifts = np.zeros((len(self.series)-1, len(self.dim)))
        for ii in xrange(len(self.series)-1):
            try:
                self.shifts[ii] = xcorr2fft(self.series[ii].img, self.series[ii+1].img)
            except:
                self.shifts[ii] = xcorr2fft(self.series[ii].seg_img>0, self.series[ii+1].seg_img>0)
            print 'Shift', ii, 'to', ii+1, self.shifts[ii]

    def _init_tree(self, initial_obj):
        from Bio import Phylo
        self.tree = Phylo.BaseTree.Tree()
        self.tree.root.split(len(initial_obj))
        for (oi,O),child in zip(initial_obj.iteritems(), self.tree.root.clades):
            child.name = str(oi) 

    def track_objects(self, match_func = match_bijective, dmax = 100):
        '''
        loops over the image series and applices the match_func to each pair of images
        '''
        for ii in xrange(len(self.series)-1):
            obj1, obj2 = self.series[ii].obj_list, self.series[ii+1].obj_list
            points1 = np.array([self.series[ii].region_props[obj]['centroid'] for obj in obj1])
            points2 = np.array([self.series[ii+1].region_props[obj]['centroid']-self.shifts[ii] for obj in obj2])

            m12, m21 = match_func(points2, points1, dmax)
            for parent, child in zip(obj1, m12):
                self.series[ii].children[parent].append(obj2[child])

            for child, parent in zip(obj2, m21):
                self.series[ii+1].parents[child].append(obj1[parent])

            print "matched", (m12>-1).sum(), 'objects.', (m12==-1).sum() , (m21==-1).sum(), 'left unmatched in time step', ii, ii+1, 'respectively'

    def make_colormaps(self):
        from matplotlib import cm
        self.colormaps = []
        self.colormaps.append(cm.jet(np.arange(len(self.series[0].labeled_obj))))
        for ii in xrange(1,len(self.series)):
            self.colormaps.append(cm.jet())
    
    def plot_time_frame(self,ti):
        plt.imshow(self.series[ii].seg_img, cm=self.colormaps[ii])

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
    
    # 
    test_series = labeled_series()
    test_series.load_from_file('../Movie_sample/Yutao1-17_1-t-???_seg.tif', '../Movie_sample/Yutao1-17_1-t-???.tif')
    test_series.calc_image_shifts()
    test_series.track_objects(match_func=match_mindist)

    
