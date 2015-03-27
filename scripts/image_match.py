# import the necessary packages
from optparse import OptionParser
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pickle
import sys
import hashlib
import masir

###########################
def image_match_csift( all_files, options ):

    # features are computed from SMQTK (https://github.com/Kitware/SMQTK.git)
    map_ingest_fname = {}
    image_files = []

    ingest = masir.IngestManager( options.dpath )
    ingest_files = dict( ingest.items() )

    # flag to avoid duplicates
    found_flags = np.zeros( len(ingest_files), dtype=np.int )

    # loop over all images
    for (i, fname) in enumerate(all_files):
        if options.ipath:
            path_fname = options.ipath + '/' + fname
        else:
            path_fname = fname

        # read in image
        image = cv2.imread( path_fname );

        if image is None:
            print path_fname + " : fail to read"
            continue

        md5 = hashlib.md5(open(path_fname, 'rb').read()).hexdigest()

        for j in range( len(ingest_files) ):
            if (found_flags[j] == 0) & (ingest_files[j].find(md5) > 0):
                map_ingest_fname[ ingest_files[j] ] = fname
                found_flags[j] = 1
                break
        if j > len(ingest_files)-1:
            print "Cannot find matching file " + fname
            sys.exit()

    # file list
    for j in range( len(ingest_files) ):
        print j, ingest_files[j], map_ingest_fname[ ingest_files[j] ]
        image_files.append( map_ingest_fname[ ingest_files[j] ] )
    pickle.dump( image_files, open( options.opath + "/csift_files.p","wb") )

    # feature matrix
    feature_matrix = np.load( open( options.dpath + "features/colordescriptor-csift/feature_data.npy" ) )
    pickle.dump( feature_matrix, open( options.opath+"/csift_feature.p","wb") )

    # affinity matrix
    dists = np.load( open( options.dpath + "features/colordescriptor-csift/kernel_data.npy") )
    pickle.dump( dists, open( options.opath+"/csift_affinity.p","wb") )

    # K nearest neighbors
    knn = {}
    k=int(options.top)
    for (i, fi) in enumerate(image_files):
        vec = sorted( zip(dists[i,:], image_files), reverse = True )
        knn[fi] = vec[:k]
        print knn[fi]
    pickle.dump( knn, open( options.opath+"/csift_knn.p","wb") )

    # Kmeans clustering
    term_crit = (cv2.TERM_CRITERIA_EPS, 100, 0.01)
    print feature_matrix

    ret, labels, centers = cv2.kmeans(np.float32(feature_matrix), int(options.cluster_count), term_crit, 10, cv2.KMEANS_RANDOM_CENTERS )

    label_list=[]
    for (i,l) in enumerate(labels):
        label_list.append(l[0])
    print label_list

    image_label = zip( image_files, label_list )
    print image_label
    pickle.dump( image_label, open( options.opath+"/csift_clustering.p","wb") )

###########################
def image_match_histogram( all_files, options ):
    histograms = {}
    image_files = []

    # loop over all images
    for (i, fname) in enumerate(all_files):
        if options.ipath:
            path_fname = options.ipath + '/' + fname
        else:
            path_fname = fname

        # read in image
        image = cv2.imread( path_fname );

        if image is None:
            print path_fname + " : fail to read"
            continue

        image_files.append(fname)

        if image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        print i, path_fname, image.shape

        v = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        v = v.flatten()
        hist = v / sum(v)
        histograms[fname] = hist

    # feature matrix
    feature_matrix = np.zeros( (len(histograms), len(hist)) )
    for (i,fi) in enumerate(image_files):
        feature_matrix[i,:] = histograms[image_files[i]]
    pickle.dump( feature_matrix, open( options.opath+"/color_featue.p","wb") )

    dists = np.zeros((len(image_files), len(image_files)))
    knn = {}
    # pairwise comparison
    for (i, fi) in enumerate(image_files):
        for (j, fj) in enumerate(image_files):
            if i <= j:
                d = cv2.compareHist( histograms[fi], histograms[fj], cv2.cv.CV_COMP_INTERSECT)
                dists[i,j] = d
                dists[j,i] = d

    pickle.dump( dists, open( options.opath+"/color_affinity.p","wb") )

    # K nearest neighbors
    k=int(options.top)
    for (i, fi) in enumerate(image_files):
        vec = sorted( zip(dists[i,:], image_files), reverse = True )
        knn[fi] = vec[:k]
        print knn[fi]
    pickle.dump( knn, open( options.opath+"/color_knn.p","wb") )

    # Kmeans clustering
    term_crit = (cv2.TERM_CRITERIA_EPS, 100, 0.01)
    print feature_matrix

    ret, labels, centers = cv2.kmeans(np.float32(feature_matrix), int(options.cluster_count), term_crit, 10, cv2.KMEANS_RANDOM_CENTERS )

    label_list=[]
    for (i,l) in enumerate(labels):
        label_list.append(l[0])
    print label_list

    image_label = zip( image_files, label_list )
    print image_label
    pickle.dump( image_label, open( options.opath+"/color_clustering.p","wb") )

###########################
def main():
    usage = "usage: %prog [options] image_list_file \n"
    usage += " image match"
    parser = OptionParser(usage=usage)

    parser.add_option("-i", "--input_path", default="",
                      action="store", dest="ipath",
                      help="input path")

    parser.add_option("-o", "--output_path", default=".",
                      action="store", dest="opath",
                      help="output path")

    parser.add_option("-a", "--data_path", default="",
                      action="store", dest="dpath",
                      help="csift data path")

    parser.add_option("-f", "--feature", default="color_histogram",
                      action="store", dest="feature",
                      help="color_histogram; sift_match;dist_info")

    parser.add_option("-m", "--method", default="Intersection",
                      action="store", dest="method",
                      help="Intersection;L1;L2")

    parser.add_option("-t", "--top", default="5",
                      action="store", dest="top",
                      help="Top nearest neighbors")

    parser.add_option("-c", "--cluster_count", default="3",
                      action="store", dest="cluster_count",
                      help="Number of clusters")

    parser.add_option("-d", "--debug", default="0",
                      action="store", dest="debug_mode",
                      help="debug intermediate results")

    (options, args) = parser.parse_args()

    if len(args) < 1 :
        print "Need one argument: image_list_file \n"
        sys.exit()

    image_files = [line.strip() for line in open(args[0])]

    if options.feature == "color_histogram":
        image_match_histogram( image_files, options )
    elif options.feature == "csift":
        image_match_csift( image_files, options )

if __name__=="__main__":
    main()
