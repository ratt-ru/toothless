## Fusion Model classification
# Arun Aniyan
# SKA SA/ RATT
# arun@ska.ac.za
# 18-02-17

# Input can be either fits image or jpg/png

# Import necessary stuff

import sys
import os
import time
import datetime
from collections import Counter

import PIL.Image
import numpy as np
import scipy.misc
from google.protobuf import text_format

os.environ['GLOG_minloglevel'] = '2' # Suppress most caffe output
import caffe
from caffe.proto import caffe_pb2

from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from scipy.misc import imsave
from skimage.transform import resize
from skimage.color import rgb2gray


# Function definitions

# Load model
def get_net(caffemodel, deploy_file, use_gpu=True):
    """
    Returns an instance of caffe.Net

    Arguments:
    caffemodel -- path to a .caffemodel file
    deploy_file -- path to a .prototxt file

    Keyword arguments:
    use_gpu -- if True, use the GPU for inference
    """
    #if use_gpu:
    #    caffe.set_mode_gpu()
    caffe.set_mode_cpu()

    # load a new model
    return caffe.Net(deploy_file, caffemodel, caffe.TEST)

# Transformer function to perform image transformation
def get_transformer(deploy_file, mean_file=None):
    """
    Returns an instance of caffe.io.Transformer

    Arguments:
    deploy_file -- path to a .prototxt file

    Keyword arguments:
    mean_file -- path to a .binaryproto file (optional)
    """
    network = caffe_pb2.NetParameter()
    with open(deploy_file) as infile:
        text_format.Merge(infile.read(), network)

    if network.input_shape:
        
        dims = network.input_shape[0].dim
    else:
        dims = network.input_dim[:4]
    

    #dims = network.input_dim

    t = caffe.io.Transformer(
            inputs = {'data': dims}
            )
    t.set_transpose('data', (2,0,1)) # transpose to (channels, height, width)

    # color images
    if dims[1] == 3:
        # channel swap
        t.set_channel_swap('data', (2,1,0))

    if mean_file:
        # set mean pixel
        with open(mean_file) as infile:
            blob = caffe_pb2.BlobProto()
            blob.MergeFromString(infile.read())
            if blob.HasField('shape'):
                blob_dims = blob.shape
                assert len(blob_dims) == 4, 'Shape should have 4 dimensions - shape is "%s"' % blob.shape
            elif blob.HasField('num') and blob.HasField('channels') and \
                    blob.HasField('height') and blob.HasField('width'):
                blob_dims = (blob.num, blob.channels, blob.height, blob.width)
            else:
                raise ValueError('blob does not provide shape or 4d dimensions')
            pixel = np.reshape(blob.data, blob_dims[1:]).mean(1).mean(1)
            t.set_mean('data', pixel)

    return t

# Load image to caffe 
def load_image(path, height, width, mode='RGB'):
    """
    Load an image from disk

    Returns an np.ndarray (channels x width x height)

    Arguments:
    path -- path to an image on disk
    width -- resize dimension
    height -- resize dimension

    Keyword arguments:
    mode -- the PIL mode that the image should be converted to
        (RGB for color or L for grayscale)
    """

    image = PIL.Image.open(path)
    image = image.convert(mode)
    image = np.array(image)
    # squash
    image = scipy.misc.imresize(image, (height, width), 'bilinear')
    return image

# Forward pass of input through the network
def forward_pass(images, net, transformer, batch_size=1):
    """
    Returns scores for each image as an np.ndarray (nImages x nClasses)

    Arguments:
    images -- a list of np.ndarrays
    net -- a caffe.Net
    transformer -- a caffe.io.Transformer

    Keyword arguments:
    batch_size -- how many images can be processed at once
        (a high value may result in out-of-memory errors)
    """
    caffe_images = []
    for image in images:
        if image.ndim == 2:
            caffe_images.append(image[:,:,np.newaxis])
        else:
            caffe_images.append(image)

    caffe_images = np.array(caffe_images)

    dims = transformer.inputs['data'][1:]

    scores = None
    for chunk in [caffe_images[x:x+batch_size] for x in xrange(0, len(caffe_images), batch_size)]:
        new_shape = (len(chunk),) + tuple(dims)
        if net.blobs['data'].data.shape != new_shape:
            net.blobs['data'].reshape(*new_shape)
        for index, image in enumerate(chunk):
            image_data = transformer.preprocess('data', image)
            net.blobs['data'].data[index] = image_data
        output = net.forward()[net.outputs[-1]]
        if scores is None:
            scores = output
        else:
            scores = np.vstack((scores, output))
        #print 'Processed %s/%s images ...' % (len(scores), len(caffe_images))

    return scores

# Resolve labels
def read_labels(labels_file):
    """
    Returns a list of strings

    Arguments:
    labels_file -- path to a .txt file
    """
    if not labels_file:
        print 'WARNING: No labels file provided. Results will be difficult to interpret.'
        return None

    labels = []
    with open(labels_file) as infile:
        for line in infile:
            label = line.strip()
            if label:
                labels.append(label)
    assert len(labels), 'No labels found'
    return labels


# Decide class based on threshold
def decide(classification):
    lbl = []
    conf = []
    for label, confidence in classification:
        lbl.append(label)
        conf.append(confidence)
    idx = np.argmax(conf)

    return lbl[idx],conf[idx]

# Perform Single classification
def classify(caffemodel, deploy_file, image_files,
        mean_file=None, labels_file=None, use_gpu=True):
    """
    Classify some images against a Caffe model and print the results

    Arguments:
    caffemodel -- path to a .caffemodel
    deploy_file -- path to a .prototxt
    image_files -- list of paths to images

    Keyword arguments:
    mean_file -- path to a .binaryproto
    labels_file path to a .txt file
    use_gpu -- if True, run inference on the GPU
    """
    # Load the model and images
    net = get_net(caffemodel, deploy_file, use_gpu)
    transformer = get_transformer(deploy_file, mean_file)
    _, channels, height, width = transformer.inputs['data']
    if channels == 3:
        mode = 'RGB'
    elif channels == 1:
        mode = 'L'
    else:
        raise ValueError('Invalid number for channels: %s' % channels)
    images = [load_image(image_file, height, width, mode) for image_file in image_files]
    labels = read_labels(labels_file)

    # Classify the image
    classify_start_time = time.time()
    scores = forward_pass(images, net, transformer)
    #print 'Classification took %s seconds.' % (time.time() - classify_start_time,)

    ### Process the results

    indices = (-scores).argsort()[:, :5] # take top 5 results
    classifications = []
    for image_index, index_list in enumerate(indices):
        result = []
        for i in index_list:
            # 'i' is a category in labels and also an index into scores
            if labels is None:
                label = 'Class #%s' % i
            else:
                label = labels[i]
            result.append((label, round(100.0*scores[image_index, i],4)))
        classifications.append(result)

    for index, classification in enumerate(classifications):
        
        #print '{:-^80}'.format(' Prediction for %s ' % image_files[index])
        
        lbl, conf = decide(classification)

    return lbl,conf


# Fusion decision model
def vote(ypreds,probs,thresh):
	# Find the repeating class among the three models
	high_vote = [item for item, count in Counter(ypreds).items() if count >1]

	if high_vote !=[]:
		final_class = high_vote[0]
		# Check if their probability is greater than 60
		idx = np.where(np.array(ypreds)==final_class)[0]
		if float(probs[idx[0]]) > thresh or float(probs[idx[1]]) > thresh:
			final_classification = final_class
			final_probability = (float(probs[idx[0]])+float(probs[idx[1]]))/2.0
		else:
			final_classification = final_class +'?'
			final_probability = min(float(probs[idx[0]]),float(probs[idx[1]]))
	else:
		final_classification = 'Strange'
		final_probability = 0

	return final_classification,final_probability

def checkfits(filename):
    if filename.rsplit('.',1)[1] == 'fits':
        image = fits2jpg(filename)
    return image


# Clipper function
def clip(data,lim):
    data[data<lim] = 0.0
    return data 

# Convert fits image to png
def fits2jpg(fname):
    hdu_list = fits.open(fname)
    image = hdu_list[0].data
    image = np.squeeze(image)
    img = np.copy(image)
    idx = np.isnan(img)
    img[idx] = 0
    img_clip = np.flipud(img)
    sigma = 3.0
    # Estimate stats
    mean, median, std = sigma_clipped_stats(img_clip, sigma=sigma, iters=10)
    # Clip off n sigma points
    img_clip = clip(img_clip,std*sigma)
    if img_clip.shape[0] !=150 or img_clip.shape[1] !=150:
        img_clip = resize(img_clip, (150,150))
    #img_clip = rgb2gray(img_clip)
    
    outfile = fname[0:-5] +'.png'
    imsave(outfile, img_clip)
    return img_clip,outfile




# Do the fusion classification
def fusion_classify(image_file):

    # Change location of files appropriately
    models = ['Prototxt/fr1vsfr2.prototxt','Prototxt/fr1vsbent.prototxt','Prototxt/fr2vsbent.prototxt']
    nets = ['Models/fr1vsfr2.caffemodel','Models/fr1vsbent.caffemodel','Models/fr2vsbent.caffemodel']
    labels = ['Labels/fr1vsfr2-label.txt','Labels/fr1vsbent-label.txt','Labels/fr2vsbent-label.txt']

    thresh = 90 # Decision cut of to make the final classification

    ypreds = []
    probs = []


    if image_file.rsplit('.',1)[1] == 'fits':
        image, outfile = fits2jpg(image_file)
        image_file = outfile


    for i in range(3):

        
        lbl, conf = classify(nets[i], models[i],[image_file],labels_file=labels[i])

        ypreds.append(lbl)
        probs.append(conf)

    classlabel,probability = vote(ypreds, probs, thresh)

    probability = round(probability,2)

    return classlabel,probability



if __name__ == '__main__':

	script_start_time = time.time()

	arg = sys.argv
	image_file = arg[1]

	# Extract filename without extension and root path
	filename = os.path.basename(image_file)
	filename = os.path.splitext(filename)[0]

	classlabel, probability = fusion_classify(image_file)
    
	print '%s is classified as %s with %.2f%% confidence.' %(filename,classlabel,probability)
	print 'Script took %s seconds.' % (time.time() - script_start_time,)






