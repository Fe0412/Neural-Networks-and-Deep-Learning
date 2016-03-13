import glob, os
from os import walk
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from PIL import Image

import theano
import theano.tensor as T
from theano.tensor.nnet.neighbours import images2neibs,neibs2images
rng = np.random

'''
Implement the functions that were not implemented and complete the
parts of main according to the instructions in comments.
'''

def plot_mul(c, D, im_num, X_mn, num_coeffs):
    '''
    Plots nine PCA reconstructions of a particular image using number
    of components specified by num_coeffs

    Parameters
    ---------------
    c: np.ndarray
        a n x m matrix  representing the coefficients of all the image blocks.
        n represents the maximum dimension of the PCA space.
        m is (number of images x n_blocks**2)

    D: np.ndarray
        an N x n matrix representing the basis vectors of the PCA space
        N is the dimension of the original space (number of pixels in a block)

    im_num: Integer
        index of the image to visualize

    X_mn: np.ndarray
        a matrix representing the mean block.

    num_coeffs: Iterable
        an iterable with 9 elements representing the number_of coefficients
        to use for reconstruction for each of the 9 plots

    n_blocks: Integer
        number of blocks comprising the image in each direction.
        For example, for a 256x256 image divided into 64x64 blocks, n_blocks will be 4
    '''
    f, axarr = plt.subplots(3, 3)

    for i in range(3):
        for j in range(3):
            nc = num_coeffs[i*3+j]
            cij = c[:nc, im_num]
            Dij = D[:, :nc]
            plot(cij, Dij, X_mn, axarr[i, j])

    #plt.show()
    f.savefig('output/hw1b_im{0}.png'.format(im_num))
    plt.close(f)

def plot_top_16(D, sz, imname):
    '''
    Plots the top 16 components from the basis matrix D.
    Each basis vector represents an image block of shape (sz, sz)

    Parameters
    -------------
    D: np.ndarray
        N x n matrix representing the basis vectors of the PCA space
        N is the dimension of the original space (number of pixels in a block)
        n represents the maximum dimension of the PCA space (assumed to be atleast 16)

    sz: Integer
        The height and width of each block

    imname: string
        name of file where image will be saved.
    '''
    f, axarr = plt.subplots(4, 4)
    for i in range(16):
        temp = D[:,i].reshape(sz,sz)
        pyplot.subplot(axarr[i/4, i%4])
        pyplot.axis('off')
        pyplot.imshow(temp, cmap = cm.Greys_r)
    #pyplot.show()
    f.savefig(imname)
    plt.close(f)

def plot(c, D, X_mn, ax):
    '''
    Plots a reconstruction of a particular image using D as the basis matrix and coeffiecient
    vectors from c

    Parameters
    ------------------------
        c: np.ndarray
            a l x m matrix  representing the coefficients of all blocks in a particular image
            l represents the dimension of the PCA space used for reconstruction
            m represents the number of blocks in an image

        D: np.ndarray
            an N x l matrix representing l basis vectors of the PCA space
            N is the dimension of the original space (number of pixels in a block)

        n_blocks: Integer
            number of blocks comprising the image in each direction.
            For example, for a 256x256 image divided into 64x64 blocks, n_blocks will be 4

        X_mn: basis vectors represent the divergence from the mean so this
            matrix should be added to all reconstructed blocks

        ax: the axis on which the image will be plotted
    '''
   
    Y = np.dot(c, D.T)
    Y = Y.reshape(256,256)
    Y = Y + X_mn
    
    pyplot.subplot(ax)
    pyplot.imshow(Y, cmap = cm.Greys_r)
    #pyplot.show()
    
    #raise NotImplementedError


def main():
    '''
    Read here all images(grayscale) from jaffe folder
    into an numpy array Ims with size (no_images, height, width).
    Make sure the images are read after sorting the filenames
    '''
    
    path = "jaffe/"
    no_image = len(glob.glob1(path,"*.tiff"))
    
    file_names = []
    for root, dirs, files in walk('jaffe/', topdown=True):
        for name in files:
            file_names.append(name)
 
    file_names.sort()
    I = np.zeros((no_image, 256*256))
    for i in range(no_image):
        img = Image.open('jaffe/' + file_names[i])
        temp = np.asarray(img)
        I[i, :] = temp.reshape((256*256,))
    
    '''
    #reading images
    path = "jaffe/"
    os.chdir(path)

    no_image = len(glob.glob1(path,"*.tiff"))
    i = 0
    I = np.zeros((no_image,256*256))
    for file in glob.glob("*.tiff"):
        x = str(file)
        im = Image.open(x).convert("L")
        temp = np.asarray(im)
        #pyplot.imshow(temp, cmap = cm.Greys_r)
        #pyplot.show()

        j = 0
        k = 0
        for j in range(256*256):
            I[i][j] = temp[j/256][j%256]
        i = i + 1
    '''
    
    
    Ims = I.astype(np.float32)
    X_mn = np.mean(Ims, 0)
    X = Ims - np.repeat(X_mn.reshape(1, -1), Ims.shape[0], 0)
    print X.shape
    
    

    '''
    Use theano to perform gradient descent to get top 16 PCA components of X
    Put them into a matrix D with decreasing order of eigenvalues

    If you are not using the provided AMI and get an error "Cannot construct a ufunc with more than 32 operands" :
    You need to perform a patch to theano from this pull(https://github.com/Theano/Theano/pull/3532)
    Alternatively you can downgrade numpy to 1.9.3, scipy to 0.15.1, matplotlib to 1.4.2
    '''
    
    x = T.dmatrix()
    Dj = T.dmatrix()
    lamb = T.dvector()
    d = theano.shared(np.random.randn(65536))
    xtx = T.dot(T.dot(x, d).T, T.dot(x, d))
    ddt = T.dot(lamb * T.dot(Dj, d).T, T.dot(Dj, d))
    cost = xtx - ddt
    gd = T.grad(cost, d)
    loop_body = theano.function(inputs = [x, Dj, lamb],
                                outputs = d,
                                updates = [(d, (d - 0.1 * gd) / T.sqrt(T.sum(T.sqr(d - 0.1 * gd))))]
                               )

    lambdas = []
    D = np.empty(shape=(0, 65536))
    for i in range(16):
        for t in range(100):
            d_temp = loop_body(X, D, lambdas)

        D = np.vstack((D, d_temp))
        la_temp = np.dot(np.dot(X, d_temp).T, np.dot(X, d_temp))
        lambdas = np.append(lambdas, la_temp)
        d.set_value(np.random.randn(65536))
        print("{}th principal component".format(i))

    D = D.T
    c = np.dot(D.T, X.T)
  
    for i in range(0,200,10):
        print("{}th reconstruction".format(i))
        plot_mul(c, D, i, X_mn.reshape((256, 256)), 
                 [1, 2, 4, 6, 8, 10, 12, 14, 16])

    plot_top_16(D, 256, 'output/hw1b_top16_256.png')
    print("end of program")



if __name__ == '__main__':
    main()
