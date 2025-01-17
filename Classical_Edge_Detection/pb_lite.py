#!/usr/bin/env python

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import argparse
import skimage.transform
import scipy.stats as st
import sklearn.cluster
from folders import * # Importing folder management and creation

def Gaussiankernel2D(kernelsize, sigma):
    """
    Generate a 2D Gaussian kernel.
    Parameters:
        kernelsize (int): Size of the kernel.
        sigma (float): Standard deviation of the Gaussian.
    Returns:
        np.ndarray: 2D Gaussian kernel.
    """
#     nsig = scales*scales
    spacing = (2*sigma+1)/kernelsize
    x = np.linspace((-sigma-spacing)/2, (sigma+spacing)/2, kernelsize+1)
    kernel1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kernel1d, kernel1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel

def Gaussian1D(sigma, mean, x, order_derivative):
    """
    Generate a 1D Gaussian function or its derivatives.
    Parameters:
        sigma (float): Standard deviation.
        mean (float): Mean of the Gaussian.
        x (np.ndarray): Range of values.
        order_derivative (int): Derivative order (0, 1, or 2).
    Returns:
        np.ndarray: 1D Gaussian or its derivative.
    """
    x = np.array(x)
    x_ = x - mean
    vari = sigma**2

    # Gaussian Function
    # g1 = (np.exp((-1*x_*x_)/(2*vari)))*(1/np.sqrt(2*np.pi*vari))
    g1 = np.exp(-0.5 * (x_**2 / vari)) / np.sqrt(2*np.pi*vari)
    if order_derivative == 0:
        return g1
    elif order_derivative == 1:
        return -g1 * (x_ / vari)
    else:
        # second derivative
        return g1 * ((x_**2 - vari) / (vari**2))

def Gaussian2D(kernelsize, sigma):
    """
    Generate a 2D Gaussian function.
    Parameters:
        kernelsize (int): Size of the kernel.
        sigma (float): Standard deviation of the Gaussian.
    Returns:
        np.ndarray: 2D Gaussian kernel
    """
    vari = sigma * sigma
    shape = (kernelsize,kernelsize)
    n,m = [(i - 1)//2 for i in shape]
    x,y = np.ogrid[-m:m+1,-n:n+1]
    g = np.exp(-(x*x + y*y) / (2*vari)) / (2*np.pi*vari)
    return g / g.sum()

def lap_gaussian2D(kernelsize, sigma):
    """
    Generate a 2D Laplacian of Gaussian (LoG) kernel.  
    Parameters:
        kernelsize (int): Size of the kernel.
        sigma (float): Standard deviation of the Gaussian.
    Returns:
        np.ndarray: Laplacian of Gaussian kernel.
    """
    vari = sigma * sigma
    shape = (kernelsize,kernelsize)
    n,m = [(i - 1)//2 for i in shape]
    x, y = np.ogrid[-m:m+1, -n:n+1]
    g = np.exp(-(x*x + y*y) / (2*vari)) / (2*np.pi*vari)
    # LoG
    h = g * ((x*x + y*y) - vari) / (vari**2)
    # Optional: normalize so sum of absolute values is 1
    h = h / (np.sum(np.abs(h)) + 1e-10)
    return h

def makefilter(sigma, x_orient, y_orient, pts, kernelsize):
    """
    Create a 2D filter using Gaussian derivatives.
    Parameters:
        sigma (float): Scale of the Gaussian.
        x_orient (int): Orientation for x-derivative.
        y_orient (int): Orientation for y-derivative.
        pts (np.ndarray): Grid of points.
        kernelsize (int): Size of the kernel.
    Returns:
        np.ndarray: Filter kernel.
    """
    gx = Gaussian1D(3*sigma, 0, pts[0,...], x_orient)
    gy = Gaussian1D(sigma,   0, pts[1,...], y_orient)

    image = gx*gy

    image = np.reshape(image,(kernelsize,kernelsize))
    im_max = np.max(np.abs(image)) + 1e-10
    image = image / im_max
    return image

#Define DOG Filters
def Oriented_DOG(sigma,orient,size):
    """
    Generate an oriented Difference of Gaussian (DoG) filter bank.   
    Parameters:
        sigma (list of float): List of sigma values for scales.
        orient (int): Number of orientations.
        size (int): Size of the kernel.   
    Returns:
        list: List of DoG filters.
    """
    kernels=[]
    border = cv2.BORDER_DEFAULT
    for scale in sigma:
        orients=np.linspace(0,360,orient, endpoint=False)
        kernel=Gaussiankernel2D(size,scale)
        sobelx64f = cv2.Sobel(kernel,cv2.CV_64F,1,0,ksize=3, borderType=border)
        # Normalize derivative
        smax = np.max(np.abs(sobelx64f)) + 1e-10
        sobelx64f /= smax
        for eachOrient in orients:
            rotated = skimage.transform.rotate(sobelx64f, eachOrient, mode='reflect')
            # Normalize
            rmax = np.max(np.abs(rotated)) + 1e-10
            rotated /= rmax
            kernels.append(rotated)
    return kernels

#Define LM filters
def LM_filters(kernelsize, sigma, num_orientations, nrotinv):
    """
    Generate Leung-Malik (LM) filter bank.
    Parameters:
        kernelsize (int): Size of the kernel.
        sigma (int): Number of scales.
        num_orientations (int): Number of orientations.
        nrotinv (int): Number of rotationally invariant filters.
    Returns:
        np.ndarray: Array of LM filters.
    """
    scalex  = np.sqrt(2) * np.arange(1,sigma+1)
    nbar  = len(scalex)*num_orientations
    nedge = len(scalex)*num_orientations
    nf    = nbar+nedge+nrotinv
    F     = np.zeros([kernelsize,kernelsize,nf], dtype=np.float32)
    hkernelsize  = (kernelsize - 1)/2

    x_range = np.arange(-hkernelsize, hkernelsize+1)
    x,y = np.meshgrid(x_range,x_range)
    orgpts =np.vstack([x.flatten(), y.flatten()])
    count = 0

    for scaleVal in scalex:
        for orient in range(num_orientations):
            angle = (math.pi * orient)/num_orientations
            c = np.cos(angle)
            s = np.sin(angle)
            rot_mat = np.array([[c, -s], [s, c]], dtype=np.float32)
            rotpts = rot_mat.dot(orgpts)
            F[:,:,count] = makefilter(scaleVal, 0, 1, rotpts, kernelsize)
            F[:,:,count+nedge] = makefilter(scaleVal, 0, 2, rotpts, kernelsize)
            count = count + 1
    count = nbar+nedge
    sigma_vals = np.sqrt(2) * np.array([1,2,3,4], dtype=np.float32)

    for i in range(len(sigma_vals)):
        g2d = Gaussian2D(kernelsize, sigma_vals[i])
        # Normalized
        g2d /= (np.max(np.abs(g2d)) + 1e-10)
        F[:, :, count] = g2d
        count += 1

    for i in range(len(sigma_vals)):
        log_2d = lap_gaussian2D(kernelsize, sigma_vals[i])
        F[:, :, count] = log_2d
        count += 1

    for i in range(len(sigma_vals)):
        log_2d = lap_gaussian2D(kernelsize, 3*sigma_vals[i])
        F[:, :, count] = log_2d
        count += 1

    return F

def gabor_fn(sigma, theta, Lambda, psi, gamma):
    """
    Generates a Gabor filter kernel.
    Parameters:
        sigma (float): Standard deviation of the Gaussian envelope.
        theta (float): Orientation of the Gabor filter (in radians).
        Lambda (float): Wavelength of the sinusoidal carrier.
        psi (float): Phase offset of the sinusoidal carrier.
        gamma (float): Spatial aspect ratio (ellipticity) of the Gabor filter.
    Returns:
        np.ndarray: A 2D Gabor filter kernel.
    """
    sigma_x = sigma
    sigma_y = float(sigma) / gamma

    # Bounding box
    nstds = 3 # Number of standard deviation sigma
    x_max = int(np.ceil(max(abs(nstds*sigma_x*np.cos(theta)),
                            abs(nstds*sigma_y*np.sin(theta)))))
    y_max = int(np.ceil(max(abs(nstds*sigma_x*np.sin(theta)),
                            abs(nstds*sigma_y*np.cos(theta)))))
    x_min, y_min = -x_max, -y_max
    (y, x) = np.meshgrid(np.arange(y_min, y_max + 1), np.arange(x_min, x_max + 1))

    # Rotation
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)
    gb = np.exp(
        -0.5 * (x_theta**2 / (sigma_x**2) + y_theta**2 / (sigma_y**2))
    ) * np.cos(2*np.pi / Lambda * x_theta + psi)
    # Normalize
    denom = np.sqrt((gb**2).sum()) + 1e-10
    gb = gb / denom
    #plt.imshow(gb,cmap='binary')
    return gb
    #4, 0.25, 1, 1.0, 1

#Define Gabor Filters
def Gabor_filter(sigma, theta, Lambda, psi, gamma,num_filters):
    """
    Creates a set of Gabor filters with varying orientations.
    Parameters:
        sigma (list): List of standard deviations for the Gaussian envelope.
        theta (float): Fixed orientation for the base Gabor filter.
        Lambda (float): Wavelength of the sinusoidal carrier.
        psi (float): Phase offset of the sinusoidal carrier.
        gamma (float): Spatial aspect ratio (ellipticity) of the Gabor filter.
        num_filters (int): Number of filters to generate with different orientations.
    Returns:
        list: A list of 2D Gabor filter kernels.
    """
    g = []
    ang = np.linspace(0, 360, num_filters, endpoint=False)  # 0-180 initially
    for s in sigma:
        base = gabor_fn(s, theta, Lambda, psi, gamma)
        for a in ang:
            image = skimage.transform.rotate(base,a,mode='reflect')
            # Re-normalize after rotation
            imax = np.max(np.abs(image)) + 1e-10
            image /= imax
            g.append(image)
    return g

#Define Texton Map by using DOG filters
def textonmap_DOG(Img, filter_bank):
    """
    Generates a texton map using a Difference of Gaussian (DoG) filter bank.
    Parameters:
        Img (np.ndarray): Input image.
        filter_bank (list): List of DoG filter kernels.
    Returns:
        np.ndarray: A texton map with filter responses stacked along the depth dimension.
    """
    tex_map = np.array(Img)
    #_,_,num_filters = filter_bank.shape
    for i in range(len(filter_bank)):
        #out = cv2.filter2D(img,-1,filter_bank[:,:,i])
        out = cv2.filter2D(Img,-1,filter_bank[i])
        tex_map = np.dstack((tex_map, out))
    return tex_map

#Define Texton Map using LM filters
def textonmap_LM(Img, filter_bank ):
    """
    Generates a texton map using an LM (Leung-Malik) filter bank.
    Parameters:
        Img (np.ndarray): Input image.
        filter_bank (np.ndarray): A 3D array representing the LM filter bank.
    Returns:
        np.ndarray: A texton map with filter responses stacked along the depth dimension.
    """
    tex_map = np.array(Img)
    _,_,num_filters = filter_bank.shape
    #num_filters = len(filter_bank)
    for i in range(num_filters):
        out = cv2.filter2D(Img,-1,filter_bank[:,:,i])
        #out = cv2.filter2D(img,-1,filter_bank[i])
        tex_map = np.dstack((tex_map, out))
    return tex_map

def Texton(img,filter_bank1,filter_bank2,filter_bank3, num_clusters):
    """
    Computes a texton map by applying multiple filter banks and clustering responses.
    Parameters:
        img (np.ndarray): Input color image.
        filter_bank1 (np.ndarray): LM filter bank.
        filter_bank2 (list): DoG filter bank.
        filter_bank3 (list): Gabor filter bank.
        num_clusters (int): Number of clusters for K-means clustering.
    Returns:
        np.ndarray: Texton map with clustered texture IDs.
    """
    p,q,_ = img.shape
    #weights are optional
    weights = [0.3,0.3,0.4] 
    tex_map_DOG = textonmap_DOG(img, filter_bank2) * weights[0]
    tex_map_LM = textonmap_LM(img, filter_bank1) * weights[1]
    tex_map_Gabor = textonmap_DOG(img, filter_bank3) *weights[2]
    combined = np.dstack((
        tex_map_DOG[:, :, 1:],
        tex_map_LM[:, :, 1:],
        tex_map_Gabor[:, :, 1:]
    ))
    m,n,r = combined.shape
    inp = np.reshape(combined,((p*q),r))
    kmeans = sklearn.cluster.KMeans(n_clusters = num_clusters, random_state = 0)
    kmeans.fit(inp)
    labels = kmeans.predict(inp)
    l = np.reshape(labels,(m,n))
    plt.imshow(l)
    return l

def Brightness(Img, num_clusters):
    """
    Generates a brightness map by clustering pixel intensities.
    Parameters:
        Img (np.ndarray): Input image (grayscale or color).
        num_clusters (int): Number of clusters for K-means clustering.
    Returns:
        np.ndarray: Brightness map with clustered intensity levels.
    """
    p,q,r = Img.shape
    inp = np.reshape(Img,((p*q),r))
    kmeans = sklearn.cluster.KMeans(n_clusters = num_clusters, random_state = 0)
    kmeans.fit(inp)
    labels = kmeans.predict(inp)
    l = np.reshape(labels,(p,q))
    plt.imshow(l,cmap = 'binary')
    return l

def Color(Img, num_clusters):
    """
    Generates a color map by clustering pixel color values.
    Parameters:
        Img (np.ndarray): Input color image.
        num_clusters (int): Number of clusters for K-means clustering.
    Returns:
        np.ndarray: Color map with clustered color levels.
    """  
    p,q,r = Img.shape
    inp = np.reshape(Img,((p*q),r))
    kmeans = sklearn.cluster.KMeans(n_clusters = num_clusters, random_state = 0)
    kmeans.fit(inp)
    labels = kmeans.predict(inp)
    l = np.reshape(labels,(p,q))
    plt.imshow(l)
    return l

def chi_sqr_gradient(Img, bins, filter1, filter2):
    """
    Computes a gradient map using the Chi-squared distance between histograms.
    Parameters:
        Img (np.ndarray): Input image (e.g., a clustered map).
        bins (int): Number of histogram bins.
        filter1 (np.ndarray): First filter for gradient computation.
        filter2 (np.ndarray): Second filter for gradient computation.
    Returns:
        np.ndarray: Gradient map based on Chi-squared distance.
    """
    epsilon = 1e-10
    chi_sqr_dist = Img*0
    for i in range(bins):
        #numpy.ma.masked_where(condition, a, copy=True)[source]
        #Mask an array where a condition is met.
        img_mask = np.ma.masked_where(Img == i,Img).mask.astype(np.int32)
        g = cv2.filter2D(img_mask,-1,filter1)
        h = cv2.filter2D(img_mask,-1,filter2)
        chi_sqr_dist = chi_sqr_dist +((g-h)**2) /((g+h)+ epsilon)
    return chi_sqr_dist/2

def Gradient(Img, bins, filter_bank):
    """
    Computes a gradient map using a filter bank.
    Parameters:
        Img (np.ndarray): Input image (e.g., a clustered map).
        bins (int): Number of histogram bins.
        filter_bank (list): List of filters for gradient computation.
    Returns:
        np.ndarray: Average gradient map across all filters.
    """    
    gradVar = Img
    for N in range(math.ceil(len(filter_bank)/2)):
        g = chi_sqr_gradient(Img, bins, filter_bank[2*N],filter_bank[2*N+1])
        gradVar = np.dstack((gradVar,g))
    mean = np.mean(gradVar,axis =2)
    return mean

#Define half disk filters for gradient calculation
def half_disc(radius):
    """
    Creates half-disc masks for gradient computation.
    Parameters:
        radius (int): Radius of the half-disc.
    Returns:
        tuple: Two binary masks (a, b) representing half-discs.
    """
    a=np.ones((2*radius+1,2*radius+1))
    y,x = np.ogrid[-radius:radius+1,-radius:radius+1]
    mask2 = x*x + y*y <= radius**2
    a[mask2] = 0
    b=np.ones((2*radius+1,2*radius+1))
    y,x = np.ogrid[-radius:radius+1,-radius:radius+1]
    p = x>-1
    q = y>-radius-1
    mask3 = p&q
    b[mask3] = 0
    return a, b

def disc_masks(sigma, orients):
    """
    Generates a set of rotated half-disc masks.
    Parameters:
        sigma (list): List of radii for the half-discs.
        orients (int): Number of orientations for the half-discs.
    Returns:
        list: A list of rotated half-disc masks.
    """    
    flt = []
    angles = np.linspace(0,360,orients, endpoint=False)
    for rad in sigma:
        a,b = half_disc(radius = rad)
        for ang in angles:
            c1 = skimage.transform.rotate(b,ang,cval =1, mode='constant')
            z1 = np.logical_or(a,c1).astype(np.int32)
    
            b2 = np.flip(b,1)
            c2 = skimage.transform.rotate(b2,ang,cval =1, mode='constant')
            z2 = np.logical_or(a,c2).astype(np.int32)
            flt.append(z1)
            flt.append(z2)
    # for each in flt:
    #     plt.imshow(each,cmap='binary')
    #     plt.show()
    return flt

def plot_LM(filters, save_dir):
    """
    Visualizes and saves the LM filter bank.
    Parameters:
        filters (np.ndarray): A 3D array representing the LM filter bank.
        save_dir (str): Directory to save the visualization.
    """
    _,_,r = filters.shape
    os.makedirs(save_dir, exist_ok=True)
    num_cols=12 
    num_rows = math.ceil(r/num_cols)
    plt.subplots(num_rows,num_cols,figsize=(20,20))
    for i in range(r):
        plt.subplot(num_rows,num_cols,i+1)
        plt.axis('off')
        plt.imshow(filters[:,:,i],cmap='binary')
    save_path=os.path.join(save_dir, 'LM.png')
    plt.savefig(save_path)
    plt.close()
        # x = filters[:,:,i]
        # border = cv2.copyMakeBorder(x,10,10,10,10,cv2.BORDER_CONSTANT,value = [255,255,255])
    #     # fig = (border,) + fig
    # return cv2.hconcat(fig)

def plot_Gab(filters, save_dir):
    """
    Visualizes and saves the Gabor filter bank.
    Parameters:
        filters (list): List of Gabor filter kernels.
        save_dir (str): Directory to save the visualization.
    """
    r = len(filters)
    num_rows = math.ceil(r / 5)
    os.makedirs(save_dir, exist_ok=True)
    plt.subplots(num_rows, 5, figsize=(20, 20))
    for i in range(r):
        plt.subplot(num_rows, 5, i+1)
        plt.axis('off')
        plt.imshow(filters[i], cmap='gray')
    save_path = os.path.join(save_dir, 'Gabor.png')
    plt.savefig(save_path)
    plt.close()

def plot_DoG(filters, save_dir):
    """
    Visualizes and saves the DoG filter bank.
    Parameters:
        filters (list): List of DoG filter kernels.
        save_dir (str): Directory to save the visualization.
    """
    r = len(filters)
    num_rows = math.ceil(r / 5)
    os.makedirs(save_dir, exist_ok=True)
    plt.subplots(num_rows, 5, figsize=(20, 20))
    for i in range(r):
        plt.subplot(num_rows, 5, i+1)
        plt.axis('off')
        plt.imshow(filters[i], cmap='gray')
    save_path = os.path.join(save_dir, 'DoG.png')
    plt.savefig(save_path)
    plt.close()

def plot_halfdiscs(filters, save_dir):
    """
    Visualizes and saves the half-disc masks.
    Parameters:
        filters (list): List of half-disc masks.
        save_dir (str): Directory to save the visualization.
    """
    r = len(filters)
    num_rows = math.ceil(r / 5)
    os.makedirs(save_dir, exist_ok=True)
    plt.subplots(num_rows, 5, figsize=(20, 20))
    for i in range(r):
        plt.subplot(num_rows, 5, i+1)
        plt.axis('off')
        plt.imshow(filters[i], cmap='binary')
    save_path = os.path.join(save_dir, 'Half-Discs.png')
    plt.savefig(save_path)
    plt.close()

def main():
    """
    Main function for the Pb-lite edge detection pipeline.
    Generates filter banks, texture maps, gradients, and final edge-detected images.
    """

    # Parse command-line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--Maps_flag', default=True)
    Args = Parser.parse_args()
    Maps_flag = Args.Maps_flag

    # Set up folder structure
    base_dir = 'results'
    base_folder(base_dir)   # Create base directory
    filterbanks_folder(base_dir)    # Create folder for filter banks
    maps_dir = maps_folder(base_dir)    # Create folder for maps
    edges_dir= edges_folder(base_dir)   # Create folder for edges

    # Process all the images
    for i in range(10):
        img_name = f"img{i+1}"
        image_folder = os.path.join(edges_dir, img_name)
        os.makedirs(image_folder, exist_ok=True)    # Create subfolder for current image

        path = 'Datasets/BSDS500/Images/'+str(i+1)+'.jpg'
        print(f"Processing: {path}")
        img = plt.imread(path)  #0 for reading img in grayscale
        img_col = plt.imread(path)  # for reading img in color
        # img = cv2.cvtColor(img,)

        # Generate filter banks and save visualizations
        """
        Generate Leung-Malik Filter Bank: (LM)
        Display all the filters in this filter bank and save image as LM.png,
        use command "cv2.imwrite(...)"
        """
        filter_bank1 = LM_filters(kernelsize = 49, sigma = 3, num_orientations = 6, nrotinv = 12)
        plot_LM(filter_bank1, os.path.join(base_dir, "filterbanks"))
        # cv2.imwrite('LM.png',flt1)

        """
        Generate Difference of Gaussian Filter Bank: (DoG)
        Display all the filters in this filter bank and save image as DoG.png,
        use command "cv2.imwrite(...)"
        """
        filter_bank2 = Oriented_DOG(sigma = [5,8,11],orient = 15, size = 49 )
        plot_DoG(filter_bank2, os.path.join(base_dir, "filterbanks"))
        # cv2.imwrite('DoG.png',flt2)

        """
        Generate Gabor Filter Bank: (Gabor)
        Display all the filters in this filter bank and save image as Gabor.png,
        use command "cv2.imwrite(...)"
        """
        filter_bank3 = Gabor_filter(sigma=[5,8,11], theta=0, Lambda=4.0, psi=0, gamma=0.70,num_filters=15)
        plot_Gab(filter_bank3, os.path.join(base_dir, "filterbanks"))

        if(Maps_flag):
            """
            Generate texture ID's using K-means clustering
            Display texton map and save image as TextonMap_ImageName.png,
            use command "cv2.imwrite('...)"
            """
            T = Texton(img_col,filter_bank1,filter_bank2,filter_bank3,num_clusters=64)
            np.save(os.path.join(maps_dir, f"T{i+1}.npy"),T)
            plt.imsave(os.path.join(image_folder,f"TextonMap_{i+1}.png"), T)

            print(f"Generated Texture Map for image_{i+1}")

            """
            Generate Brightness Map
            Perform brightness binning
            """
            B = Brightness(Img = img, num_clusters=16)
            np.save(os.path.join(maps_dir, f"B{i+1}.npy"),B)
            plt.imsave(os.path.join(image_folder, f"BrightnessMap_{i+1}.png"), B,cmap='binary')

            print(f"Generated Brightness Map for image_{i+1}")

            """
            Generate Color Map
            Perform color binning or clustering
            """
            C = Color(img_col, 16)
            np.save(os.path.join(maps_dir, f"C{i+1}.npy"),C)
            plt.imsave(os.path.join(image_folder, f"ColorMap_{i+1}.png"), C)

            print(f"Generated Color Map for image_{i+1}")
        else:
            # Load precomputed maps
            T = np.load(os.path.join(maps_dir, f"T{i+1}.npy"))
            B = np.load(os.path.join(maps_dir, f"B{i+1}.npy"))
            C = np.load(os.path.join(maps_dir, f"B{i+1}.npy"))

        """
        Generate Half-disk masks
        Display all the Half-disk masks and save image as HDMasks.png,
        use command "cv2.imwrite(...)"
        """
        c = disc_masks([5,7,16], 8)
        plot_halfdiscs(c, os.path.join(base_dir, "filterbanks"))

        """
        Generate Texton Gradient (Tg)
        Perform Chi-square calculation on Texton Map
        Display Tg and save image as Tg_ImageName.png,
        use command "cv2.imwrite(...)"
        """
        Tg = Gradient(T, 64, c)
        plt.imsave(os.path.join(image_folder, f"Tg_{i+1}.png"), Tg)

        print(f"Generated Texton Gradient for image_{i+1}")

        """
        Generate Brightness Gradient (Bg)
        Perform Chi-square calculation on Brightness Map
        Display Bg and save image as Bg_ImageName.png,
        use command "cv2.imwrite(...)"
        """
        Bg = Gradient(B, 16, c)
        plt.imsave(os.path.join(image_folder, f"Bg_{i+1}.png"), Bg,cmap='binary')

        print(f"Generated Brightness Gradient for image_{i+1}")

        """
        Generate Color Gradient (Cg)
        Perform Chi-square calculation on Color Map
        Display Cg and save image as Cg_ImageName.png,
        use command "cv2.imwrite(...)"
        """
        Cg = Gradient(C, 16, c)
        plt.imsave(os.path.join(image_folder, f"Cg_{i+1}.png"), Cg)

        print(f"Generated Color Gradient for image_{i+1}")
        # cv2.imshow('temp',temp)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        print("Generating PbLite output...")
        
        """
        Read Sobel Baseline
        use command "cv2.imread(...)"
        """
        sobelBaseline = plt.imread(f'Datasets/BSDS500/SobelBaseline/{i+1}.png',0)

        """
        Read Canny Baseline
        use command "cv2.imread(...)"
        """
        cannyBaseline = plt.imread(f'Datasets/BSDS500/CannyBaseline/{i+1}.png',0)

        """
        Combine responses to get pb-lite output
        Display PbLite and save image as PbLite_ImageName.png
        use command "cv2.imwrite(...)"
        """
        alpha, beta, gamma = 0.33, 0.33, 0.34
        temp = alpha*Tg + beta*Bg + gamma*Cg

        pblite_out = temp * (0.50*cannyBaseline+0.50*sobelBaseline)

        plt.imsave(os.path.join(image_folder, f"PbLite_output_{i+1}.png"), pblite_out, cmap="gray")
        print(f"pb_lite output for image_{i+1} Generated")

    # plt.imshow(pblite_out,cmap='binary')
    # plt.show()
    print("Processing complete!")

if __name__ == '__main__':
    main()