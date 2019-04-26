import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as spim
import os
import skimage as sk
from skimage import io

def parse_vgi(path=None):
    volhandle = False
    slice_paths = []
    for file in os.listdir(path):
        if file.endswith('.vgi'):
            with open(path+file) as f:
                searchlines = f.readlines()
                for line in searchlines:
                    if line.startswith('size'):
                        SZ = line.split()[-3:]
                        nx = np.int(SZ[0])
                        ny = np.int(SZ[1])
                        nz = np.int(SZ[2])
                    if line.startswith('resolution'):
                        voxel_size = float(line.split()[-1])
        if file.endswith('.vol'):
            volhandle = file
        if file.endswith('tif'):
            slice_paths.append(file)
    slice_paths.sort()
    if not volhandle:
        im = np.zeros((ny,nx,nz),dtype=np.float32)
        for i,p in enumerate(slice_paths):
            im[:,:,i] = io.imread(path+p)
    else:
        im = np.zeros((nx,ny,nz),dtype=np.float32)
        im = np.fromfile(path+volhandle,dtype=np.float32,count=-1)
        im = np.reshape(im,(nx,ny,nz),order='F')
    return nx, ny, nz, im, voxel_size


def min_max_norm_2D(im):
    im = im-im.min()
    im = im/im.max()
    return im


def roi_casing_2D(im, threshold=0.2,  buffer=10):
    c = sk.measure.find_contours(im,threshold)
    c.sort(key=len,reverse=True)
    c = c[0]
    xmax = np.int(np.ceil(c[:,0].max())+buffer)
    xmin = np.int(np.floor(c[:,0].min())-buffer)
    ymax = np.int(np.ceil(c[:,1].max())+buffer)
    ymin = np.int(np.floor(c[:,1].min())-buffer)
    im = im[xmin:xmax,ymin:ymax]
    return im,xmin,xmax,ymin,ymax


def find_casing_centre_2D(im, threshold=0.2, buffer=3, step=0.1, decimals=1, plot=False):
    c = sk.measure.find_contours(im,threshold)
    c.sort(key=len,reverse=True)
    c = c[0]
    ox = (im.shape[0]-1)/2
    oy = (im.shape[1]-1)/2
    rx = np.arange(ox-buffer,ox+buffer+step/2,step)
    ry = np.arange(oy-buffer,oy+buffer+step/2,step)
    results = np.zeros((len(rx),len(ry)))
    for i, xoff in enumerate(rx):
        for j, yoff in enumerate(ry):
            r,phi = cart2pol(c[:,1],c[:,0],origin=[xoff,yoff])
            results[i,j] = r.max()-r.min()
    i,j = (results == results.min()).nonzero()
    
    
    oy = np.float(np.round(ry[j],decimals=decimals))
    ox = np.float(np.round(rx[i],decimals=decimals))
    if plot:
        fig, (ax0,ax1) = plt.subplots(ncols=2, figsize=(10,5))
        im0 = ax0.imshow(results)
        fig.colorbar(im0, ax=ax0, shrink=0.75)
        r,phi = cart2pol(c[:,1],c[:,0],origin=[ox,oy])
        ax1.plot(phi,r,'.k')
    return ox,oy,c


def find_coil_2D(im, ox, oy, threshold=0, section=0):
    c = sk.measure.find_contours(im,threshold)
    c.sort(key=len, reverse=True)
    c = c[section]
    r,phi = cart2pol(c[:,1],c[:,0],origin=[ox,oy])
    return r,phi,c


def unsharp_mask(im, sigma=2, gain=100):
    fgau = spim.filters.gaussian_filter(im,sigma=sigma)
    fminus = im-fgau
    final = im+fminus*gain
    return final


def cart2pol(x, y, origin=[0,0]):
    o = origin
    rho = np.sqrt((x-o[0])**2+(y-o[1])**2)
    phi = np.arctan2((y-o[1]),(x-o[0]))
    phi = np.rad2deg(phi)
    return(rho, phi)


def gaussian_diff(im, sigma1=0.3, sigma2=2, gain=100):
    fg0 = spim.filters.gaussian_filter(im,sigma=sigma1)
    fg1 = spim.filters.gaussian_filter(im,sigma=sigma2)
    fminus = fg0-fg1
    final = fminus*gain
    return final


def parse_pipe(im,pipe,display=False):
    ox = None
    oy = None
    switch = False
    r = None
    phi = None
    c = None
    xmin = None
    xmax = None
    ymin = None
    ymax = None
    c_casing = None
    
    for p in pipe:
        if p == 'slice':
            im = im[:,:,pipe['slice']]
        if p == 'min_max_norm_2D':
            im = min_max_norm_2D(im)
        if p == 'roi_casing_2D':
            im,xmin,xmax,ymin,ymax = roi_casing_2D(im,
                                                   threshold=pipe[p]['threshold'],
                                                   buffer=pipe[p]['buffer'])
        if p == 'median_filter':
            im = spim.filters.median_filter(im,
                                            footprint=pipe[p]['footprint'])
        if p == 'unsharp_mask':
            im = unsharp_mask(im,
                              sigma = pipe[p]['sigma'],
                              gain = pipe[p]['gain'])
        if p == 'gaussian_diff':
            im = gaussian_diff(im,
                               sigma1 = pipe[p]['sigma1'],
                               sigma2 = pipe[p]['sigma2'],
                               gain = pipe[p]['gain'])
        if p == 'find_casing_centre_2D':
            ox,oy,c_casing = find_casing_centre_2D(im,
                                                   threshold=pipe[p]['threshold'],
                                                   buffer=pipe[p]['buffer'],
                                                   step=pipe[p]['step'],
                                                   decimals=pipe[p]['decimals'])
        if p == 'find_coil_2D':
            if pipe[p]['threshold'] == 'mean':
                r,phi,c = find_coil_2D(im,
                                       ox,
                                       oy,
                                       threshold=im.mean())
            else:
                r,phi,c = find_coil_2D(im,
                                       ox,
                                       oy,
                                       threshold=pipe[p]['threshold'])
            switch = True
        if display:
            fig, ax0 = plt.subplots(figsize=(10,10))
            im0 = ax0.imshow(im, cmap='gray')
            ax0.set_title(p)
            fig.colorbar(im0, ax=ax0, shrink=0.85)
            if ox == None:
                pass
            else:
                ax0.plot(ox,oy,'r+')
                ax0.plot(c_casing[:,1],c_casing[:,0],'r',lw=1)
            if switch:
                ax0.plot(c[:,1],c[:,0],'b',lw=1)
    return im,r,phi,c,xmin,xmax,ymin,ymax,c_casing,ox,oy

