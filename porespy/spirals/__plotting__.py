import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import cm
import matplotlib as mpl
from __funcs__ import parse_vgi

def unload_data(path_tomo, path_data, path_gradient=False):
    nx,ny,nz,im,voxel_size = parse_vgi(path_tomo)
    data1 = np.load(path_data)
    if path_gradient:
        data2 = np.load(path_gradient)
        gradient = data2['gradient']
    else:
        gradient = []
    phi = data1['phi']
    radius = data1['radius']
    coil = data1['coil']
    xmin = data1['xmin']
    xmax = data1['xmax']
    ymin = data1['ymin']
    ymax = data1['ymax']
    casing = data1['casing']
    originx = data1['originx']
    originy = data1['originy']
    
    return im, voxel_size, phi, radius, coil, xmin, xmax, ymin, ymax, casing, originx, originy, gradient

def phi_radius_subset(phi, radius, rmin, voxel_size, dist=100, shift=180):
    phi0 = np.concatenate(phi[rmin:rmin+dist])+shift
    radius0 = np.concatenate(radius[rmin:rmin+dist])*voxel_size
    return phi0, radius0

def colored_plot(phi, radius, ylow=1.5, yhigh=9, gmin=0.1, gmax=0.3, save=False, path='plot.png',threshold=40, a_0=0.01):
    set_mpl_style()
    result_rad = []
    result_phi = []
    for j in range(360):
        test = np.sort(radius[(phi>j-0.5) & (phi<j+0.5)])
        count = 0
        c = []
        for i,val in enumerate(test[:-1]):
            a = test[i+1]-val
            if a < a_0:
                count += 1
                c.append(val)
            if a > a_0 or i == len(test)-2:
                if len(c) < threshold:
                    c = []
                else:
                    c_mean = np.array(c).mean()
                    c = []
                    result_rad.append(c_mean)
                    result_phi.append(j)
    result_rad = np.array(result_rad)
    result_phi = np.array(result_phi)
    gap = result_rad[1:]-result_rad[:-1]
    gap_phi = result_phi[1:][gap>0]
    start_height = result_rad[:-1][gap>0]
    gap = gap[gap>0]
    
    viridis = cm.get_cmap('viridis', 1024)
    fig, ax0, = plt.subplots(figsize=(12,20))
    im0 = ax0.scatter(result_phi,result_rad,marker='.',linewidth=1,alpha=1,cmap='viridis',s=0.5)
    for i in range(len(gap)):
        if gap[i] > gmax:
            col = 'xkcd:white'
        elif gap[i] < gmin:
            col = 'xkcd:white'
        else:
            col = viridis((gap[i]-gmin)/(gmax-gmin))
        ax0.add_patch(Rectangle((gap_phi[i]-0.5,start_height[i]),width=1,height=gap[i],linewidth=0,color=col))
    norm = mpl.colors.Normalize(vmin=gmin,vmax=gmax)
    sm = plt.cm.ScalarMappable(cmap=viridis, norm=norm)
    sm.set_array([])
    v = np.linspace(gmin, gmax, 10, endpoint=True)
    cb1 = fig.colorbar(sm, ticks=v)

    ax0.set_ylim(ylow,yhigh)
    ax0.set_xlim(0,360)

    rng = np.array(range(13))*30

    ax0.set_xticks(rng)
    ax0.tick_params(labelsize=20)
    ax0.set_xlabel('Angular Position')
    ax0.set_ylabel('Spiral Radius [mm]')

    if save:
        fig.savefig(path,dpi=300)
        plt.close(fig=fig)
    return