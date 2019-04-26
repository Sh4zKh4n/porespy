import matplotlib.pyplot as plt


def set_mpl_style(sfont=20,mfont=30,lfont=40):

    sfont = 20
    mfont = 30
    lfont = 40

    line_props = {'linewidth': 4,
                  'markersize': 10}
    
    font_props = {'size': sfont}
    
    axes_props = {'titlesize': lfont,
                  'labelsize': mfont,
                  'linewidth': 3,
                  'labelpad': 10,
                  'titlepad':10}
    
    xtick_props = {'labelsize': sfont,
                   'top': True,
                   'direction': 'in',
                   'major.size': 10,
                   'major.width': 3,
                   'major.pad': 5}
    
    ytick_props = {'labelsize': sfont,
                   'right': True,
                   'direction': 'in',
                   'major.size': 10,
                   'major.width': 3,
                   'major.pad': 5}
    
    legend_props = {'fontsize': mfont,
                    'frameon': False}
    
    figure_props = {'titlesize': sfont}

    plt.rc('font', **font_props)
    plt.rc('lines', **line_props)
    plt.rc('axes', **axes_props)
    plt.rc('xtick', **xtick_props)
    plt.rc('ytick', **ytick_props)
    plt.rc('legend', **legend_props)
    plt.rc('figure', **figure_props)
