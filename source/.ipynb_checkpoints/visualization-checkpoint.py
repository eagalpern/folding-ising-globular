import numpy as np
import pandas as pd
import seaborn as sns
import py3Dmol
from matplotlib import pyplot as plt, colors
from matplotlib.colors import Normalize,BoundaryNorm,rgb2hex
from matplotlib import cm, colormaps
from matplotlib.ticker import MaxNLocator
import matplotlib.patches as mpatch
from matplotlib.collections import PatchCollection
import matplotlib.path as mpath
import matplotlib
import scipy.signal as scs

def rand_cmap(nlabels, type='bright', first_color_black=True, last_color_black=False,
              start_with_cmap=True, verbose=True):
    """
    Creates a random colormap to be used together with matplotlib. Useful for segmentation tasks
    :param nlabels: Number of labels (size of colormap)
    :param type: 'bright' for strong colors, 'soft' for pastel colors
    :param first_color_black: Option to use first color as black, True or False
    :param last_color_black: Option to use last color as black, True or False
    :param verbose: Prints the number of labels and shows the colormap. True or False
    :return: colormap for matplotlib
    """
    from matplotlib.colors import LinearSegmentedColormap
    import colorsys
    import numpy as np


    if type not in ('bright', 'soft'):
        print ('Please choose "bright" or "soft" for type')
        return

    if verbose:
        print('Number of labels: ' + str(nlabels))

    # Generate color map for bright colors, based on hsv
    if type == 'bright':
        randHSVcolors = [(np.random.uniform(low=0.0, high=1),
                          np.random.uniform(low=0.2, high=1),
                          np.random.uniform(low=0.9, high=1)) for i in range(nlabels)]

        # Convert HSV list to RGB
        randRGBcolors = []
        for HSVcolor in randHSVcolors:
            randRGBcolors.append(colorsys.hsv_to_rgb(HSVcolor[0], HSVcolor[1], HSVcolor[2]))

        if start_with_cmap:
            c_=sns.color_palette()
            randRGBcolors[:len(c_)]=c_

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]



        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    # Generate soft pastel colors, by limiting the RGB spectrum
    if type == 'soft':
        low = 0.6
        high = 0.95
        randRGBcolors = [(np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high)) for i in range(nlabels)]

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]
        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    # Display colorbar
    if verbose:
        from matplotlib import colors, colorbar
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(15, 0.5))

        bounds = np.linspace(0, nlabels, nlabels + 1)
        norm = colors.BoundaryNorm(bounds, nlabels)

        cb = colorbar.ColorbarBase(ax, cmap=random_colormap, norm=norm, spacing='proportional', ticks=None,
                                   boundaries=bounds, format='%1i', orientation=u'horizontal')

    return random_colormap,randRGBcolors


def view_3d_exon_hist(ali_seq,colors,pdb_filename):
    

    view = py3Dmol.view(js='https://3dmol.org/build/3Dmol.js')
    
    view.clear()
    view.addModel(open(pdb_filename,'r').read(),'pdb')
    
    #else:
    #view = py3Dmol.view(query='pdb:'+pdb,width=800, height=600)

    view.setBackgroundColor('white')

    view.setStyle({'cartoon':{'color':'white'}})
    view.setStyle({'chain':'B'},{'opacity':0 })
    view.setStyle({'chain':'C'},{'opacity':0 })
    view.setStyle({'chain':'D'},{'opacity':0 })
    view.setStyle({'chain':'E'},{'opacity':0 })
    view.setStyle({'chain':'F'},{'opacity':0 })
    view.setStyle({'chain':'G'},{'opacity':0 })
    view.setStyle({'chain':'O'},{'opacity':0 })


    #change residue color
    for i,res in enumerate(ali_seq):
        if res>0: #gaps are res=-1 and we dont want them
            view.addStyle({'chain':'A','resi':[str(res)]},{'cartoon':{'color':colors[i]}})
        #view.addStyle({'chain':'A','resi':[str(alinum.no[e_])]},{'cartoon':{'color':'white'}})
    view.zoomTo(viewer=(0,0))
    return view




def find_common_bs(exon_freq,order,thresh,border,npos):
    final_bs=scs.argrelmax(exon_freq,order=order)[0]
    final_bs=final_bs[exon_freq[final_bs]>thresh]
    final_bs=final_bs[final_bs>=border]
    final_bs=final_bs[final_bs<=(npos-border)]
    final_bs=np.hstack([0,final_bs,npos])
    return final_bs

###### SECONDARY STRUCTURE VISUALIZATION #######

def coords2path(coord_set1):

    coords_f1 = []
    instructions1 = []

    for c in coord_set1:
        for n in range(len(c)):
            coords_f1.append(c[n])
            if n == 0:
                instructions1.append(1)
            else:
                instructions1.append(2)

    return coords_f1, instructions1

def NormalizeData(data):
    if np.min(data) == np.max(data):
        warnings.warn("Warning: scores are the same for all residues")
        return data
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def build_loop(ax,loop,idx,ssidx,loop_coords,linelen,nlines,prev_ss,next_ss,
               z=1,clr='r',mat=0,size=75):

    # if loop is smaller than 3 residues and has gaps on both sides, don't draw
    if prev_ss == "B" and next_ss == "B" and loop[1]-loop[0] < 2:
        return

    i0 = loop[0]
    if loop[0] != 0 and prev_ss != "B":
        i0 = loop[0]-1
    elif i0 == 0:
        i0 = 0.06
    else:
        i0 = loop[0]
    i1 = loop[1]+2
    if loop[1] == linelen-1:
        i1 += 2

    o = 2
    if idx == nlines-1:
        o = 0
    if next_ss == "B":
        o = -1.5
    if next_ss == None:
        o=-4.1

    rectangle = mpatch.Rectangle((i0/6.0,-0.25-5.5*idx+2.0*ssidx),
                                 (i1-i0+o)/6.0,0.5,
                                  fc=clr,ec='k',zorder=0)
    


    ax.add_patch(rectangle)

    xy = rectangle.get_xy()
    w = rectangle.get_width()
    h = rectangle.get_height()
    loop_coords.append(np.array([[xy[0],xy[1]],
                                 [xy[0],xy[1]+h],
                                 [xy[0]+w,xy[1]+h],
                                 [xy[0]+w,xy[1]],
                                 [xy[0],xy[1]]]))

def build_strand(ax,strand,idx,ssidx,strand_coords,next_ss,z=1,clr='r',
                 imagemat=0,size=75):

    delta = 0 if next_ss == None else 1
    arrow=mpatch.FancyArrow(((strand[0]+delta-1)/6.0),-5.5*idx+2.0*ssidx,
                            (strand[1]-strand[0]+1)/6.0,0,
                            width=1.0,fc=clr,linewidth=0.5,ec='k',
                            zorder=z,head_width=2.0,
                            length_includes_head=True,head_length=2.0/6.0)

    


    ax.add_patch(arrow)

    strand_coords.append(arrow.get_xy())


def build_helix(ax,helix,idx,ssidx,coord_set1, coord_set2, clr='r',size=37.5,
                z=1,bkg=(0.195,0,0.051),imagemat=0):

    i = helix
    l = i[1]-i[0]+1
    points = [[i[0]/6.0,-0.25-5.5*idx+2.0*ssidx],
              [i[0]/6.0+1.0/6,0.75-5.5*idx+2.0*ssidx],\
              [i[0]/6.0+2.0/6,0.75-5.5*idx+2.0*ssidx],
              [i[0]/6.0+1.0/6,-0.25-5.5*idx+2.0*ssidx]]
    #hlx = plt.Polygon(points,fc=clr,ec='k',zorder=1,linewidth=2)
    #coords= hlx.get_xy()
    coord_set2.append(points+[points[0]])

    for j in range((l-2)-1):
        if j % 2 == 0:
            points = [[i[0]/6.0+(1.0+j)/6,0.75-5.5*idx+2.0*ssidx],
                      [i[0]/6.0+(2.0+j)/6,0.75-5.5*idx+2.0*ssidx],
                      [i[0]/6.0+(3.0+j)/6,-0.75-5.5*idx+2.0*ssidx],
                      [i[0]/6.0+(2.0+j)/6,-0.75-5.5*idx+2.0*ssidx]]
            coord_set1.append(points+[points[0]])
            #hlx = mpatch.Polygon(points,fc=bkg,zorder=0)

        else:
            points = [[i[0]/6.0+(1.0+j)/6,-0.75-5.5*idx+2.0*ssidx],
                      [i[0]/6.0+(2.0+j)/6,-0.75-5.5*idx+2.0*ssidx],
                      [i[0]/6.0+(3.0+j)/6,0.75-5.5*idx+2.0*ssidx],
                      [i[0]/6.0+(2.0+j)/6,0.75-5.5*idx+2.0*ssidx]]
            coord_set2.append(points+[points[0]])
            #hlx = mpatch.Polygon(points,fc=clr,zorder=z)


    if (l-2-1)%2 == 1:

        points = [[i[1]/6.0-1.0/6,-0.75-5.5*idx+2.0*ssidx],\
                  [i[1]/6.0,-0.75-5.5*idx+2.0*ssidx],\
                  [i[1]/6.0+1.0/6,0.25-5.5*idx+2.0*ssidx],
                  [i[1]/6.0,0.25-5.5*idx+2.0*ssidx]]

        coord_set2.append(points+[points[0]])
        #hlx = mpatch.Polygon(points,fc=clr,zorder=0)

    else:
        points = [[i[1]/6.0-1.0/6,0.75-5.5*idx+2.0*ssidx],
                  [i[1]/6.0,0.75-5.5*idx+2.0*ssidx],\
                  [i[1]/6.0+1.0/6,-0.25-5.5*idx+2.0*ssidx],
                  [i[1]/6.0,-0.25-5.5*idx+2.0*ssidx]]
        coord_set1.append(points+[points[0]])

        #hlx = plt.Polygon(points,fc=bkg,zorder=10)

def SS_breakdown(ss):

    i = 0
    curSS = ''
    jstart = -1
    jend = -1

    strand = []
    loop = []
    helix = []
    ssbreak = []

    ss_order = []
    ss_bounds = []

    last_ss = ''

    SS_equivalencies = {'H':['H'],
                        '-':['-'],
                        'S':[' ','S','C','T','G','I','P'],
                        ' ':[' ','S','C','T','G','I','P'],
                        'C':[' ','S','C','T','G','I','P'],
                        'T':[' ','S','C','T','G','I','P'],
                        'G':[' ','S','C','T','G','I','P'],
                        'I':[' ','S','C','T','G','I','P'],
                        'P':[' ','S','C','T','G','I','P'],
                        'E':['E','B'],
                        'B':['E','B']}

    cur_SSDict = {'H':'helix',
                  '-':'break',
                  'E':'strand',
                  'B':'strand'}

    for i in range(len(ss)):

        if i == 0:
            curSS = SS_equivalencies[ss[i]]
            jstart = i
            if ss[i] in cur_SSDict.keys():
                last_ss = cur_SSDict[ss[i]]
            else:
                last_ss = 'loop'
            continue

        if ss[i] in curSS:
            jend = i

        if ss[i] not in curSS or i == len(ss)-1:
            if 'E' in curSS and jend-jstart+1 >= 3:
                strand.append((jstart,jend))
                ss_bounds.append((jstart,jend))
                ss_order.append('E')
                last_ss = 'strand'
            elif 'H' in curSS and jend-jstart+1 >=4:
                helix.append((jstart,jend))
                ss_bounds.append((jstart,jend))
                ss_order.append('H')
                last_ss = 'helix'
            elif ' ' in curSS and last_ss !='loop':
                if jend < jstart:
                    jend = jstart
                loop.append((jstart,jend))
                ss_bounds.append((jstart,jend))
                ss_order.append('L')
                last_ss = 'loop'
            elif '-' in curSS:
                if jend < jstart:
                    jend = jstart
                ssbreak.append((jstart,jend))
                ss_bounds.append((jstart,jend))
                ss_order.append('B')
                last_ss = 'break'
            elif last_ss == 'loop':
                if jend < jstart:
                    jend = jstart
                if len(loop) > 0:
                    jstart = loop[-1][0]
                    loop = loop[0:-1]
                    ss_bounds = ss_bounds[0:-1]
                    ss_order = ss_order[0:-1]
                loop.append((jstart,jend))
                ss_bounds.append((jstart,jend))
                ss_order.append('L')
                last_ss = 'loop'
            else:
                if jend < jstart:
                    jend = jstart
                loop.append((jstart,jend))
                ss_bounds.append((jstart,jend))
                ss_order.append('L')
                last_ss = 'loop'


            jstart = i
            curSS = SS_equivalencies[ss[i]]

    return strand,loop,helix, ssbreak, ss_order, ss_bounds



def plot_coords(ax,coords_all,colors,size):

    for i,coords in enumerate(coords_all):

        if not coords:
            continue

        coords_f1, instructions1 = coords2path(coords)

        #If loop or bottom helix layer, zorder = 0
        if i in [0,1]:
            z = 0
        else:
            z = 10

        path = mpath.Path(np.array(coords_f1),np.array(instructions1))
        patch = mpatch.PathPatch(path, facecolor='none',zorder=z)
        ax.add_patch(patch)
        
        x_range = [0, size]   
        im = ax.imshow([colors], extent=[min(x_range), max(x_range), 0.5, 3.0], aspect='auto', interpolation='none', zorder=z)
        im.set_clip_path(patch)


def generate_exon_colors(colors_,dssp_data):
    new_colors=np.array(colors_+[(1.0, 1.0, 1.0)])
    return new_colors[list(dssp_data.exon.values)]


def plot_ss(dssp_data,colors_,ax):
    ss=dssp_data.secondary_structure.values
    ss[ss==['-']]='T'
    exon_colors=generate_exon_colors(colors_,dssp_data)
    strand,loop,helix,ss_break,ss_order,ss_bounds =SS_breakdown(ss)

    sz = 0
    c = 'none'
    bc = 'none'
    nlines = 1

    factor=6.0

    #Parse color and scoring args
    #CMAP, bvals = parse_color(args,seq_wgaps,pdbseq,bfactors,msa,extra_gaps)

    bvals = [i for i in range(len(dssp_data))]

    mat = np.tile(NormalizeData(bvals), (100,1))
    #set sizes of SS chunks
    ss_prev = 0

    for i in range(len(ss_order)):

        if ss_order[i] == 'H':
            ss_prev = ss_bounds[i][1]/6.0+1/6.0
        else:
            ss_prev = ss_bounds[i][1]/6.0

    if ss_order[-1] == 'H':
        sz = ss_bounds[-1][1]/6.0+1/6.0
    elif ss_order[-1] in ['E','B']:
        sz = ss_bounds[-1][1]/6.0
    elif ss_order[-1] == 'L':
        sz = (ss_bounds[-1][1])/6.0   

    #Plot secondary structure chunks
    strand_coords = []
    loop_coords = []
    helix_coords1 = []
    helix_coords2 = []


    #fig, ax = plt.subplots(ncols=1, figsize=(50,30))


    for i in range(len(ss_order)):
        prev_ss = None
        next_ss = None
        if i != 0:
            prev_ss = ss_order[i-1]
        if i != len(ss_order)-1:
            next_ss = ss_order[i+1]

        if ss_order[i] == 'L':
            build_loop(ax,ss_bounds[i],0,1,loop_coords,len(dssp_data),1,prev_ss,next_ss,z=0,clr=c,mat=mat,size=sz)
        elif ss_order[i] == 'H':
            build_helix(ax,ss_bounds[i],0,1,helix_coords1,helix_coords2,z=i,clr=c,bkg=bc,imagemat=mat,size=sz)
        elif ss_order[i] == 'E':
            build_strand(ax,ss_bounds[i],0,1,strand_coords,next_ss,z=i,clr=c,imagemat=mat,size=sz)
    plot_coords(ax,[loop_coords,helix_coords2,strand_coords,helix_coords1],colors=exon_colors,size=len(dssp_data)/factor)

    ax.set_ylim([0.5,3])

#############
def plot_ss_(dssp_data,exon_colors,ax):
    ss=dssp_data.secondary_structure.values
    ss[ss==['-']]='T'
    #exon_colors=generate_exon_colors(colors_,dssp_data)
    strand,loop,helix,ss_break,ss_order,ss_bounds =SS_breakdown(ss)

    sz = 0
    c = 'none'
    bc = 'none'
    nlines = 1

    factor=6.0

    #Parse color and scoring args
    #CMAP, bvals = parse_color(args,seq_wgaps,pdbseq,bfactors,msa,extra_gaps)

    bvals = [i for i in range(len(dssp_data))]

    mat = np.tile(NormalizeData(bvals), (100,1))
    #set sizes of SS chunks
    ss_prev = 0

    for i in range(len(ss_order)):

        if ss_order[i] == 'H':
            ss_prev = ss_bounds[i][1]/6.0+1/6.0
        else:
            ss_prev = ss_bounds[i][1]/6.0

    if ss_order[-1] == 'H':
        sz = ss_bounds[-1][1]/6.0+1/6.0
    elif ss_order[-1] in ['E','B']:
        sz = ss_bounds[-1][1]/6.0
    elif ss_order[-1] == 'L':
        sz = (ss_bounds[-1][1])/6.0   

    #Plot secondary structure chunks
    strand_coords = []
    loop_coords = []
    helix_coords1 = []
    helix_coords2 = []

    #fig, ax = plt.subplots(ncols=1, figsize=(50,30))

    for i in range(len(ss_order)):
        prev_ss = None
        next_ss = None
        if i != 0:
            prev_ss = ss_order[i-1]
        if i != len(ss_order)-1:
            next_ss = ss_order[i+1]

        if ss_order[i] == 'L':
            build_loop(ax,ss_bounds[i],0,1,loop_coords,len(dssp_data),1,prev_ss,next_ss,z=0,clr=c,mat=mat,size=sz)
        elif ss_order[i] == 'H':
            build_helix(ax,ss_bounds[i],0,1,helix_coords1,helix_coords2,z=i,clr=c,bkg=bc,imagemat=mat,size=sz)
        elif ss_order[i] == 'E':
            build_strand(ax,ss_bounds[i],0,1,strand_coords,next_ss,z=i,clr=c,imagemat=mat,size=sz)
    plot_coords(ax,[loop_coords,helix_coords2,strand_coords,helix_coords1],
                colors=exon_colors,size=len(dssp_data)/factor)

    ax.set_ylim([0.5,3])
    
def map_t_seq_3d(t_, breaks, seq_len, vmin=250, vmax=450, cmap_='viridis', rgb=False):
    if len(t_) == len(breaks):
        cmap = colormaps.get_cmap(cmap_)
        norm = Normalize(vmin, vmax)
        rgba_values = cmap(norm(t_))
        units_len = np.concatenate([breaks[1:] - breaks[:-1], np.array([seq_len - breaks[-1]])])
        colors_ = []
        temperatures_ = []

        for r, rgba in enumerate(rgba_values):
            if rgb:
                rgb_values = [int(255 * v) for v in rgba[:3]]
                colors_ += [rgb_values] * units_len[r]
            else:
                colors_ += [rgba[:3]] * units_len[r]

            temperatures_ += [t_[r]] * units_len[r]



        return np.array(temperatures_), np.array(colors_)  # Return values from the function
    else:
        print('Error')
        return None, None  # Return None if there's an error