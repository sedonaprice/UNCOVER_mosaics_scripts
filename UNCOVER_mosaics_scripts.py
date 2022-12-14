##################################################################
# UNCOVER_mosaics_scripts/UNCOVER_mosaics_scripts.py             #
#                                                                #
# Copyright 2022 Sedona Price <sedona.price@gmail.com>           #
# Licensed under a 3-clause BSD style license - see LICENSE.rst  #
##################################################################

import os, copy

import numpy as np


import matplotlib as mpl
try:
    mpl.use('Agg')
except:
    pass

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import matplotlib.patheffects as path_effects
from matplotlib.path import Path
from matplotlib.patches import PathPatch

from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker

from astropy.io import fits
from astropy.wcs import WCS

from astropy.visualization import make_lupton_rgb


talk_fonts_dict = {'font.family':' Arial, Helvetica Neue, Helvetica,sans-serif',
                   'mathtext.fontset': 'dejavusans',
                   'text.usetex': False ,
                   'axes.linewidth': 1,
                   'xtick.major.width': 1,
                   'xtick.minor.width': 1,
                   'ytick.major.width': 1,
                   'ytick.minor.width': 1}
talk_font_label_dict = {'fontweight': 550}


plt.rcParams.update(talk_fonts_dict)



def _get_dict_NIRCAM_primary(rgb_shape, noinsets=False):

    if not noinsets:
        # INCLUDES INSETS

        pad_factor = 0.4 
        crop_y_fac = [0.0, 0.875]

        crop_y_pix = [int(crop_y_fac[0]*rgb_shape[0]),
                    int(crop_y_fac[1]*rgb_shape[0])]

        crop_x_fac = [0.0, 0.75] 

        crop_x_pix = [int(crop_x_fac[0]*rgb_shape[1]),
                    int(crop_x_fac[1]*rgb_shape[1])]


        rgb_shape = ((crop_y_pix[1]-crop_y_pix[0]),
                    (crop_x_pix[1]-crop_x_pix[0]))


        xpad = int(np.ceil(rgb_shape[1]*pad_factor))
        rgb_big_shape = (rgb_shape[0], rgb_shape[1]+xpad, 3)

        dict_NIRCAM_primary = {'pad_factor': pad_factor,
                            'crop_x_pix': crop_x_pix,
                            'crop_y_pix': crop_y_pix,
                            'filters_fontsize': 10,
                            'w_logo': 0.25,
                            'loc_logo': [-0.025/(1.+pad_factor) * 2., 0.025],
                            'annotate_field_label': 'Abell 2744',
                            'annotate_field_loc': (0.51,0.035),
                            'annotate_field_fontsize': 16,
                            'annotate_field_fontweight': 'bold',
                            'annotate_field_ha': 'left',
                            'annotate_field_va': 'bottom',
                            'ruler_loc': (0.2,0.025),
                            'ruler_vertical': False,
                            'filterlabel_loc': 'center left',
                            'filterlabel_bboxtoanchor': (0.025/(1.+pad_factor), 0.975),
                            'compassRA': 3.685, 
                            'compassDEC': -30.41196,
                            'compass_len': 0.5,
                            'compass_units': 'arcmin',
        }

        dict_NIRCAM_primary['rgb_big_shape'] = rgb_big_shape

        dict_NIRCAM_primary['clu_dicts'] =  {
                    'Primary': {
                                'xpos': (1.-0.26)*rgb_big_shape[1] - 0.025*rgb_big_shape[0],
                                'ypos': 0.025*rgb_big_shape[0],
                                'width': 0.26 * rgb_big_shape[1], 
                                'height': 0.26 * rgb_big_shape[1], 
                                'label': 'primary cluster', 
                                'label_loc': (0.5, 1.03),
                                'label_va': 'bottom',
                                'label_ha': 'center',
                                'inset_wedge': { 'direction': 'right',
                                                'position_pix': [0.45*rgb_big_shape[1],
                                                            0.35*rgb_big_shape[0]],
                                                'position_radec': [3.5890613224392545,-30.399657801823526],
                                },
                                'cutout': {'xlims_pix': [0.41*rgb_big_shape[1],
                                                        0.41*rgb_big_shape[1]+0.085*rgb_big_shape[1]] ,
                                            'ylims_pix': [0.275*rgb_big_shape[0],
                                                        0.275*rgb_big_shape[0]+0.085*rgb_big_shape[1]],
                                            'corners_radec': [[3.599700721107946,-30.40989946601362],
                                                            [3.5771050501813324,-30.390410732982055]], # LL / UR
                                },
                                },
                    'N':       {
                                'xpos': 0.015*rgb_big_shape[1],
                                'ypos': 0.5*rgb_big_shape[0],
                                'width': 0.22 * rgb_big_shape[1],
                                'height': 0.22 * rgb_big_shape[1], 
                                'label': None, #'N core',
                                'inset_wedge': { 'direction': 'left',
                                                'position_pix': [0.49*rgb_big_shape[1],
                                                            0.67*rgb_big_shape[0]],
                                                'position_radec': [3.578427776750986,-30.35596683807201],
                                },
                                'cutout': {'xlims_pix': [0.455*rgb_big_shape[1],
                                                        0.455*rgb_big_shape[1]+0.065*rgb_big_shape[1]] ,
                                            'ylims_pix': [0.621*rgb_big_shape[0],
                                                        0.621*rgb_big_shape[0]+0.065*rgb_big_shape[1]],
                                            'corners_radec': [[3.5877317917809206,-30.3626667037864],
                                                            [3.570465904230491,-30.347754496723287]], # LL / UR
                                },
                                },
                    'NW':      {
                                'xpos': (1.-0.185)*rgb_big_shape[1] - 0.025*rgb_big_shape[0],
                                'ypos': (1.-0.025)*rgb_big_shape[0] - (0.185)*rgb_big_shape[1],
                                'width': 0.185 * rgb_big_shape[1], 
                                'height': 0.185 * rgb_big_shape[1], 
                                'label': None,  #'NW core',
                                'inset_wedge': { 'direction': 'right',
                                                'position_pix': [0.6*rgb_big_shape[1],
                                                            0.54*rgb_big_shape[0]],
                                                'position_radec': [3.5491804906615614,-30.375710889030846],
                                },
                                'cutout': {'xlims_pix': [0.55*rgb_big_shape[1],
                                                        0.55*rgb_big_shape[1]+0.11*rgb_big_shape[1]] ,
                                            'ylims_pix': [0.425*rgb_big_shape[0],
                                                        0.425*rgb_big_shape[0]+0.11*rgb_big_shape[1]],
                                            'corners_radec': [[3.562472513062995,-30.389419870323387],
                                                            [3.5332341394003977,-30.36417771733208]],  # LL / UR
                                },
                                },
                    }


    else:
        # NO INSETS

        pad_factor = 0.
        crop_y_fac = [0.,1.] 

        crop_y_pix = [int(crop_y_fac[0]*rgb_shape[0]),
                    int(crop_y_fac[1]*rgb_shape[0])]

        crop_x_fac = [0., 0.875]

        crop_x_pix = [int(crop_x_fac[0]*rgb_shape[1]),
                    int(crop_x_fac[1]*rgb_shape[1])]


        rgb_shape = ((crop_y_pix[1]-crop_y_pix[0]),
                    (crop_x_pix[1]-crop_x_pix[0]))

        xpad = int(np.ceil(rgb_shape[1]*pad_factor))
        rgb_big_shape = (rgb_shape[0], rgb_shape[1]+xpad, 3)

        dict_NIRCAM_primary = {'pad_factor': pad_factor,
                            'crop_x_pix': crop_x_pix,
                            'crop_y_pix': crop_y_pix,
                            'filters_fontsize': 10,
                            'w_logo': 0.25,
                            'loc_logo': [(1.-0.25*(rgb_big_shape[0]/rgb_big_shape[1]))-0.05, 0.025],
                            'annotate_field_label': 'Abell 2744',
                            'annotate_field_loc': (0.51,0.045), 
                            'annotate_field_fontsize': 16,
                            'annotate_field_fontweight': 'bold',
                            'annotate_field_ha': 'left',
                            'annotate_field_va': 'bottom',
                            'ruler_loc': (0.95, 0.3), 
                            'ruler_vertical': True,
                            'ruler_fontsize': 10., 
                            'filterlabel_loc': 'center left',
                            'filterlabel_bboxtoanchor': (0.025, 0.975), 
                            'compassRA': 3.6262468, 
                            'compassDEC': -30.327671, 
                            'compass_len': 0.5,
                            'compass_units': 'arcmin',
                            'compass_fontsize': 9.5, 
        }


        dict_NIRCAM_primary['rgb_big_shape'] = rgb_big_shape


        dict_NIRCAM_primary['clu_dicts'] = None

    return dict_NIRCAM_primary




def _get_dict_NIRISS_parallel(rgb_shape):

    pad_factor = 0.1 

    crop_y_fac = [0.175, 0.875]

    crop_y_pix = [int(crop_y_fac[0]*rgb_shape[0]),
                  int(crop_y_fac[1]*rgb_shape[0])]

    crop_x_fac = [0.585, 1.]

    crop_x_pix = [int(crop_x_fac[0]*rgb_shape[1]),
                  int(crop_x_fac[1]*rgb_shape[1])]


    rgb_shape = ((crop_y_pix[1]-crop_y_pix[0]),
                 (crop_x_pix[1]-crop_x_pix[0]))

    dict_NIRISS_parallel = {'pad_factor': pad_factor,
                           'crop_x_pix': crop_x_pix,
                           'crop_y_pix': crop_y_pix,
                           'filters_fontsize': 12,
                           'w_logo': 0.285,
                           'loc_logo': [0.04, 0.025], 
                           'annotate_field_label': 'NIRISS Parallel',
                           'annotate_field_loc': (1.-0.035,0.085),
                           'annotate_field_ha': 'right',
                           'annotate_field_va': 'bottom',
                           'annotate_field_fontsize': 16,
                           'annotate_field_fontweight': 'bold',
                           'annotate_field_label2': 'Overlaps HFF',
                           'annotate_field_loc2': (1.-0.035,0.035),
                           'annotate_field_ha2': 'right',
                           'annotate_field_va2': 'bottom',
                           'annotate_field_fontsize2': 14,
                           'annotate_field_fontweight2': 'normal',
                           'annotate_field_fontstyle2': 'italic',
                           'ruler_loc': (0.005,0.12), 
                           'ruler_vertical': False, 
                           'ruler_fontsize': 10.5, 
                           'filterlabel_loc': 'center right',
                           'filterlabel_bboxtoanchor': (1.-(0.025/(1.+pad_factor)), 0.975),
                           'compassRA': 3.412784,
                           'compassDEC': -30.353649,
                           'compass_len': 0.5,
                           'compass_units': 'arcmin',
    }


    xpad = int(np.ceil(rgb_shape[1]*pad_factor))
    rgb_big_shape = (rgb_shape[0], rgb_shape[1]+xpad, 3)

    dict_NIRISS_parallel['rgb_big_shape'] = rgb_big_shape


    dict_NIRISS_parallel['clu_dicts'] =  None


    return dict_NIRISS_parallel



def _rebin(arr, new_2dshape):
    shape = (new_2dshape[0], arr.shape[0] // new_2dshape[0],
             new_2dshape[1], arr.shape[1] // new_2dshape[1])
    return arr.reshape(shape).sum(-1).sum(-2)


def _get_pixscale(hdr):
    if 'CD2_2' in hdr.keys():
        pixscale = np.abs(hdr['CD2_2'])
    elif 'PC2_2' in hdr.keys():
        pixscale = np.abs(hdr['PC2_2'])
    elif 'CDELT2' in hdr.keys():
        pixscale = np.abs(hdr['CDELT2'])
    else:
        raise ValueError("Could not get pixel scale from header!")

    pixscale *= 3600. # Convert from degrees to arcsec
    return pixscale
    


def _pstamp_radec2pix(ra, dec, wcs, loc='center', npix=101):

    if isinstance(loc, str):
        if loc == 'center':
            npixhalf = (npix-1)/2.
            npix_off = [int(npixhalf), int(npixhalf)]
        else:
            raise ValueError
    else:
        # Assume a tuple in pstamp pixel coords: 0:npix-1 for x,y
        npix_off = loc

    x, y = wcs.wcs_world2pix(ra, dec, 1)
    xint = x.astype(int); yint=y.astype(int)


    if loc is None:
        # No offsets:
        xoff = float(x)
        yoff = float(y)
    else:
        # Offset relative to the number of pixels in the pstamp
        xoff = float(x) - (xint-npix_off[0])
        yoff = float(y) - (yint-npix_off[1])

    return xoff, yoff



# Parallel to make_lupton_rgb
def _make_linear_rgb(image_r, image_g, image_b, minimum=0, maximum=None,
            filename=None):

    if maximum is None:
        maximum = np.max([image_r.max(), image_g.max(), image_b.max()])

    try:
        _ = len(maximum)
    except:
        maximum =  3*[maximum]

    try:
        _ = len(minimum)
    except:
        minimum = 3*[minimum]


    rs = (image_r - minimum[0])/(maximum[0]-minimum[0])
    rs[rs>1.] = 1.
    rs[rs<0.] = 0.

    gs = (image_g - minimum[1])/(maximum[1]-minimum[1])
    gs[gs>1.] = 1.
    gs[gs<0.] = 0.

    bs = (image_b - minimum[2])/(maximum[2]-minimum[2])
    bs[bs>1.] = 1.
    bs[bs<0.] = 0.


    rgb = np.dstack([rs,gs,bs])

    if filename:
        import matplotlib.image
        matplotlib.image.imsave(filename, rgb, origin='lower')

    return rgb


# Parallel to make_lupton_rgb
def _make_log_rgb(image_r, image_g, image_b,
                  minimum=0, maximum=None,
                  scalea=1500,
                  filename=None):

    if maximum is None:
        maximum = [image_r.max(), image_g.max(), image_b.max()]

    try:
        _ = len(maximum)
    except:
        maximum =  3*[maximum]


    if minimum is None:
        minimum = [image_r.min(), image_g.min(), image_b.min()]

    try:
        _ = len(minimum)
    except:
        minimum = 3*[minimum]


    images_s = []
    for i, img in enumerate([image_r, image_g, image_b]):
        # First, clip the data values using minimum / maximum:
        imc = (img - minimum[i])/(maximum[i]-minimum[i])
        # Clear memory:
        img = None
        imc[imc>1.] = 1.
        imc[imc<0.] = 0.

        # Then apply scaling, following DS9 convention, as given in
        # http://ds9.si.edu/doc/ref/how.html
        ims = np.log10(scalea * imc + 1)/np.log10(scalea)

        imc = None

        # Clean up the very upper end > 1 values:
        ims[ims>1.] = 1.

        images_s.append(ims)

        ims = None

    rgb = np.dstack(images_s)

    if filename:
        import matplotlib.image
        matplotlib.image.imsave(filename, rgb, origin='lower')

    return rgb





def _plot_ruler_arcsec(ax, pixscale, len_arcsec=0.5, len_arcmin=None,
        ruler_unit='arcsec', dscale=None, add_kpc=False,
        vertical=False, show_ruler_label=True,
        ruler_loc='lowerleft', color='white', ybase_offset=0.02,
        delx=0.075, dely=0.075, delx_text=0.04, dely_text=0.04,
        text_path_effects=None,
        lw=2,fontsize=8):
    ####################################
    # Show a ruler line:
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    len_line_angular = len_arcsec/(pixscale)

    if ruler_unit.lower() == 'arcsec':
        if len_arcsec % 1. == 0.:
            string = r'{}"'.format(int(len_arcsec))
        else:
            intpart = str(len_arcsec).split('.')[0]
            decpart = str(len_arcsec).split('.')[1]
            string = r'{}."{}'.format(intpart, decpart)

        if add_kpc & (dscale is not None):
            string += r' = {:0.1f} kpc'.format(len_arcsec/(1.*dscale))

    elif ruler_unit.lower() == 'kpc':
        if len_arcsec/dscale % 1. == 0.:
            string = r'{:0.0f} kpc'.format(int(len_arcsec/dscale))
        else:
            string = r'{:0.1f} kpc'.format(len_arcsec/dscale)
    elif ruler_unit.lower() == 'arcmin':
        if len_arcmin % 1. == 0.:
            string = r"{}".format(int(len_arcmin)) + u'\u2032'
        else:
            intpart = str(len_arcmin).split('.')[0]
            decpart = str(len_arcmin).split('.')[1]
            string = r"{}'{}".format(intpart, decpart)

        len_line_angular = len_arcmin/(pixscale) * 60.




    if vertical:

        if 'left' in ruler_loc:
            x_base = xlim[0] + (xlim[1]-xlim[0])*delx
            sign_y = 1.
            x_text = x_base + (xlim[1]-xlim[0])*(delx_text)
            ha = 'left'
        elif 'right' in ruler_loc:
            x_base = xlim[1] - (xlim[1]-xlim[0])*delx
            sign_y = -1.
            x_text = x_base - (xlim[1]-xlim[0])*(delx_text)
            ha = 'right'
        if 'upper' in ruler_loc:
            y_base = ylim[1] - (ylim[1]-ylim[0])*(dely)
            y_text = y_base
            va = 'top'
        elif 'lower' in ruler_loc:
            y_base = ylim[0] + (ylim[1]-ylim[0])*(dely)
            y_text = y_base
            va = 'bottom'
        else:
            # Assume it's a position, in axes fractions:

            x_base = xlim[0] + (xlim[1]-xlim[0])*ruler_loc[0]
            
            y_base = ylim[0] + (ylim[1]-ylim[0])*(ruler_loc[1])

            if ruler_loc[0] <= 0.5:
                ha = 'left'
                sign_x_text = 1.
            else:
                ha = 'right'
                sign_x_text = -1.
            if ruler_loc[1] <= 0.5:
                va = 'bottom'
                ytext_extra = 0.
                sign_y = 1.
            else:
                va = 'top'
                ytext_extra = len_line_angular
                sign_y = -1.

            x_text = x_base + (xlim[1]-xlim[0])*(delx_text)*sign_x_text
            y_text = y_base + ytext_extra



        ax.plot([x_base, x_base], [y_base+sign_y*len_line_angular, y_base],
                    c=color, ls='-',lw=lw, solid_capstyle='butt')

        if show_ruler_label:
            txt = ax.annotate(string, xy=(x_text, y_text), xycoords="data", xytext=(0,0),
                        color=color, textcoords="offset points", ha=ha, va=va,
                        fontsize=fontsize)

    else:
        if 'left' in ruler_loc:
            x_base = xlim[0] + (xlim[1]-xlim[0])*delx
            sign_x = 1.
            ha = 'left'
        elif 'right' in ruler_loc:
            x_base = xlim[1] - (xlim[1]-xlim[0])*delx
            sign_x = -1.
            ha = 'right'
        if 'upper' in ruler_loc:
            y_base = ylim[1] - (ylim[1]-ylim[0])*(ybase_offset+dely)
            y_text = y_base - (ylim[1]-ylim[0])*(dely_text)
            va = 'top'
        elif 'lower' in ruler_loc:
            y_base = ylim[0] + (ylim[1]-ylim[0])*(ybase_offset+dely)
            y_text = y_base + (ylim[1]-ylim[0])*(dely_text)
            va = 'bottom'
        else:
            # Assume it's a position, in axes fractions:
            x_base = xlim[0] + (xlim[1]-xlim[0])*ruler_loc[0]
            y_base = ylim[0] + (ylim[1]-ylim[0])*(ruler_loc[1]+ybase_offset)
            y_text = y_base + (xlim[1]-xlim[0])*(dely_text)
            sign_x = 1.
            if ruler_loc[0] <= 0.5:
                ha = 'left'
            else:
                ha = 'right'
            if ruler_loc[1] <= 0.5:
                va = 'bottom'
            else:
                va = 'top'

        ax.plot([x_base+sign_x*len_line_angular, x_base], [y_base, y_base],
                    c=color, ls='-',lw=lw, solid_capstyle='butt')

        if show_ruler_label:
            txt = ax.annotate(string, xy=(x_base, y_text), xycoords="data", xytext=(0,0),
                        color=color, textcoords="offset points", ha=ha, va=va,
                        fontsize=fontsize)

    if (text_path_effects is not None) & show_ruler_label:
        txt.set_path_effects(text_path_effects)

    return ax

def _plot_crosshair(ax, ra, dec, wcs, pixscale, sep_arcsec=0.5, len_arcsec=0.5,
                    color='white', lw=1, npix=None):
    ####################################
    # Show crosshairs:  at top, right

    if npix is None:
        raise ValueError


    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    len_crosshair_pix = len_arcsec/pixscale
    sep_pix = sep_arcsec/pixscale

    x_base, y_base = _pstamp_radec2pix(ra, dec, wcs, loc='center', npix=npix)
    sign_x = 1.  # right
    sign_y = 1.  # up

    # Check if object center in plot area
    cpad = sep_pix + len_crosshair_pix
    if ((x_base < xlim[0]-cpad) | (x_base > xlim[1]+cpad) | \
        (y_base < ylim[0]-cpad) | (y_base > ylim[1]+cpad)):
        pass
    else:
        ax.plot([x_base, x_base],
                [y_base+sign_y*sep_pix, y_base+sign_y*(sep_pix+len_crosshair_pix)],
                c=color, ls='-', lw=lw, solid_capstyle='butt')

        ax.plot([x_base+sign_x*sep_pix, x_base+sign_x*(sep_pix+len_crosshair_pix)],
                [y_base, y_base],
                c=color, ls='-', lw=lw, solid_capstyle='butt')

    return ax

def _plot_compass_rose(ax, ra, dec, wcs, xpad=0, ypad=0,
                       len=1., units='arcsec',
                       fontsize=9.5, color='white',
                       text_path_effects=None):


    if units == 'arcsec':
        len_deg = len/3600.
    elif units == 'arcmin':
        len_deg = len/60.
    elif units == 'degrees':
        len_deg = len
    else:
        raise ValueError

    text_offsetN = len_deg*0.25
    text_offsetEx = len_deg*0.35
    text_offsetEy = -len_deg*0.045


    corner = _pstamp_radec2pix(ra, dec, wcs, loc=None)
    Nend = _pstamp_radec2pix(ra, dec+len_deg, wcs, loc=None)
    Eend = _pstamp_radec2pix(ra+len_deg, dec, wcs, loc=None)

    Nlend = _pstamp_radec2pix(ra, dec+len_deg+text_offsetN, wcs, loc=None)
    Elend = _pstamp_radec2pix(ra+len_deg+text_offsetEx, dec+text_offsetEy,
                              wcs, loc=None)


    for end, lend, lbl in zip([Nend, Eend], [Nlend, Elend], ['N', 'E']):
        ax.arrow(corner[0]+xpad, corner[1]+ypad,
                 end[0]-corner[0], end[1]-corner[1],
                 color=color, head_width=90., width = 1.) 

        if lbl == 'N':
            va = 'bottom'
            ha = 'center'
        elif lbl == 'E':
            va = 'center'
            ha = 'right'
        txt = ax.annotate(lbl, [lend[0]+xpad, lend[1]+ypad], va=va, ha=ha,
                    color=color, fontsize=fontsize, xycoords='data')

        if text_path_effects is not None:
            txt.set_path_effects(text_path_effects)

    return ax


def load_data(fname_img, filt=None, upsample=False):

    """
    Load image into a data dictionary.
    There is an option to upsample the LW (or downsample the SW)
    
    Parameters
    ----------

        fname_img: str
            FITS image filename

        filt: str
            Name of filter

        upsample: bool, optional
            Option to upsample the LW filters to match the SW filter pixelscale (if True)
            Otherwise, downsample the SW filters to match the LW pixelscale.
            Default: False 

    Returns
    -------
        data: dict
            Dictionary containing the image, header, WCS, and filter.

    """

    if filt is None:
        raise ValueError("Must pass filter name with keyword arguement 'filt'!")

    data = {}

    # print("reading img: {}".format(fname_img))

    # get images
    hdu = fits.open(fname_img, memmap=True)
    img = copy.deepcopy(hdu[0].data)

    hdr = copy.deepcopy(hdu[0].header)
    wcs = WCS(copy.deepcopy(hdr))


    # Use PHOTFNU value to convert to Jy
    # then multiply by 10^9 to get nJy
    pfnu_filt_nJy = hdr['PHOTFNU'] * 1.e9

    img *= pfnu_filt_nJy



    pixscale = _get_pixscale(hdr)

    if not upsample:
        # Rebin to 0.04" platescale:
        if int(np.ceil(pixscale*1000.)) == 20:
            img_orig = copy.deepcopy(img)
            # Preserve flux: just sum within rebinned pixels
            img = _rebin(img_orig, (int(img_orig.shape[0]/2),
                                            int(img_orig.shape[1]/2)))

            hdr_orig = copy.deepcopy(hdr)
            # Modify header:
            for num in [1,2]:
                hdr['NAXIS{}'.format(num)] = img.shape[0]

                hdr['CRPIX{}'.format(num)] = (hdr_orig['CRPIX{}'.format(num)]+0.5)/2.
                hdr['CD{}_{}'.format(num,num)] = hdr_orig['CD{}_{}'.format(num,num)]*2.

            wcs = WCS(copy.deepcopy(hdr))
    else:
        # Upsample to 0.02 platescale:
        if int(np.ceil(pixscale*1000.)) == 40:
            # print("Upsampling!")
            img_orig = copy.deepcopy(img)
            hdr_orig = copy.deepcopy(hdr)

            # Just block upsample values:

            img = np.zeros((int(img_orig.shape[0]*2),
                            int(img_orig.shape[1]*2)))
            for offsety in [0,1]:
                for offsetx in [0,1]:
                    # Preserve flux: downscale
                    img[offsety:img.shape[0]-1+offsety:2,
                        offsetx:img.shape[1]-1+offsetx:2] = img_orig *0.25


            # Modify header:
            for num in [1,2]:
                hdr['NAXIS{}'.format(num)] = img.shape[0]

                hdr['CRPIX{}'.format(num)] = hdr_orig['CRPIX{}'.format(num)]*2. - 0.5
                hdr['CD{}_{}'.format(num,num)] = hdr_orig['CD{}_{}'.format(num,num)]/2.

            wcs = WCS(copy.deepcopy(hdr))

    data['IMG'] = img
    data['WCS'] = wcs
    data['HDR'] = hdr
    data['filt'] = filt


    # Garbage collect:
    img = None
    wcs = None
    hdr = None

    hdu.close()

    hdu = None


    return data



def make_dataRGB(data_all=None, filters=None):
    """
    Take an input dict of dicts and create summed images for R/G/B.
    
    Parameters
    ----------

        data_all: dict
            Dictionary, containing dictionaries of data for multiple filters. 
            Each sub-dictionary (named with the filter name) -- as returned by load_data() -- 
            has the following structure (eg, for 'F444W'):
                data_all['F444W'] = {'IMG': <image>, 'WCS': <WCS>, 'HDR': <HDR>, 'filt': 'F444W'}

        filt: str
            Name of filter. For summing multiple filters, the name should 
            be a concatenation of the individual filter names, separated by '+'
            eg, 'F356W+F410M+F444W'

        upsample: bool, optional
            Option to upsample the LW filters to match the SW filter pixelscale (if True)
            Otherwise, downsample the SW filters to match the LW pixelscale.
            Default: False 

    Returns
    -------
        data: dict
            Dictionary containing the composite R/G/B images, headers, WCSs, and filters

    """
    
    colorprefix = ['b','g','r']
    dataRGB = {}
    for colprefix, filtsum in zip(colorprefix, filters):
        filtlist = filtsum.split('+')
        imsum = np.zeros(data_all[filtlist[-1]]['IMG'].shape)
        for filt in filtlist:
            imsum += data_all[filt]['IMG']
        dataRGB['{}IMG'.format(colprefix)] = imsum
        dataRGB['{}WCS'.format(colprefix)] = data_all[filtlist[-1]]['WCS']
        dataRGB['{}HDR'.format(colprefix)] = data_all[filtlist[-1]]['HDR']
        dataRGB['{}filt'.format(colprefix)] = filtsum
        
    return dataRGB





def plot_full_mosaic_RGB(fileout=None, data=None, 
                         dict_fieldopts=None, 
                         imscale_type='log', 
                         minimum=0, maximum=None, scalea=1500, 
                         Q=8, stretch=2, 
                         show_filters=True, 
                         len_ruler_arcmin=1., show_wedge_outline=False, 
                         plot_scale_inches=7., dpi=300): 

    """
    Script to plot UNCOVER NIRCAM & NIRISS mosaics, with settings about positioning / cropping 
    passed as part of a dict.
    
    Parameters
    ----------
        fileout: str or None
            If specified, save figure to this file. If None, the figure will instead be displayed.
            Default: None

        data: dict
            Dictionary containing information for the R, G, B images, as made by make_dataRGB.
            For each of prefix = ['b', 'g', 'r'], 
            it contains 'IMG', 'WCS', 'HDR', 'filt' (prepended by the prefix), where
            IMG contains the 2D image (or composite image), WCS is a astropy.wcs.WCS instance created from the 
            image HDR, and filt is the filter (or composite filter name)

        dict_fieldopts: dict
            Dictionary containing options for making the mosaic image.
            For example, see _get_dict_NIRCAM_primary(shape), _get_dict_NIRISS_primary(shape), 

        imscale_type: str, optional
            Type of scaling to apply to image. Options: 'log', 'linear', 'asinh'. Default: 'log'

        minimum: float or array-like or None, optional
            Minimum value for image scaling. 
            If float, applied to all 3 images (R/G/B). 
            If array-like, must have length 3, with the ordering [r_min, g_min, b_min].
            If None, uses the minimum of each image.
            Default: 0.

        maximum: float or array-like or None, optional
            Maximum value for image scaling (only used for 'linear' or 'log').
            If float, applied to all 3 images (R/G/B). 
            If array-like, must have length 3, with the ordering [r_max, g_max, b_max].
            If None, uses the maximum of each image.
            Default: None

        scalea: float, optional
            Scale factor for 'log' scaling, following the DS9 scaling convention. 
            See http://ds9.si.edu/doc/ref/how.html
            Default: 1500. 

        Q: float, optional
            Factor for asinh scaling, using make_lupton_rgb. Default: 8.

        stretch: float, optional
            Factor for asinh stretch, using make_lupton_rgb. Default: 2.

        show_filters: bool, optional
            Option to show the color label with the filters. Default: True

        len_ruler_arcmin: float, optional
            Length of the scale bar, in arcmin. Default: 1.

        show_wedge_outline: bool, optional
            Option to show an outline around the inset wedges. 
            Can be helpful for iterating on wedge enpoint positioning. Default: False

        plot_scale_inches: float, optional
            Height of the resulting figure. Default: 7.

        dpi: float, optional. 
            Resolution for the saved figure. 
            Excessively high dpi for very large mosaics can result in matplotlib problems.
            Default: 300.


    """

    if data is None:
        errmsg =  " Must pass 'data', \n"
        errmsg += " a dict containing the loaded images RGB filter images! "
        raise ValueError(errmsg)

    if dict_fieldopts is None:
        errmsg =  " Must pass 'dict_fieldopts', \n"
        errmsg += " a dict containing the settings for making a mosaic! "
        raise ValueError(errmsg)



    # Unpack plotting options per-field:
    pad_factor = dict_fieldopts.get('pad_factor', 0.)
    w_logo = dict_fieldopts.get('w_logo', 0.2)
    loc_logo = dict_fieldopts.get('loc_logo', [0.,0.])
    filters_fontsize = dict_fieldopts.get('filters_fontsize', 10)
    annotate_field_label = dict_fieldopts.get('annotate_field_label', 'Field')
    annotate_field_loc = dict_fieldopts.get('annotate_field_loc', [0.95,0.05])
    annotate_field_ha = dict_fieldopts.get('annotate_field_ha', 'right')
    annotate_field_va = dict_fieldopts.get('annotate_field_va', 'bottom')
    annotate_field_fontsize = dict_fieldopts.get('annotate_field_fontsize', 14)
    annotate_field_fontweight = dict_fieldopts.get('annotate_field_fontweight', 'normal')
    ruler_loc = dict_fieldopts.get('ruler_loc', [0.1,0.1])
    ruler_vertical = dict_fieldopts.get('ruler_vertical', False)
    filterlabel_loc = dict_fieldopts.get('filterlabel_loc', 'center left')
    filterlabel_bboxtoanchor = dict_fieldopts.get('filterlabel_bboxtoanchor', [0.05,0.95])
    clu_dicts = dict_fieldopts.get('clu_dicts', None)

    offset_factor = 0.5*pad_factor



    nrows = ncols = 1
    scale = plot_scale_inches



    f, ax = plt.subplots(nrows, ncols) 

    f.subplots_adjust(top=1.0, bottom=0, right=1.0, left=0, hspace=0, wspace=0)


    # Get npix from arcsec size:
    pixscale = _get_pixscale(data['rHDR'])

    if 'crop_x_pix' in dict_fieldopts.keys():
        xpx = dict_fieldopts['crop_x_pix']
        xoff_crop = xpx[0]
    else:
        xpx = [0,data['rIMG'].shape[1]]
        xoff_crop = 0
    if 'crop_y_pix' in dict_fieldopts.keys():
        ypx = dict_fieldopts['crop_y_pix']
        yoff_crop = ypx[0]
    else:
        ypx = [0,data['rIMG'].shape[0]]
        yoff_crop = 0



    if imscale_type == 'asinh':
        rgb = make_lupton_rgb(data['rIMG'][ypx[0]:ypx[1],xpx[0]:xpx[1]],
                              data['gIMG'][ypx[0]:ypx[1],xpx[0]:xpx[1]],
                              data['bIMG'][ypx[0]:ypx[1],xpx[0]:xpx[1]],
                              Q=Q, stretch=stretch, minimum=minimum)
    elif imscale_type == 'linear':
        rgb = _make_linear_rgb(data['rIMG'][ypx[0]:ypx[1],xpx[0]:xpx[1]],
                               data['gIMG'][ypx[0]:ypx[1],xpx[0]:xpx[1]],
                               data['bIMG'][ypx[0]:ypx[1],xpx[0]:xpx[1]],
                               minimum=minimum, maximum=maximum)
    elif imscale_type == 'log':
        rgb = _make_log_rgb(data['rIMG'][ypx[0]:ypx[1],xpx[0]:xpx[1]],
                            data['gIMG'][ypx[0]:ypx[1],xpx[0]:xpx[1]],
                            data['bIMG'][ypx[0]:ypx[1],xpx[0]:xpx[1]],
                            minimum=minimum, maximum=maximum, scalea=scalea)
    else:
        raise ValueError("scale imscale_type='{}' unknown!'".format(imscale_type))

    # Embed rgb in a BIGGER array:
    xoff_bigarr = int(np.ceil(rgb.shape[1]*offset_factor))
    xpad = int(np.ceil(rgb.shape[1]*pad_factor))

    rgb_big = np.zeros((rgb.shape[0], rgb.shape[1]+xpad, rgb.shape[2]))
    rgb_big[:, xoff_bigarr:rgb.shape[1]+xoff_bigarr,:] = rgb[:,:,:]



    aspect_ratio = rgb_big.shape[1]*1./rgb_big.shape[0]

    # Do a round about way to ensure no rounding offsets
    #  leading to missing pixels
    npixtmp = scale*(aspect_ratio)*dpi
    npixtmp_int = int(np.ceil(npixtmp))
    aspect_ratio *= npixtmp_int/npixtmp

    f.set_size_inches(ncols*scale*(aspect_ratio),nrows*scale)


    # Free up memory:
    rgb = None

    ax.imshow(rgb_big, origin='lower', interpolation='none')

    ax.set_xticks([])
    ax.set_yticks([])

    if show_filters:
        textpropsdicts = {}
        fontsize = filters_fontsize
        sep_textboxes = 1
        lw_outline = fontsize/15.
        col_outline = 'grey'
        for col in ['b', 'g', 'r', 'w']:
            textprops = dict(color=col, size=fontsize,ha='left',va='top')
            textpropsdicts[col] = textprops

        yboxes = []
        for j, col in enumerate(['b', 'g', 'r']):
            if j > 0:
                ybx = TextArea('-', textprops=textpropsdicts['w'])

                yboxes.append(ybx)

            ybx = TextArea(data['{}filt'.format(col)],
                        textprops=textpropsdicts[col])

            ybx._text.set_path_effects([path_effects.Stroke(linewidth=lw_outline,
                                                       foreground=col_outline),
                                   path_effects.Normal()])

            yboxes.append(ybx)

        ybox = HPacker(children=yboxes,align="left", pad=0, sep=sep_textboxes)

        anchored_ybox = AnchoredOffsetbox(loc=filterlabel_loc, child=ybox,
                                pad=0., frameon=False,
                                bbox_to_anchor=filterlabel_bboxtoanchor,
                                bbox_transform=ax.transAxes, borderpad=0.)
        ax.add_artist(anchored_ybox)


    # --------------------------------------------------------------
    if not isinstance(ruler_loc, str):
        ruler_loc_new = [ruler_loc[0]+(0.75*xoff_bigarr/float(rgb_big.shape[1])),
                         ruler_loc[1]]
    else:
        ruler_loc_new = ruler_loc
    ruler_fontsize = dict_fieldopts.get('ruler_fontsize', 9.5)
    ruler_lw = dict_fieldopts.get('ruler_lw', 1.5)

    ax = _plot_ruler_arcsec(ax, pixscale, len_arcmin=len_ruler_arcmin,
            ruler_unit='arcmin', color='white', lw=ruler_lw,
            vertical=ruler_vertical, ruler_loc=ruler_loc_new, fontsize=ruler_fontsize,
            text_path_effects = [path_effects.Stroke(linewidth=ruler_fontsize/25.,
                                                       foreground='white'),
                                   path_effects.Normal()],
                        delx_text=0.04*.25, dely_text=0.035*.25)


    # Add compass rose:
    ax = _plot_compass_rose(ax, dict_fieldopts['compassRA'],
                            dict_fieldopts['compassDEC'], data['rWCS'],
                            xpad=xoff_bigarr-xoff_crop, 
                            ypad=-yoff_crop,
                            len=dict_fieldopts['compass_len'],
                            units=dict_fieldopts['compass_units'],
                            fontsize=9.,
                            color='white',
                            text_path_effects = [path_effects.Stroke(linewidth=9./25.,
                                            foreground='white'),
                                            path_effects.Normal()])


    # --------------------------------------------------------------
    # # Add logo:

    # LOGO FILENAME
    _path = os.path.abspath(__file__)
    _dir_flogo = os.sep.join([os.path.dirname(_path), "assets", ""])
    f_logo = _dir_flogo+'UNCOVER_logo_white_dpi1800.png'


    # read image file
    arr_logo = plt.imread(f_logo)#, format='png')
    h_logo = w_logo * (arr_logo.shape[0]/ arr_logo.shape[1])


    if loc_logo is not None:
        # Draw logo
        axlogo = ax.inset_axes([loc_logo[0], loc_logo[1], w_logo, h_logo],transform=ax.transAxes)

        axlogo.imshow(arr_logo)
        axlogo.axis('off')

    # Free up memory
    arr_logo = None


    # --------------------------------------------------------------
    # Setup insets for cluster centers:

    lw_inset = 2
    bcolor_inset = 'darkred'

    alpha_lower = 0.1
    alpha_upper = 0.5 
    len_alpha = 1000
    cmap_alpha = ListedColormap(["#def2ff"]) 
    val_alpha = 1.
    Z, _ = np.meshgrid(np.linspace(alpha_lower,alpha_upper, num=len_alpha),
                        np.linspace(alpha_lower,alpha_upper, num=len_alpha))

    # --------------------------------------------------------------
    # Insets for the clusters:
    axes_clu = []
    if clu_dicts is not None:

        for key in clu_dicts.keys():
            clu_dict = clu_dicts[key]
            axclu = ax.inset_axes([clu_dict['xpos'], clu_dict['ypos'],
                    clu_dict['width'], clu_dict['height']], transform=ax.transData,
                    zorder=20)
                
            if 'cutout' in clu_dict.keys():
                if clu_dict['cutout']['corners_radec'] is not None:
                    x0, y0 = data['rWCS'].wcs_world2pix(clu_dict['cutout']['corners_radec'][0][0],
                        clu_dict['cutout']['corners_radec'][0][1], 1)
                    x1, y1 = data['rWCS'].wcs_world2pix(clu_dict['cutout']['corners_radec'][1][0],
                        clu_dict['cutout']['corners_radec'][1][1], 1)

                    x0 = int(x0 + xoff_bigarr-xoff_crop)
                    y0 = int(y0 -yoff_crop)
                    x1 = int(x1 + xoff_bigarr-xoff_crop)
                    y1 = int(y1 -yoff_crop)
                else:
                    x0 = int(clu_dict['cutout']['xlims_pix'][0])
                    x1 = int(clu_dict['cutout']['xlims_pix'][1])
                    y0 = int(clu_dict['cutout']['ylims_pix'][0])
                    y1 = int(clu_dict['cutout']['ylims_pix'][1])


                axclu.imshow(rgb_big[y0:y1,x0:x1,:], origin='lower',
                          interpolation='none')

            else:
                x0 = y0 = x1 = y1 = None
                axclu.set_facecolor('purple')

            if clu_dict['label'] is not None:
                lbl_loc = clu_dict.get('label_loc', (0.05,0.95))
                if lbl_loc is None:
                    lbl_loc = (0.05,0.95)
                va = clu_dict.get('label_va', 'top')
                if va is None:
                    va = 'top'
                ha = clu_dict.get('label_ha', 'left')
                if ha is None:
                    ha = 'left'

                axclu.annotate(clu_dict['label'], lbl_loc,
                            va=va, ha=ha, color='white',
                            xycoords='axes fraction',
                            fontsize=10)

            axclu.set_xticks([])
            axclu.set_yticks([])
            for spos in ['top', 'bottom', 'left', 'right']:
                axclu.spines[spos].set_color(bcolor_inset)
                axclu.spines[spos].set_linewidth(lw_inset)

            axes_clu.append(axclu)

            # --------------------------------------------------------------
            # Add the wedges to clusters:

            if 'inset_wedge' in clu_dict.keys():
                if clu_dict['inset_wedge']['position_radec'] is not None:
                    # Get from wcs / hdr in dict..

                    xw, yw = data['rWCS'].wcs_world2pix(clu_dict['inset_wedge']['position_radec'][0],
                        clu_dict['inset_wedge']['position_radec'][1], 1)

                    position_pix = [int(xw + xoff_bigarr-xoff_crop),
                                    int(yw-yoff_crop)]

                else:
                    position_pix = clu_dict['inset_wedge']['position_pix']


                if clu_dict['inset_wedge']['direction'] == 'right':
                    xextra = 0.
                    ord = 1
                else:
                    xextra = clu_dict['width']
                    ord = -1

                corners = [ [clu_dict['xpos']+xextra, clu_dict['ypos']],
                             position_pix,
                            [clu_dict['xpos']+xextra,
                              clu_dict['ypos']+clu_dict['height']],
                            [clu_dict['xpos']+xextra, clu_dict['ypos']] ]

                xbounds = None
                ybounds = None
                for cor in corners:
                    if xbounds is None: xbounds = [cor[0], cor[0]]
                    if ybounds is None: ybounds = [cor[1], cor[1]]
                    if cor[0] < xbounds[0]: xbounds[0] = cor[0]
                    if cor[0] > xbounds[1]: xbounds[1] = cor[0]

                    if cor[1] < ybounds[0]: ybounds[0] = cor[1]
                    if cor[1] > ybounds[1]: ybounds[1] = cor[1]

                corners_ax = []
                for cor in corners:
                    corners_ax.append([(cor[0]-xbounds[0])/(xbounds[1]-xbounds[0]),
                                       (cor[1]-ybounds[0])/(ybounds[1]-ybounds[0])])

                ax2 = ax.inset_axes([xbounds[0], ybounds[0], xbounds[1]-xbounds[0],
                                    ybounds[1]-ybounds[0]], transform=ax.transData,
                                    zorder=10)
                ax2.axis("off")

                path = Path(corners_ax)
                if show_wedge_outline:
                    edgeccolwedge = 'red'
                else:
                    edgeccolwedge ='none'
                patch = PathPatch(path, facecolor='none', edgecolor=edgeccolwedge)
                ax2.add_patch(patch)


                im = ax2.imshow(Z*0.+val_alpha, interpolation='bilinear',
                                alpha = Z[:,::ord]**2, 
                                cmap=cmap_alpha, origin='lower', aspect='auto',
                                extent=[0,1,0,1], clip_path=patch, clip_on=True, zorder=5)
                im.set_clip_path(patch)




    # --------------------------------------------------------------
    # Annotate main axis:
    if annotate_field_label is not None:
        ax.annotate(annotate_field_label, annotate_field_loc,
                    ha=annotate_field_ha, va=annotate_field_va, color='white',
                    xycoords='axes fraction', fontweight=annotate_field_fontweight,
                    fontsize=annotate_field_fontsize, zorder=100)


    for enum in range(2,3):
        if 'annotate_field_label{}'.format(enum) in dict_fieldopts.keys():
            annotate_field_fontstyle = dict_fieldopts.get('annotate_field_fontstyle{}'.format(enum),
                                                          None)
            prop = {}
            if annotate_field_fontstyle is not None:
                if annotate_field_fontstyle == 'italic':
                    prop={'fontname':'arial'}
            ax.annotate(dict_fieldopts['annotate_field_label{}'.format(enum)],
                        dict_fieldopts['annotate_field_loc{}'.format(enum)],
                        ha=dict_fieldopts['annotate_field_ha{}'.format(enum)],
                        va=dict_fieldopts['annotate_field_va{}'.format(enum)],
                        color='white', xycoords='axes fraction',
                        fontweight=dict_fieldopts['annotate_field_fontweight{}'.format(enum)],
                        fontsize=dict_fieldopts['annotate_field_fontsize{}'.format(enum)],
                        style=annotate_field_fontstyle,
                        zorder=100, **prop)


    # Turn off primary axis
    ax.set_axis_off()


    if fileout is not None:
        f.savefig(fileout, dpi=dpi, pad_inches=0)

        plt.close('all')

        return None

    else:
        plt.show()
