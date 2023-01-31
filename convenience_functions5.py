from astropy import visualization as aviz
from astropy.nddata.utils import block_reduce, Cutout2D
from astropy.nddata import CCDData
from astropy.io import fits
from astropy.stats import sigma_clipped_stats, SigmaClip
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import re

#=============================================================================

def find_filter(image, filter_list, filter_keywords):
    """
    Given an image and some filter keywords, search if any of these keywords
    have a filter matching the filter list. The function will return the index
    within the filter list where the match occurred.

    Arguments
    ---------
        image : string
            filename of the image
        filter_list : list of strings
            possible filter names to match with the filter keywords
        filter_keywords : list of strings
            header keywords contining filter information

    Returns
    -------
        Index withing the filter list, where the filter match occurred
    """
        
    hdr=fits.getheader(image, ext=0)
    for key in filter_keywords:
        filt = hdr[key]
        if filt in filter_list:
            return filter_list.index(filt)

#=============================================================================
        
def center_inv_median(image_matrix, frac_size=0.5):
    """
    Given an image return the inverse of the median from the central region
    of the image (1/4 of the image). To use whithin the make_mflat function.

    Arguments
    ---------
        image_matrix : 2D np.array
            matrix of the image.

    Returns
    -------
        inv_med : float
            Inverse of the central region median
    """
    #  Orientation
    ysz, xsz = image_matrix.shape
    yc, xc = int(ysz/2), int(xsz/2)                     # Center
    dy, dx = int(ysz*frac_size/2), int(xsz*frac_size/2) # Step

    #  Limits
    a, b = yc - dy, yc + dy
    c, d = xc - dx, xc + dy

    roi = image_matrix[a:b, c:d]

    inv_med = 1/np.median(roi)

    return inv_med

#=============================================================================

def center_median(image_matrix, frac_size=0.5):

    #  Orientation
    ysz, xsz = image_matrix.shape
    yc, xc = int(ysz/2), int(xsz/2)                     # Center
    dy, dx = int(ysz*frac_size/2), int(xsz*frac_size/2) # Step

    #  Limits
    a, b = yc - dy, yc + dy
    c, d = xc - dx, xc + dy

    roi = image_matrix[a:b, c:d]

    return np.median(roi),(np.std(roi))

#=============================================================================

def flat_scale(flat_list, normalize=True):
    """
    Given a list of flat field fits files, parse through all extensions, 
    recording the median inside a selectable region near the center of the 
    image.

    Arguments
    ---------
        flat_list : list
            list of flat field images to analyse
        normalize : bool
            when set, a normalization across all amplifiers is carried out
            and the returned table will contain numbers between 0 and 1.

    Returns
    -------
        scales : numpy 2D array (n_files, n_extensions)
            median of the central region values for every extension in each 
            image of the input file list. If "normalize" parameter is set 
            then this table is divided by its maximum value.
    """
    
    for j,flat in enumerate(flat_list,start=0):
        hdus = fits.open(flat)
        if (j == 0):
            image_indices = np.arange(len(hdus))
            selection = [hdu.size > 0 for hdu in hdus]
            image_indices = image_indices[selection]
            scales = np.zeros([len(flat_list),len(image_indices)])
    
        for i,idx in enumerate(image_indices,start=0):
            scales[j,i], _ = center_median(hdus[idx].data, frac_size=0.25)

        hdus.close()
    
    scales = np.amax(scales, axis=1)
    
    if normalize:
        scales /= scales.max()
        
    return scales


#=============================================================================
        
def image_snippet(image, center, width=50, axis=None, fig=None,
                  is_mask=False, pad_black=False, **kwargs):
    """
    Display a subsection of an image about a center.

    Parameters
    ----------

    image : numpy array
        The full image from which a section is to be taken.

    center : list-like
        The location of the center of the cutout.

    width : int, optional
        Width of the cutout, in pixels.

    axis : matplotlib.Axes instance, optional
        Axis on which the image should be displayed.

    fig : matplotlib.Figure, optional
        Figure on which the image should be displayed.

    is_mask : bool, optional
        Set to ``True`` if the image is a mask, i.e. all values are
        either zero or one.

    pad_black : bool, optional
        If ``True``, pad edges of the image with zeros to fill out width
        if the slice is near the edge.
    """
    if pad_black:
        sub_image = Cutout2D(image, center, width, mode='partial', fill_value=0)
    else:
        # Return a smaller subimage if extent goes out side image
        sub_image = Cutout2D(image, center, width, mode='trim')
    show_image(sub_image.data, cmap='gray', ax=axis, fig=fig,
               show_colorbar=False, show_ticks=False, is_mask=is_mask,
               **kwargs)

#=============================================================================

def _mid(sl):
    return (sl.start + sl.stop) // 2

#=============================================================================

def display_cosmic_rays(cosmic_rays, images, titles=None,
                        only_display_rays=None):
    """
    Display cutouts of the region around each cosmic ray and the other images
    passed in.

    Parameters
    ----------

    cosmic_rays : photutils.segmentation.SegmentationImage
        The segmented cosmic ray image returned by ``photuils.detect_source``.

    images : list of images
        The list of images to be displayed. Each image becomes a column in
        the generated plot. The first image must be the cosmic ray mask.

    titles : list of str
        Titles to be put above the first row of images.

    only_display_rays : list of int, optional
        The number of the cosmic ray(s) to display. The default value,
        ``None``, means display them all. The number of the cosmic ray is
        its index in ``cosmic_rays``, which is also the number displayed
        on the mask.
    """
    # Check whether the first image is actually a mask.

    if not ((images[0] == 0) | (images[0] == 1)).all():
        raise ValueError('The first image must be a mask with '
                         'values of zero or one')

    if only_display_rays is None:
        n_rows = len(cosmic_rays.slices)
    else:
        n_rows = len(only_display_rays)

    n_columns = len(images)

    width = 12

    # The height below is *CRITICAL*. If the aspect ratio of the figure as
    # a whole does not allow for square plots then one ends up with a bunch
    # of whitespace. The plots here are square by design.
    height = width / n_columns * n_rows
    fig, axes = plt.subplots(n_rows, n_columns, sharex=False, sharey='row',
                             figsize=(width, height))

    # Generate empty titles if none were provided.
    if titles is None:
        titles = [''] * n_columns

    display_row = 0

    for row, s in enumerate(cosmic_rays.slices):
        if only_display_rays is not None:
            if row not in only_display_rays:
                # We are not supposed to display this one, so skip it.
                continue

        x = _mid(s[1])
        y = _mid(s[0])

        for column, plot_info in enumerate(zip(images, titles)):
            image = plot_info[0]
            title = plot_info[1]
            is_mask = column == 0
            ax = axes[display_row, column]
            image_snippet(image, (x, y), width=80, axis=ax, fig=fig,
                          is_mask=is_mask)
            if is_mask:
                ax.annotate('Cosmic ray {}'.format(row), (0.1, 0.9),
                            xycoords='axes fraction',
                            color='cyan', fontsize=20)

            if display_row == 0:
                # Only set the title if it isn't empty.
                if title:
                    ax.set_title(title)

        display_row = display_row + 1

    # This choice results in the images close to each other but with
    # a small gap.
    plt.subplots_adjust(wspace=0.1, hspace=0.05)

#=============================================================================

def iraf2python(my_string):
    
    """
    This function is from Bruno Quint.
    
    Parse a string containing [XX:XX, YY:YY] to pixels.
    Parameter
    ---------
        my_string : str
    """
    
    my_string = my_string.strip()
    my_string = my_string.replace('[', '')
    my_string = my_string.replace(']', '')
    x, y = my_string.split(',')
    x = x.split(':')
    y = y.split(':')

    x = np.array(x, dtype=int)
    y = np.array(y, dtype=int)

    x[0] -= 1
    y[0] -= 1

    return x, y

#=============================================================================

def show_image(image,
               percl=99, percu=None, is_mask=False,
               figsize=(10, 10), fontsize=14,
               cmap='viridis', stretch=aviz.LinearStretch(), mask=None,
               show_colorbar=True, show_ticks=True,
               fig=None, ax=None, input_ratio=None):
    """
    Show an image in matplotlib with some basic astronomically-appropriat stretching.

    Parameters
    ----------
    image
        The image to show
    percl : number
        The percentile for the lower edge of the stretch (or both edges if ``percu`` is None)
    percu : number or None
        The percentile for the upper edge of the stretch (or None to use ``percl`` for both)
    figsize : 2-tuple
        The size of the matplotlib figure in inches
    """
    
    # Changing general font size for Pyplot
    plt.rcParams.update({'font.size': fontsize})
    
    if percu is None:
        percu = percl
        percl = 100 - percl

    if (fig is None and ax is not None) or (fig is not None and ax is None):
        raise ValueError('Must provide both "fig" and "ax" '
                         'if you provide one of them')
    elif fig is None and ax is None:
        if figsize is not None:
            # Rescale the fig size to match the image dimensions, roughly
            image_aspect_ratio = image.shape[0] / image.shape[1]
            figsize = (max(figsize) * image_aspect_ratio, max(figsize))

        fig, ax = plt.subplots(1, 1, figsize=figsize)


    # To preserve details we should *really* downsample correctly and
    # not rely on matplotlib to do it correctly for us (it won't).

    # So, calculate the size of the figure in pixels, block_reduce to
    # roughly that,and display the block reduced image.

    # Thanks, https://stackoverflow.com/questions/29702424/how-to-get-matplotlib-figure-size
    fig_size_pix = fig.get_size_inches() * fig.dpi

    ratio = (image.shape // fig_size_pix).max()

    if ratio < 1:
        ratio = 1

    ratio = input_ratio or ratio

    reduced_data = block_reduce(image, ratio)

    if not is_mask:
        # Divide by the square of the ratio to keep the flux the same in the
        # reduced image. We do *not* want to do this for images which are
        # masks, since their values should be zero or one.
         reduced_data = reduced_data / ratio**2

    # Of course, now that we have downsampled, the axis limits are changed to
    # match the smaller image size. Setting the extent will do the trick to
    # change the axis display back to showing the actual extent of the image.
    extent = [0, image.shape[1], 0, image.shape[0]]

    norm = aviz.ImageNormalize(reduced_data,
                               interval=aviz.AsymmetricPercentileInterval(percl, percu),
                               stretch=stretch)

    if is_mask:
        # The image is a mask in which pixels should be zero or one.
        # block_reduce may have changed some of the values, so reset here.
        reduced_data = reduced_data > 0
        # Set the image scale limits appropriately.
        scale_args = dict(vmin=0, vmax=1)
    else:
        scale_args = dict(norm=norm)

        
    if mask is not None:
        msk = block_reduce(mask, ratio)
        msk = msk > 0
        reduced_data[msk == 1] = np.nan


    im = ax.imshow(reduced_data, origin='lower',
                   cmap=cmap, extent=extent, aspect='equal', **scale_args)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(256))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(256))

    if show_colorbar:
        sep, wid = 0.01, 0.03
        cax = fig.add_axes([ax.get_position().x1+sep,ax.get_position().y0,wid,ax.get_position().height])
        plt.colorbar(im, cax=cax)

    if not show_ticks:
        ax.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)

#=============================================================================

def soardisplay(image, 
             interval_class=aviz.ZScaleInterval(contrast=0.5), 
             stretch_class=aviz.LinearStretch(),
             figsize=(10, 10),
             fontsize=14,
             cmap='gist_gray',
             mask=None,
             is_mask=False, 
             trim_overscan=True, 
             show_colorbar=True, 
             show_labels=True,
             show_title=True,
             fig=None, ax=None, 
             input_ratio=None):
    """
    Show an image in matplotlib with some basic astronomically-appropriate stretching.

    Parameters
    ----------
    image : string
        The image to show
    interval_class : astropy.visualization.BaseInterval class, used to define the pixel value 
                      limits to be employed in the image visualization. 
                      Default: ZScaleInterval(contrast=0.5)
    stretch_class : astropy.visualization.BaseStretch class, used to map the selected pixel values 
                     range into a color intensity value in the image visualization.
                     Default: LinearStretch()
    figsize : 2-tuple
        The size of the matplotlib figure in inches.
        Defalut: (10,10)
    fontsize : int
        Matplotlib fontsize for use on the plot labels.
        Default: 14
    cmap : string
        default string names for Matplotlib color maps.
        Default: 'gist_gray'
    mask : string
        filename for a pixel mask to be applied to this image. Masked pixels are not displayed.
        Default: None
    is mask : boolean
        indicates that the input image is mask, and pixels can only assume values of 0 or 1 
        Defalut: False
    trim_overscan : boolean
        indicates that data is supposed to be extracted only for pixels in DATASEC (keyword) range
        Defalut: True
    show_colorbar: boolean
        indicates whether colorbar should be plotted or not (on the right side of the plot)
        Defalut: True
    show_label: boolean
        indicates whether labels and ticks should be plotted or not around the image frame
        Defalut: True
    show_title: boolean
        indicates whether image and object name should be written or not on top of the plot
        Defalut: True
    fig, ax: figure and axis instances 
        on input, the plot will use these instances to draw the image and objects
        Default: None, None
    input_ratio: 
        don't have a clue! Have to make some tests or ask Bruno Quint
        Default: None
    ----------

    Changelog:
    2016?    - Original by Bruno Quint
    feb/2021 - Largely modified by F.Maia to set up a global reference frame and properly draw 
                the pixel axes. Also implemented many optional keywords.

    ----------
    """
       
    # Changing general font size for Pyplot
    plt.rcParams.update({'font.size': fontsize})

    # Creating figure and axes instance
    if (fig is None and ax is not None) or (fig is not None and ax is None):
        raise ValueError('Must provide both "fig" and "ax" if you provide one of them')
    elif fig is None and ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Getting header information
    hdus = fits.open(image)
    h0 = hdus[0].header
    
    # Detecting SOAR imager used
    inst_list = ['SAM','SOI','Goodman']
    instrument = h0['INSTRUME']
    inst_flag = []
    for inst in inst_list:
        p = re.compile(inst,re.IGNORECASE)
        inst_flag.append(p.search(instrument) is not None)

    # Selecting extensions with image data
    image_indices = np.arange(len(hdus))
    selection = [hdu.size > 0 for hdu in hdus]
    image_indices = image_indices[selection]
    n_ext = len(image_indices)

    # Looping over the image extensions
    if n_ext == 1: img = hdus[0].data
    else: 
        for i in image_indices:
        
            # Getting useful data section
            hdr = hdus[i].header
            binning = int(hdr['CCDSUM'][0])

            if hdr.__contains__('DATASEC') & trim_overscan:
                dx, dy = iraf2python(hdr['DATASEC'])
            else:
                dx, dy = [0, hdr['NAXIS1']] , [0, hdr['NAXIS2']]
            amp = hdus[i].data[dy[0]:dy[1], dx[0]:dx[1]]
            ampsz = np.shape(amp)

            if mask is not None:
                msk = CCDData.read(mask, hdu=i, unit="adu").data[dy[0]:dy[1], dx[0]:dx[1]]
                amp[msk == 1] = np.nan

            # Creating ccd structure
            if (i == image_indices[0]):
                if inst_flag[0]: 
                    ccdsz = (ampsz[0]*2, ampsz[1]*2)
                elif inst_flag[1]:
                    ccdsz = (ampsz[0], ampsz[1]*n_ext + int(102/binning))
                img = np.empty(ccdsz)
                img[:] = np.nan

            # Copying amplifier data into the right section of the ccd
            if inst_flag[0]:
                sy = (i in (3,4))*ampsz[0]
                sx = (i in (2,4))*ampsz[1] 
            elif inst_flag[1]:
                sy = 0
                sx = (i-1)*ampsz[1] + (i > n_ext/2.)*int(102/binning)
            else:
                sx, sy = 0, 0
            img[ 0+sy : ampsz[0]+sy, 0+sx : ampsz[1]+sx ] = amp
            
    # To preserve details we should *really* downsample correctly and
    # not rely on matplotlib to do it correctly for us (it won't).

    # So, calculate the size of the figure in pixels, block_reduce to
    # roughly that,and display the block reduced image.

    # Thanks, https://stackoverflow.com/questions/29702424/how-to-get-matplotlib-figure-size
        
    fig_size_pix = fig.get_size_inches() * fig.dpi / [2.0, 2.0]
    ratio = (img.shape // fig_size_pix).max()

    if ratio < 1:
        ratio = 1
    ratio = input_ratio or ratio

    reduced_data = block_reduce(img, ratio)

    # Divide by the square of the ratio to keep the flux the same in the
    # reduced image. We do *not* want to do this for images which are
    # masks, since their values should be zero or one.
    if not is_mask:
        reduced_data = reduced_data / ratio**2

    # The image is a mask in which pixels should be zero or one.
    # block_reduce may have changed some of the values, so reset here.
    else:
        reduced_data = reduced_data > 0

    # Of course, now that we have downsampled, the axis limits are changed to
    # match the smaller image size. Setting the extent will do the trick to
    # change the axis display back to showing the actual extent of the image.
    extent = [0, img.shape[1], 0, img.shape[0]]
    norm = aviz.ImageNormalize(img, interval=interval_class, stretch=stretch_class)
    
    # Set the image scale limits appropriately.
    if is_mask:
        scale_args = dict(vmin=0, vmax=1)
    else:
        scale_args = dict(norm=norm)

    # Display the image
    if inst_flag[2]: maj_tick = 512
    else: maj_tick=256
        
    im = ax.imshow(reduced_data, origin='lower',
                   cmap=cmap, extent=extent, aspect='equal', **scale_args)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(maj_tick))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(maj_tick))

    if not show_labels:
        ax.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
        ax.tick_params(axis=u'both', which=u'both',length=0)
         
    if show_colorbar:
        sep, wid = 0.01, 0.03
        cax = fig.add_axes([ax.get_position().x1+sep,ax.get_position().y0,wid,ax.get_position().height])
        plt.colorbar(im, cax=cax)

    if show_title:
        obj = str(h0['object'])
        fig.suptitle(image+'  -  '+obj, fontsize=fontsize, y=0.92)
        
    hdus.close()
