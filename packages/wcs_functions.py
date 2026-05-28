import os, re
from copy import copy, deepcopy
import numpy as np
from datetime import datetime
from astropy.io import fits
from astropy.coordinates import Angle, SkyCoord
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats, mad_std
from astropy.coordinates import Angle
from astropy import units as u
from sklearn.neighbors import NearestNeighbors
from scipy.ndimage import median_filter, gaussian_filter
from photutils.detection import DAOStarFinder
from astroquery.vizier import Vizier
from astroquery.gaia import Gaia
from inspect import signature

from matplotlib import pyplot as plt
from packages.dataprocessing_functions import header_init

import warnings
from astropy.utils.exceptions import AstropyUserWarning
from astropy.wcs import WCS, FITSFixedWarning


def wcs_solve(image, 
        instrument='INSTRUME',
        match_radius=30,
        offset_max_shift=500,
        fit_plate_scale=True, 
        reset_wcs=False,
        SIP_header_file='SIP_FILE',
        **kwargs):

    #.Reading data
    if isinstance(image,str): hdu1 = fits.open(image, mode='update')
    else: hdu1 = image

    #.Getting image data (img) and header data (hdr) from the first fits extension
    img1 = hdu1[0].data 
    hdr1 = hdu1[0].header
    instrument = hdr1.get(instrument, default=instrument)

    #.Reseting WCS if needed
    if reset_wcs: 
        hdr1 = header_init(hdr1, instrument=instrument)
        hdr1.remove('WCSSOLVE', ignore_missing=True)

    #..aborting if WCS is already solved
    if isinstance(image,str): filename = image
    else: filename = os.path.basename(hdr1['FILENAME'])
    
    wcssolve = hdr1.get('WCSSOLVE', default=None)
    if wcssolve is not None:
        res_x, res_y = re.findall(r'\d+\.\d\d', wcssolve)
        n_fit = re.findall(r'N=\d+', wcssolve)[0]
        if float(res_x) < 1 and float(res_y) < 1:
            print(f'.WCS already solved for {filename}: MAE {res_x}, {res_y} pixels ({n_fit})')
            if isinstance(image,str): 
                hdu1.close()
                return
            else: return hdu1

    #.Reading important header keywords
    fwhm = hdr1.get('FWHM', default=10)
    exptime = hdr1['EXPTIME']
    ra  = Angle(hdr1['CRVAL1'], unit=u.degree)
    dec = Angle(hdr1['CRVAL2'], unit=u.degree)
    FoV = np.array([hdr1['NAXIS1']*np.sqrt(hdr1['CD1_1']**2 + hdr1['CD2_1']**2), 
                    hdr1['NAXIS2']*np.sqrt(hdr1['CD2_2']**2 + hdr1['CD1_2']**2)])*60

    #.Setting parameters based on instrument:
    if instrument.lower().find('sam') >= 0:
        kwargs['query_filters']         = {'Gmag': '< 20'}
        kwargs['query_geometry']        = {'width': f"{FoV[0]+1:.1f} arcmin", 'height': f"{FoV[1]+1:.1f} arcmin"}
        kwargs['offset_max_shift']      = 500
        kwargs['offset_distance_norm']  = [True, True]
        kwargs['offset_threshold']      = copy(match_radius)
        kwargs['offset_median_filter']  = None
        kwargs['rotation_threshold']    = copy(match_radius)
        kwargs['fit_plate_scale']       = False

    elif instrument.lower().find('goodman') >= 0:
        kwargs['query_filters']         = {'Gmag': '< 19'}
        kwargs['query_geometry']        = {'radius': f"{(FoV[0]+1)/2:.1f} arcmin"}
        kwargs['offset_max_shift']      = 200
        kwargs['offset_distance_norm']  = [True, False]
        kwargs['offset_threshold']      = [copy(match_radius), copy(match_radius)+10]
        kwargs['offset_median_filter']  = 6
        kwargs['rotation_threshold']    = copy(match_radius)
        kwargs['fit_plate_scale']       = True

    else:
        kwargs['query_filters']         = {'Gmag': '< 20','IPDfow': '< 1','sepsi': '< 2'},
        kwargs['query_geometry']        = {'width': f"{FoV[0]+1:.1f} arcmin", 'height': f"{FoV[1]+1:.1f} arcmin"}
        kwargs['offset_max_shift']      = offset_max_shift
        kwargs['fit_plate_scale']       = fit_plate_scale

    #.Processing SIP header file
    SIP_file = None
    if instrument.lower().find('sam') >= 0:
        if os.path.isfile(SIP_header_file): SIP_file = SIP_header_file
        elif SIP_header_file in hdr1:
            if os.path.isfile(hdr1[SIP_header_file]): SIP_file = hdr1[SIP_header_file]
            else: print('SIP file not found')

    if SIP_file is not None:
        hdr1 = SIP_file_to_header(SIP_file, hdr1, max_correction=None)
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FITSFixedWarning)
        wcs = WCS(hdr1, fix=False)

#======================================================
    #.Using sigma-clipping to model the background
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=AstropyUserWarning)
        back_mean, back_median, back_std = sigma_clipped_stats( img1, 
            cenfunc='median', stdfunc=mad_std,
            sigma_lower=3, sigma_upper=2, maxiters=3 )
    
    img_back = np.clip(img1 - back_median, a_min=0, a_max=None)

    if round(fwhm) <= 10: kernel_fwhm = 1.5*fwhm
    else: kernel_fwhm = fwhm

    daofinder = DAOStarFinder(3*back_std, kernel_fwhm,  
                              roundlo=-2.0, roundhi=2.0,
                              sharplo=0.01, sharphi=10.0,
                              exclude_border=True, peakmax=hdr1['SATURATE'])
    tab = daofinder.find_stars(img_back)#, mask=detection_mask)
    
    #.Aborting if no detections were found
    if not tab: nstar = 0
    else: nstar = len(tab)
    if nstar < 10: 
        print(f".WCS solving {filename}: not solved ({nstar} stars)")
        return

    #.Compiling detections table to compare with the catalog
    tab['mag'] += (25 + 2.5*np.log10(exptime))
    data_pix = np.transpose((tab['xcentroid'], tab['ycentroid'], tab['mag']))

#======================================================
    #.Querying online for the astrometric catalog

    # cat = query_Vizier(SkyCoord(ra=ra, dec=dec, frame='icrs'),
    #                             catalog=catalog, **kwargs)
    sig = signature(query_Gaia)
    query_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters.keys()}
    cat = query_Gaia(SkyCoord(ra=ra, dec=dec, frame='icrs'), **query_kwargs)

    ncat=len(cat)

    #..getting celestial positions of the catalog stars
    cat_pos = np.lib.recfunctions.structured_to_unstructured(np.array(cat))
    cat_mag = cat_pos[:,2].reshape(ncat,1)
    cat_pos = cat_pos[:,0:2]

    #..transforming positions to pixel values using the image WCS
    cat_pix = wcs.all_world2pix(cat_pos,0)
    cat_pix = np.hstack((cat_pix, cat_mag))

    #.matching catalogs density
    data_pix, cat_pix = cat_match_density(data_pix, cat_pix)

#======================================================
    #.finding the initial translation between the catalog and image stars
    sig = signature(cat_translation)
    offset_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters.keys()}
    xoff, yoff = cat_translation(data_pix, cat_pix, **offset_kwargs)

    #..updating offsets into the WCS
    hdr1 = wcs_update(hdr1, translation=np.array([xoff,yoff]))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FITSFixedWarning)
        wcs = WCS(hdr1, fix=False)
    #..updating catalog source positions to new WCS
    cat_pix = wcs.all_world2pix(cat_pos,0)
    cat_pix = np.hstack((cat_pix, cat_mag))

    #.finding the optimal rotation between the catalog and image stars
    #.(the 2nd,3rd iterations can reduce residuals by 50% each)
    sig = signature(cat_rotation)
    rotation_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters.keys()}

    for n in range(10):
        scale, rotation, translation, residuals, nmatches = cat_rotation(
          data_pix[:,0:2], cat_pix[:,0:2], **rotation_kwargs)
          
        if (np.isnan(scale) or np.any(np.isnan(rotation)) or np.any(np.isnan(translation))): break
        if (n > 2) and (nmatches < 20): break
       
        hdr1 = wcs_update(hdr1, translation=translation, rotation=rotation, scale=scale)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FITSFixedWarning)
            wcs = WCS(hdr1, fix=False)

        if np.all(residuals < 0.5): break

        cat_pix = wcs.all_world2pix(cat_pos,0)
        cat_pix = np.hstack((cat_pix, cat_mag))

        rotation_kwargs['rotation_threshold'] /= 1.25

#======================================================
    #.Printing
    print(f".WCS solving {filename}: MAE {residuals} pixels ({nmatches} stars)")

#======================================================
    #.Saving new header WCS to image
    get_date = datetime.now().strftime("%x %H:%M")

    if isinstance(residuals,str):
        if isinstance(image,str): 
            hdu1.close()

    else:
        wcsinf = f"{get_date} Catalog I/350: MAE {residuals[0]:.2f}, {residuals[1]:.2f} pixels (N={nmatches})"
        hdr1.set('WCSSOLVE', wcsinf)

        hdu1[0].header = hdr1
        if isinstance(image,str): 
            hdu1.writeto(image, overwrite=True)
            hdu1.close()
        else: return hdu1


def query_Gaia(coordinate, 
               query_filters={'Gmag': '< 20'},
               query_geometry={'width': '4 arcmin', 'height': '4 arcmin'},
               query_sort='Gmag'):
    
    filters = copy(query_filters)
    filters['phot_g_mean_mag'] = filters.pop('Gmag')
    columns = set(['RA','DEC','phot_g_mean_mag']+list(filters.keys()))
    
    Gaia.ROW_LIMIT = -1
    cat = Gaia.query_object(coordinate=coordinate, 
                            columns=columns,**query_geometry)
    
    for column,filter in filters.items():
        mask = eval(f"cat['{column}'] {filter}")
        if np.any(mask): cat = cat[mask]

    cat.rename_column('phot_g_mean_mag','Gmag')
    cat.rename_column('RA','RAJ2000')
    cat.rename_column('DEC','DEJ2000')
    cat.sort(query_sort)
    return cat['RAJ2000','DEJ2000',query_sort]


def query_Vizier(coordinate, query_catalog='I/350', 
                 query_filters={'Gmag': '< 20'}, 
                 query_geometry={'width': '4 arcmin', 'height': '4 arcmin'},
                 query_sort='Gmag',
                 server=None):
    
    Vizier.ROW_LIMIT = -1
    Vizier.TIMEOUT = 60                               
    if server is not None: Vizier.VIZIER_SERVER = server   

    query = Vizier.query_region(coordinate, 
                                catalog=query_catalog, 
                                column_filters=query_filters,
                                **query_geometry)

    cat=query[0]
    cat.sort(query_sort)
    return cat['RAJ2000','DEJ2000',query_sort]


def cat_match_density(dat, cat):
    
    #.gathering data information
    ndat = dat.shape[0]
    dat_min = np.nanmin(dat[:,0:2],axis=0)
    dat_max = np.nanmax(dat[:,0:2],axis=0)
    dat_size = np.round((dat_max-dat_min)/256)*256
    #.gathering catalog information
    ncat = cat.shape[0]
    cat_min = np.nanmin(cat[:,0:2],axis=0)
    cat_max = np.nanmax(cat[:,0:2],axis=0)
    cat_size = cat_max-cat_min
  
    #.checking for density differences between tables
    dat_den = ndat/np.prod(dat_size)
    cat_den = ncat/np.prod(cat_size)
    den_ratio = dat_den/cat_den

    #..removing faintest stars from the densest table
    if den_ratio > 1.5: 
        sort = np.argsort(dat[:,2])
        ndat = int(round(ndat/den_ratio))
        dat = dat[sort[:ndat],:]

    elif den_ratio < 0.75:
        sort = np.argsort(cat[:,2])
        ncat = int(round(ncat*den_ratio))
        cat = cat[sort[:ncat],:]
      
    return dat, cat


def cat_translation(dat, cat, 
                    offset_threshold=30,
                    offset_max_shift=250, 
                    offset_median_filter=None,
                    offset_distance_norm=[False, False], 
                    offset_figure=None):

    if isinstance(offset_threshold, int) or isinstance(offset_threshold,float):
        match_threshold = [offset_threshold]*2
    elif isinstance(offset_threshold,list):
        match_threshold = offset_threshold[0:2]
    
    #.finding rough offset solution
    xoffset, yoffset = cat_grid_offset(dat[:,0:2], cat[:,0:2], 
                                    match_threshold[0],
                                    grid_size=offset_max_shift,
                                    norm_box=offset_median_filter,
                                    use_distance=offset_distance_norm[0],
                                    figure=offset_figure)
    
    #.refining offset solution
    xoffset2, yoffset2 = cat_grid_offset(dat[:,0:2], cat[:,0:2],
                                    match_threshold[1], 
                                    grid_center=(xoffset,yoffset),
                                    grid_size=match_threshold[1]*1.414,
                                    grid_spacing=match_threshold[1]/10,
                                    smooth_sigma=1,
                                    use_distance=offset_distance_norm[1],
                                    figure=offset_figure)
    return xoffset2, yoffset2


def cat_grid_offset(dat, cat, match_radius, 
               grid_center=(0,0), 
               grid_size=(250,250), 
               grid_spacing=None,
               norm_box=None,
               smooth_sigma=None,
               use_distance=True,
               figure=None):
    
    if type(grid_size) is float: grid_size=(round(grid_size),round(grid_size))
    if type(grid_size) is int: grid_size=(grid_size, grid_size)
    if grid_spacing is None: grid_spacing = match_radius/2.
    if (norm_box == 0): norm_box = None

    #.building translation grid
    tgrid_x = np.arange(grid_center[0]-grid_size[0], 
                        grid_center[0]+grid_size[0], round(grid_spacing))
    tgrid_y = np.arange(grid_center[1]-grid_size[1], 
                        grid_center[1]+grid_size[1], round(grid_spacing))
    grid_x, grid_y = np.meshgrid(tgrid_x, tgrid_y)
    
    #.setting up diagnostic variables
    grid_shape = (len(tgrid_x), len(tgrid_y))
    nmatches = np.zeros(grid_shape)
    distance = np.full(grid_shape, float(match_radius*2))

    #.setting up Neighbors structure
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto')
    nbrs.fit(cat)

    #..finding the matches at each grid position
    for i in range(len(tgrid_x)):
        for j in range(len(tgrid_y)):
            xoff, yoff = tgrid_x[i], tgrid_y[j]
            dist, _ = nbrs.kneighbors(dat+[xoff, yoff])
            matches = dist <= match_radius
            nmatches[j,i] = np.count_nonzero(matches)
            if nmatches[j,i] > 1: distance[j,i] = np.median(dist[matches])

    if norm_box is not None:
        norm = median_filter(nmatches, size=norm_box)
        med_norm = np.nanmedian(norm)
        if med_norm != 0: 
            norm[norm == 0] = med_norm
            nmatches = nmatches/norm

    #..building goodness-of-fit indicator
    indicator = nmatches
    if use_distance: indicator /= distance
    if smooth_sigma is not None:
        indicator = gaussian_filter(indicator, sigma=smooth_sigma)

    #..obtaining optimal offsets in X and Y
    best_solution = np.where(indicator == np.nanmax(indicator))
    x_out, y_out = np.mean(grid_x[best_solution]), np.mean(grid_y[best_solution])

    #-----------------------------------------------------------------------------
    if figure:

        figure = plt.figure(figsize=(12,5))

        plotlabel = 'nmatches'
        if norm_box is not None: plotlabel = plotlabel+' sharpness'
        elif use_distance: plotlabel = plotlabel+' / mean nn-distance'
        
        #.Plotting colormesh of the number of matches for each offset    
        ax1 = figure.add_subplot(1,2,1)
        cm = ax1.pcolormesh(grid_x, grid_y, 
                            indicator, cmap='inferno_r')
        plt.colorbar(cm, label=plotlabel, ax=ax1)
        ax1.set_xlabel(r'X$_\mathrm{offset}$ (pix)')
        ax1.set_ylabel(r'Y$_\mathrm{offset}$ (pix)')
        ax1.scatter(x_out, y_out, marker='x', s=20, c='r')

        #.Plotting surface of the number of matches for each offset
        ax2 = figure.add_subplot(1,2,2, projection='3d', anchor='W')     
        ax2.plot_surface(grid_x, grid_y, indicator, cmap='inferno_r',
                        linewidth=0, antialiased=True)
        ax2.set_xlabel(r'X$_\mathrm{offset}$ (pix)')
        ax2.set_ylabel(r'Y$_\mathrm{offset}$ (pix)')

        plt.annotate(f"({x_out}, {y_out})",(0.40,0.05),xycoords='figure fraction')
    #-----------------------------------------------------------------------------

    return x_out, y_out


def cat_rotation(dat, cat, rotation_threshold, fit_plate_scale=True):

    #.matching the catalog with the translated image postions
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(cat)
    distance, index = nbrs.kneighbors(dat)
    #..selecting only matches with distance inferior to the selected threshold
    mask = (distance < rotation_threshold)
    distance, index = distance[mask], index[mask]

    #..saving the matched pairs in new arrays
    matched_data = dat[mask.flatten()]
    matched_cat = cat[index.flatten()]
    nmatches = len(matched_data)
    
    #..aborting if there are not enough matches
    if nmatches < 10: return np.nan, np.nan*np.identity(2), np.full(2,np.nan), 'not solved', nmatches
        
    #.Using Kabsh algoritm to find the optimal scaling and rotation of the data
    scale, rotation, translation = rigid_transform_3D(matched_cat, matched_data, scale=fit_plate_scale)
    # print("scale:",scale,"\n","translation:",translation,"\n","rotation:\n",rotation)

    #..calculating the residuals of the transformation:
    corrected_dat = scale*(rotation @ matched_data.T).T + translation
    res = corrected_dat-matched_cat
    mae = np.sum(abs(res), axis=0)/nmatches
    # mad = np.median(abs(res), axis=0)
    # var = np.sum(res**2, axis=0)/nmatches
    # rmse = np.sqrt(var)
    # rchi =  np.sqrt(np.sum(res**2/var, axis=0)/(nmatches-7))

    # print(f"transormation residuals (pixels): {mae} ({nmatches})")

    return scale, rotation, translation, mae, nmatches


def rigid_transform_3D(A, B, scale=True):

    assert len(A) == len(B)
    
    N = A.shape[0];  # total points
    
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    # center the points
    AA = A - np.tile(centroid_A, (N, 1))
    BB = B - np.tile(centroid_B, (N, 1))

    # @ is matrix multiplication for array
    if scale:
        H = np.transpose(BB) @ AA / N
    else:
        H = np.transpose(BB) @ AA

    U, S, Vt = np.linalg.svd(H)

    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[1, :] *= -1
        R = Vt.T @ U.T

    if scale:
        varA = np.var(A, axis=0).sum()
        c = 1 / (1 / varA * np.sum(S))  # scale factor
        t = -R @ (centroid_B.T * c) + centroid_A.T
    else:
        c = 1
        t = -R @ centroid_B.T + centroid_A.T

    return c, R, t


def wcs_update(header, translation=(0,0), rotation=np.identity(2), scale=1, keep_origin=True):
   
    #.Saving tangent point coordinates (pixel)
    tan_x, tan_y = header['CRPIX1'], header['CRPIX2']
    
    #.Loading WCS coefficients from header
    CRpix = np.array([header['CRPIX1'], header['CRPIX2']])
    CD_matrix = np.array([ [header['CD1_1'], header['CD1_2']],
                           [header['CD2_1'], header['CD2_2']] ])
    
    #.Calculating new WCS coefficients from the transformations
    CD_update = scale*(CD_matrix @ rotation)
    inverse_rotation = np.linalg.inv(rotation)
    CRpix_update = inverse_rotation @ (CRpix - translation) / scale

    #.Updating header with new WCS values
    header['CD1_1'] = CD_update[0,0]
    header['CD1_2'] = CD_update[0,1]
    header['CD2_1'] = CD_update[1,0]
    header['CD2_2'] = CD_update[1,1]
    header['CRPIX1'] = CRpix_update[0]
    header['CRPIX2'] = CRpix_update[1]
    
    #.Tangent point probably has changed. Returning to the original location
    if keep_origin: 
        header = tangent_shift(header, (tan_x, tan_y))

    return header


def tangent_shift(header, new_crpix, is_shift=False):
    ncrpix1, ncrpix2 = new_crpix

    if is_shift: 
        dx = ncrpix1
        dy = ncrpix2
        ncrpix1 += header["CRPIX1"]
        ncrpix2 += header["CRPIX2"]
    else: 
        dx = ncrpix1 - header["CRPIX1"]
        dy = ncrpix2 - header["CRPIX2"]

    CD_matrix = np.array([  [header["CD1_1"], header["CD1_2"]], 
                            [header["CD2_1"], header["CD2_2"]]  ])
    d_alpha, d_delta = CD_matrix @ np.array([dx,dy])
    d_alpha /= np.cos(header["CRVAL2"]*np.pi/180)

    header["CRPIX1"] = ncrpix1
    header["CRPIX2"] = ncrpix2
    header["CRVAL1"] += d_alpha
    header["CRVAL2"] += d_delta

    return header
    

def SIP_file_to_header(sip_file, header, max_correction=70):

    with open(sip_file) as file:

    #.Reading SIP coefficients from file
        dt=np.dtype([('keyword', 'U8'), ('value', 'f8'),('comment','U100')])
        tab = np.genfromtxt(file, comments='#', delimiter=',', dtype=dt)
    #.Writing SIP coefficients to header
        for row in tab: header.set(*row)
    #.Fixing other header keywords
        header = tangent_shift(header,(0,2056))
        header['CRPIX1'] = 0 - header['A_0_0']
        header['CRPIX2'] = 2056 - header['B_0_0']
        header.remove('A_0_0')
        header.remove('B_0_0')
        header['A_ORDER'] = int(header['A_ORDER'])
        header['B_ORDER'] = int(header['B_ORDER'])
        if "-SIP" not in header['CTYPE1']: header['CTYPE1'] += "-SIP"
        if "-SIP" not in header['CTYPE2']: header['CTYPE2'] += "-SIP"
    #.Setting up maximum SIP correction
        if max_correction is not None:
            if np.ndim(max_correction) == 0: max_correction = (max_correction, max_correction)
            header.set("A_DMAX", max_correction[0], "X maximum correction [pixel]")
            header.set("B_DMAX", max_correction[1], "Y maximum correction [pixel]")
    return header


