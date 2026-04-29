import os, re
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

from matplotlib import pyplot as plt

import warnings
from astropy.utils.exceptions import AstropyUserWarning
from astropy.wcs import WCS, FITSFixedWarning


def wcs_solve(image, 
		catalog='I/350', cat_magnitude='Gmag', 
		cat_constraints={'Gmag': '< 20','IPDfow': '< 1','sepsi': '< 2'},
		cat_max_objects=150,
        fit_scale_plate=False, 
        match_radius=30,
        max_shift=500,
        SIP_header_file=os.path.join('packages','SAMI_SIP_coefficients.txt'),
        figure=None):

    #.Reading data
    if isinstance(image,str): hdu1 = fits.open(image, mode='update')
    else: hdu1 = image

    #.Getting image data (img) and header data (hdr) from the first fits extension
    img1 = hdu1[0].data 
    hdr1 = hdu1[0].header

    #..aborting if WCS is already solved
    if isinstance(image,str): filename = image
    else: filename = os.path.basename(hdr1['FILENAME'])

    wcssolve = hdr1.get('WCSSOLVE', default=None)
    if wcssolve is not None:
        res_x, res_y = re.findall(r'\d+\.\d\d', wcssolve)
        if float(res_x) < 3 and float(res_y) < 3:
            print(f'.WCS already solved for {filename}: MAE {res_x}, {res_y} pixels')
            if isinstance(image,str): 
                hdu1.close()
                return
            else: return hdu1

    #..reading important header keywords
    fwhm = hdr1.get('FWHM', default=10)
    exptime = hdr1['EXPTIME']
    ra  = Angle(hdr1['CRVAL1'], unit=u.degree)
    dec = Angle(hdr1['CRVAL2'], unit=u.degree)
    FoV = np.array([hdr1['NAXIS1']*np.sqrt(hdr1['CD1_1']**2 + hdr1['CD2_1']**2), 
                    hdr1['NAXIS2']*np.sqrt(hdr1['CD2_2']**2 + hdr1['CD1_2']**2)])*60
    instrument = hdr1['INSTRUME']

    #.Processing SIP header file
    SIP_file = None
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
    # total_error = np.sqrt(back_std**2 + (img_back)/hdr1['GAIN'])
    # detection_mask = (img_back) < 5*total_error

    #.Detecting sources using IRAFStarfinder method
    # (actually, DAOFinder is faster than IRAFStarFinder)
    # (find_peaks is even faster, but unreliable)

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

    #.Querying Vizier for the catalog
    Vizier.ROW_LIMIT = -1
    Vizier.VIZIER_SERVER = "vizier.cfa.harvard.edu"   # more stable than default
    Vizier.TIMEOUT = 60                               # helps if the server is slow

    if instrument.lower().find('goodman') >= 0:
        geometry_kwd = {'radius': f"{(FoV[0]+1)/2:.1f} arcmin"}
    else:
        geometry_kwd = {'width': f"{FoV[0]+1:.1f} arcmin", 'height': f"{FoV[1]+1:.1f} arcmin"}

    query = Vizier.query_region(SkyCoord(ra=ra, dec=dec, frame='icrs'),
                                catalog=catalog, column_filters=cat_constraints,
                                **geometry_kwd)

    cat=query[0]
    cat.sort(cat_magnitude)
    ncat=len(cat)

    #..getting celestial positions of the catalog stars
    cat_pos = np.array([cat['RAJ2000'],cat['DEJ2000']]).T

    #..transforming positions to pixel values using the image WCS
    cat_pix = wcs.all_world2pix(cat_pos,0)

    #..adding a magnitude column to the catalog stars
    if cat_magnitude in cat.colnames: 
        cat_mag = np.array(cat[cat_magnitude].data).reshape(ncat,1)
    else: cat_mag = np.full(ncat,1)
    cat_pix = np.hstack((cat_pix, cat_mag))
    
    #.matching catalogs density
    data_pix, cat_pix = cat_match_density(data_pix, cat_pix)

    # #..cutting the faintest objects to enforce the number limit
    # nlim = int(cat_max_objects*np.prod(FoV+1)/np.prod(FoV))
    # if ncat > nlim: 
    #     cat_pix = cat_pix[0:nlim-1,:]
    #     data_pix = data_pix[0:nlim-1,:]
    # # nlim = np.min([ncat,nlim])

#======================================================
    
    #.finding the initial translation between the catalog and image stars
    xoff, yoff = cat_translation(data_pix, cat_pix, 
                                 match_radius=match_radius,
                                 max_shift=max_shift, 
                                 instrument=instrument, figure=figure)
    hdr1 = wcs_update(hdr1, translation=np.array([xoff,yoff]))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FITSFixedWarning)
        wcs = WCS(hdr1, fix=False)
    cat_pix = wcs.all_world2pix(cat_pos,0)
    cat_pix = np.hstack((cat_pix, cat_mag))

    #.finding the optimal rotation between the catalog and image stars
    #.(the 2nd,3rd iterations can reduce residuals by 50% each)
    match_threshold = match_radius
    for n in range(4):
        scale, rotation, translation, residuals, nmatches = cat_rotation(
          data_pix[:,0:2], cat_pix[:,0:2], 
          match_threshold, fit_scale_plate=fit_scale_plate)
          
        if (np.isnan(scale) or np.any(np.isnan(rotation)) or np.any(np.isnan(translation))): break
       
        hdr1 = wcs_update(hdr1, translation=translation, rotation=rotation, scale=scale)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FITSFixedWarning)
            wcs = WCS(hdr1, fix=False)
        cat_pix = wcs.all_world2pix(cat_pos,0)
        cat_pix = np.hstack((cat_pix, cat_mag))
        match_threshold /= 1.5

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
        wcsinf = f"{get_date} Catalog {catalog}: MAE {residuals[0]:.2f}, {residuals[1]:.2f} pixels (N={nmatches})"
        hdr1.set('WCSSOLVE', wcsinf)

        hdu1[0].header = hdr1
        if isinstance(image,str): 
            hdu1.writeto(image, overwrite=True)
            hdu1.close()
        else: return hdu1


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


def cat_translation(dat, cat, match_radius=30, max_shift=500, instrument='SAMI', figure=None):

    if instrument.lower().find('sam') >= 0:
        usedist=[True, True]
    elif instrument.lower().find('goodman') >= 0:
        usedist=[True, False]
    else: 
        usedist=[False, False]

    #.finding rough offset solution
    xoffset, yoffset = cat_grid_offset(dat[:,0:2], cat[:,0:2], 
                                    match_radius,
                                    grid_size=max_shift,
                                    norm_box=6,
                                    use_distance=usedist[0],
                                    figure=figure)
    #.refining offset solution
    xoffset, yoffset = cat_grid_offset(dat[:,0:2], cat[:,0:2],
                                    match_radius, 
                                    grid_center=(xoffset,yoffset),
                                    grid_size=match_radius*1.414,
                                    grid_spacing=match_radius/10,
                                    smooth_sigma=True,
                                    use_distance=usedist[1],
                                    figure=figure)
    return xoffset, yoffset


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
    if grid_spacing is None: grid_spacing = match_radius
    if (norm_box == 0): norm_box = None

    #.building translation grid
    tgrid_x = np.arange(grid_center[0]-grid_size[0], grid_center[0]+grid_size[0], grid_spacing)
    tgrid_y = np.arange(grid_center[1]-grid_size[1], grid_center[1]+grid_size[1], grid_spacing)
    grid_x, grid_y = np.meshgrid(tgrid_x, tgrid_y)
    
    #.setting up diagnostic variables
    grid_shape = (len(tgrid_x), len(tgrid_y))
    nmatches = np.zeros(grid_shape)
    distance = np.full(grid_shape, match_radius*2)

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
    if figure is not None:

        figure = plt.figure(figsize=(12,5))

        plotlabel = 'nmatches'
        if norm_box is not None: plotlabel = plotlabel+' sharpness'
        elif use_distance: plotlabel = plotlabel+' / mean nn-distance'
        
        #.Plotting colormesh of the number of matches for each offset    
        ax1 = figure.add_subplot(1,2,1)
        cm = ax1.pcolormesh(grid_x-grid_spacing/2, grid_y-grid_spacing/2, 
                            indicator, cmap='inferno_r')
        plt.colorbar(cm, label=plotlabel, ax=ax1)
        ax1.set_xlabel(r'X$_\mathrm{offset}$ (pix)')
        ax1.set_ylabel(r'Y$_\mathrm{offset}$ (pix)')
        ax1.scatter(x_out-grid_spacing/2, y_out-grid_spacing/2, marker='x', s=20, c='r')

        #.Plotting surface of the number of matches for each offset
        ax2 = figure.add_subplot(1,2,2, projection='3d', anchor='W')     
        ax2.plot_surface(grid_x, grid_y, indicator, cmap='inferno_r',
                        linewidth=0, antialiased=True)
        ax2.set_xlabel(r'X$_\mathrm{offset}$ (pix)')
        ax2.set_ylabel(r'Y$_\mathrm{offset}$ (pix)')
    #-----------------------------------------------------------------------------

    return x_out, y_out


def cat_rotation(dat, cat, match_threshold, fit_scale_plate=True):

    #.matching the catalog with the translated image postions
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(cat)
    distance, index = nbrs.kneighbors(dat)
    #..selecting only matches with distance inferior to the selected threshold
    mask = (distance < match_threshold)
    distance, index = distance[mask], index[mask]

    #..saving the matched pairs in new arrays
    matched_data = dat[mask.flatten()]
    matched_cat = cat[index.flatten()]
    nmatches = len(matched_data)
    
    #..aborting if there are not enough matches
    if nmatches < 3: return np.nan, np.nan*np.identity(2), np.full(2,np.nan), 'not solved', nmatches
        
    #.Using Kabsh algoritm to find the optimal scaling and rotation of the data
    scale, rotation, translation = rigid_transform_3D(matched_cat, matched_data, scale=fit_scale_plate)
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

