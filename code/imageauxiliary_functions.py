import re
import numpy as np
from astropy.io import fits
from astropy.nddata import CCDData
from astropy.stats import mad_std
from multiprocessing.shared_memory import SharedMemory
from scipy.spatial.distance import mahalanobis


def image_extensions(image, is_hdu=False):
    
    if is_hdu:
        image_indices = np.arange(len(image))
        selection = [hdu.size > 0 for hdu in image]

    else: 
        with fits.open(image) as hdus:
            image_indices = np.arange(len(hdus))
            selection = [hdu.size > 0 for hdu in hdus]
    
    return image_indices[selection]


def get_filter_label(filter_list, 
                     ignore_filters=['open','no_filter','unavailable'],
                     ignore_labels=['s\d{4}','SAM.*','\dx\d','BTF.*']):

    #.Ignoring 'bad' filter keywords to get the filter label
    igvals = '|'.join(ignore_filters)
    p = re.compile(igvals,re.IGNORECASE)
    mask = [p.search(flt) == None for flt in filter_list]
    filter = [flt for flt,msk in zip(filter_list,mask) if msk]
    #.Splitting filter label into parts
    labels = re.split(r'[._\-\s]+', filter[0])
    #.Ignoring meaningless parts to compose final filter label
    igvals = '|'.join(ignore_labels)
    p = re.compile(igvals,re.IGNORECASE)
    mask = [p.search(lab) == None for lab in labels]
    labels = list(np.array(labels)[mask])
    #.Composing final label with leftover parts
    label = labels[0]
    if len(labels) > 1: label += '-'+''.join(labels[1:])

    return label


def ifc_filters(ifc, 
                obstype_selection='*',
                filter_keywords='filter1,filter2', 
                ignore_values='open,no_filter,unavailable'):

    
    if isinstance(filter_keywords, str): keywds = filter_keywords.split(',')
    else: keywds = filter_keywords
    igvalues = '|'.join(ignore_values.split(','))

    objects_ifc = ifc.filter(obstype=obstype_selection, regex_match=True)

    #.Obtendo uma lista com todos os possiveis filtros presentes no FileCollection
    filters = objects_ifc.values(keywds[0],unique=True)
    for i in np.arange(1,len(keywds)):
        filters.extend(objects_ifc.values(keywds[i],unique=True))

    #.Removendo itens duplicados nas duas (ou mais) rodas de filtros
    filters=list(set(filters))
    filters.sort()

    #..removendo filtros "abertos" (segundo o input dado em 'ignore_values')
    p = re.compile(igvalues,re.IGNORECASE)
    flag = [p.search(flt) == None for flt in filters]
    filters = [filt for filt,flg in zip(filters,flag) if flg]

    return filters


def image_filters(file_list, 
                  filter_keywords='filter1,filter2',
                  ignore_values='open,no_filter,unavailable'):
    
    filter_list=[]
    for file in file_list:

        hdr=fits.getheader(file, ext=0)
        if isinstance(filter_keywords, str): keywds = filter_keywords.split(',')
        else: keywds = filter_keywords
        filters=[hdr[key] for key in keywds]

        igvals = '|'.join(ignore_values.split(','))
        p = re.compile(igvals,re.IGNORECASE)
        flag = [p.search(flt) == None for flt in filters]
        filters = [filters for filters,flg in zip(filters,flag) if flg]
        filter_list.append(filters[0])
    
    return filter_list


def image_to_ccddata(image, is_hdu=False):

    ccddata=[]
    
    if is_hdu: hdul = image
    else: hdul = fits.open(image)

    for hdu in hdul:
        if hdu.name == 'MASK': continue
        if hdu.size == 0: data = ()
        else: data = hdu.data
        ccddata.append(CCDData(data, meta=hdu.header, unit="adu"))

    if not is_hdu: hdul.close()

    return ccddata


def center_mode(image_matrix, frac_size=0.5):

    #  Orientation
    ysz, xsz = image_matrix.shape
    yc, xc = int(ysz/2), int(xsz/2)                     # Center
    dy, dx = int(ysz*frac_size/2), int(xsz*frac_size/2) # Step

    #  Limits
    a, b = yc - dy, yc + dy
    c, d = xc - dx, xc + dy

    roi = image_matrix[a:b, c:d]
    med, mad = np.median(roi), mad_std(roi)
    hist, bin_edge = np.histogram(roi,bins='rice',range=(med-6*mad,med+6*mad))
    bin_cen = (bin_edge[:-1]+bin_edge[1:])/2

    return bin_cen[np.argmax(hist)]


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
            scales[j,i] = center_mode(hdus[idx].data, frac_size=0.90)

        hdus.close()
    
    scales = np.amax(scales, axis=1)
    
    if not normalize:
        scales /= scales[0]
        
    return scales


def find_filter(image_file, filter_list, filter_keywords):
    """
    Given an image and some filter keywords, search if any of these keywords
    have a filter matching the filter list. The function will return the index
    within the filter list where the match occurred.

    Arguments
    ---------
        image_file : string or list of strings
            filename of the image
        filter_list : list of strings
            possible filter names to match with the filter keywords
        filter_keywords : list of strings
            header keywords contining filter information

    Returns
    -------
        Index within the filter list, where the filter match occurred
    """

    if type(image_file) is not list: image_file = [ image_file ]

    if isinstance(filter_keywords, str): keywds = filter_keywords.split(',')
    else: keywds = filter_keywords
    filt_index=[]

    for image in image_file:

        hdr=fits.getheader(image, ext=0)
        for key in keywds:
            filt = hdr[key]
            if filt in filter_list:
                filt_index.append(filter_list.index(filt))
        
    return filt_index


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


def goodman_saturate(header):
    ccdsum = np.array([ float(bin) for bin in header['CCDSUM'].split() ])
    gain = header['GAIN']
    well_sat = np.prod(ccdsum)*0.8*205000/gain
    adc_sat = 0.8*65535
    header['SATURATE'] = np.min([well_sat, adc_sat])
    return(header)


def image_to_memory(input, name='shm'):

    if isinstance(input,str): hdul = fits.open(input)
    else: hdul = input

    image_exts = image_extensions(hdul, is_hdu=True)
    hdrs = np.array([hdul[ext].header.tostring() for ext in image_exts])
    data = np.stack([hdul[ext].data for ext in image_exts])
    
    shm_hdrs = SharedMemory(create=True, size=hdrs.nbytes, name=name+'_hdrs')
    shm_data = SharedMemory(create=True, size=data.nbytes, name=name+'_data')

    shm_hdrs_array = np.ndarray(hdrs.shape, hdrs.dtype, buffer=shm_hdrs.buf)
    shm_data_array = np.ndarray(data.shape, data.dtype, buffer=shm_data.buf)

    np.copyto(shm_hdrs_array, hdrs)
    np.copyto(shm_data_array, data)

    out = [{'id': name+'_hdrs', 'shape': hdrs.shape, 'dtype': hdrs.dtype}]
    out.append({'id': name+'_data', 'shape': data.shape, 'dtype': data.dtype})

    shm_hdrs.close()
    shm_data.close()

    if not hdul[0]._has_data: 
        hdr0 = np.array(hdul[0].header.tostring())
        shm_hdr0 = SharedMemory(create=True, size=hdr0.nbytes, name=name+'_hdr0')
        shm_hdr0_array = np.ndarray(hdr0.shape, hdr0.dtype, buffer=shm_hdr0.buf)
        np.copyto(shm_hdr0_array, hdr0)
        out.append({'id': name+'_hdr0', 'shape': hdr0.shape, 'dtype': hdr0.dtype})
        shm_hdr0.close()

    if isinstance(input,str): hdul.close()

    return out


def memory_to_ccddata(shm_dict):

    if len(shm_dict) > 2: 
        shm_hdr0 = SharedMemory(name=shm_dict[2]['id'])
        hdr0 = np.ndarray(shm_dict[2]['shape'], shm_dict[2]['dtype'], buffer=shm_hdr0.buf)
        ccddata = [(CCDData((), meta=fits.Header.fromstring(str(hdr0)), unit="adu"))]
        shm_hdr0.close()
    else: ccddata=[]
    
    shm_hdrs = SharedMemory(name=shm_dict[0]['id'])
    shm_data = SharedMemory(name=shm_dict[1]['id'])
    hdrs = np.ndarray(shm_dict[0]['shape'], shm_dict[0]['dtype'], buffer=shm_hdrs.buf)
    data = np.ndarray(shm_dict[1]['shape'], shm_dict[1]['dtype'], buffer=shm_data.buf)

    for i in range(shm_dict[0]['shape'][0]):
        ccddata.append(CCDData(data[i], 
                               meta=fits.Header.fromstring(str(hdrs[i])), 
                               unit='adu'))

    shm_hdrs.close()
    shm_data.close()

    return ccddata


def unlink_memory(shm_dict):

    for mem_dict in shm_dict:

        mem_block = SharedMemory(name=mem_dict['id']) 
        mem_block.close()
        mem_block.unlink()


def moffat2d(xy,I,x0,y0,a,b):
    """
    Definition of the two-dimensional Moffat intensity profile:
    
    f(r) = I * ( 1 + ((x-x0)/a)^2 + ((y-y0)/a)^2 )^-b

    where   x, y are the data pixel coordinates
            f(r) is the data intensity at each coordinate 
            x0, y0 are the source peak (centre) pixel coordinates
            I is the central intensity.
                for PDF normalization: I = (b-1)/(pi*a^2)
            a is the Moffat characteristic radius
                FWHM = 2.a.sqrt(2^(1/b) - 1)   - seeing
                r_50 = a.sqrt(2^(1/(b-1)) -1)  - half flux radius
            b is the Moffat characteristic exponent (~2.5-4.0)

    Arguments
    ---------
        (x,y) : 2D-array (2 x N)
            X, Y pixel coordinates of the source data
        x0, y0: float
            souce peak (centre) pixel coordinates
        I : float
            maximum intensity (central value) of the profile
        a : float
            Moffat characteristic radius of the profile.
        b : float
            Moffat characteristic exponent. For stellar objects it usually 
            falls in the 2.5-4.0 range. 

    Returns
    -------
        intensities : 1D-array
            the program will return a array of the calculated intensities at
            each supplied radius            
    """
    x, y = xy
    return I*( 1 + ((x-x0)/a)**2 + ((y-y0)/a)**2 )**(-b)


def moffatxy(xy,I,x0,y0,a,b,c,g):
    """
    Definition of the elliptical two-dimensional Moffat intensity profile:
    
    f(r) = I / ( 1 + mhd^2 )^g

    where   mhd is the Mahalanobis distance from the source center, calculated as:

            mhd = (XY - u0) x ICOV x (XY - u0)^t

            ICOV is the inverse of the covariance matrix defined as:
                COV = [[a^2, b.a.c]
                       [b.a.c, c^2]]

            XY are the data pixel coordinates vector
                XY = [x, y]
            u0 is the source peak (center) pixel coordinates vector
                u0 = (x0, y0)
            I is the central intensity.
                for PDF normalization: I = (b-1)/(pi*a^2)
            a,c are the elliptical Moffat characteristic radii in x,y axis
                FWHM = 2.a.sqrt(2^(1/b) - 1)   - seeing
                r_50 = a.sqrt(2^(1/(b-1)) -1)  - half flux radius
            b is the x,y correlation coefficient
            g is the Moffat characteristic exponent (~2.5-4.0)
            f(x,y) is the model source intensity at each coordinate 


    Arguments
    ---------
        (x,y) : 2D-array (2 x N)
            X, Y pixel coordinates of the source data
        x0, y0: float
            souce peak (centre) pixel coordinates
        I : float
            maximum intensity (central value) of the profile
        a : float
            Moffat characteristic radius of the profile (major-axis).
        c : float
            Moffat characteristic radius of the profile (minor axis).
        b : float
            correlation coefficient between a and c.
        g : float
            Moffat characteristic exponent. For stellar objects it usually 
            falls in the 2.5-4.0 range. 

    Returns
    -------
        intensities : 1D-array
            the program will return a array of the calculated intensities at
            each coordinate pair            
    """
    xy0 = np.array([x0,y0]).reshape((2,1))
    cov_mat = np.asmatrix([[a**2, b*a*c], [b*a*c, c**2]])
    cov_inv = np.linalg.inv(cov_mat)
    squared_mahalanobis = np.diag(np.dot(np.dot((xy-xy0).T,cov_inv),(xy-xy0)))

    return I/( 1 + squared_mahalanobis )**g
