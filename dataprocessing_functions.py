import os, sys, glob
from datetime import datetime
import numpy as np
from astropy.io import fits
from astropy.nddata import CCDData
from astropy.coordinates import Angle
from astropy.time import Time
from astropy import units as u
from astropy.table import vstack
from photutils.detection import DAOStarFinder
from scipy.optimize import curve_fit
from functools import partial
from multiprocessing import Pool
from multiprocessing.shared_memory import SharedMemory
from ccdproc import ImageFileCollection, subtract_overscan, trim_image, subtract_bias, flat_correct, cosmicray_lacosmic, Combiner

import imageauxiliary_functions as iaf

import warnings
from photutils.detection.daofinder import NoDetectionsWarning
from scipy.optimize import OptimizeWarning
from ccdproc.image_collection import AstropyUserWarning


def header_setup(dataset, instrument='SAMI', multiprocessing=False):

    #.If input dataset is a folder, expand it into a ImageFileCollection
    if isinstance(dataset,str): 
        ifc = ImageFileCollection(dataset, ext=0,
         keywords=['obstype','ut','ccdsum','airmass','exptime','object'],
         glob_exclude="*master*.fits, bpm*.fits", glob_include="*.fits")
        
    #.Otherwise the input dataset is already a ImageFileCollection
    else: ifc = dataset
    image_list = ifc.files_filtered(include_path=True)

    if multiprocessing:
        with Pool() as pool:
            pool.map(partial(header_image, instrument=instrument), 
                     image_list)

    else: 
        for file in image_list:
            header_image(file, instrument=instrument)
    return


def header_image(file, instrument='SAMI'):

    hdul = fits.open(file, mode='update')
    image_exts = iaf.image_extensions(hdul, is_hdu=True)
    for ext in image_exts:
        hdr = hdul[ext].header
        hdr = header_remove_duplicate(hdr)
        hdr = header_init(hdr, instrument=instrument)
        hdul[ext].header = hdr
    hdul.close()


def header_remove_duplicate(header, 
                            duplicated_keywords=None, 
                            allowed_duplicates = ('COMMENT', 'HISTORY', 'HIERARCH')):

    if duplicated_keywords is None:

        uniq_keywords = []
        add_to_keywords = uniq_keywords.append

        header_keywords = list(header.keys())
        for keyw in header_keywords:

            if (keyw in uniq_keywords) and (keyw not in allowed_duplicates):
                header.remove(keyw)
            else:
                add_to_keywords(keyw)

    else: 
        for keyw in duplicated_keywords:
            header.remove(keyw)

    return header


def header_init(header, instrument='SAMI'):

    header.set("CTYPE1", "RA---TAN", "Coordinate type")
    header.set("CTYPE2", "DEC--TAN", "Coordinate type")
    header.set("RADESYSa", "FK5", "Default coordinate system", before="CTYPE1")
    header.set("CUNIT2", "deg", "Coordinate unit", after="CTYPE1")
    header.set("CUNIT1", "deg", "Coordinate unit", after="CTYPE1")
    header.set("EQUINOX", 2000., "Equinox of WCS")

    if "RADECSYS"in header: del header["RADECSYS"]
    if "WCSASTRM" in header: del header["WCSASTRM"]
    if "RADECEQ" in header: del header["RADECEQ"]
    if "WAT0_001" in header: del header["WAT0_001"]
    if "WAT1_001" in header: del header["WAT1_*"]
    if "WAT2_001" in header: del header["WAT2_*"]
    if "PC1_1" in header: del header["PC1_*"]
    if "PC2_1" in header: del header["PC2_*"]

    if instrument == "SAMI":
        ccdsum = np.array([ float(bin) for bin in header['CCDSUM'].split() ])
        point_ra = Angle(header['TELRA'], unit = u.hourangle).value*15
        point_dec = Angle(header['TELDEC'], unit = u.degree).value
        crpix1 = 2048/ccdsum[0]
        crpix2 = 2056/ccdsum[1]
        CDval = 1.25e-05*ccdsum
        camrot = header['ROTOFFS']
        header["GAIN"] = 2.1
        header["RDNOISE"] = 4.7
        header["SATURATE"] = 0.8*65536
        header.set("SIP_FILE",'SAMI_SIP_coefficients.txt','Higher order WCS corrections',
                   before='CDELT1')

    elif instrument == "Goodman":
        if "PG0_0" in header: del header["PG*"]
        if "N_PRM_0" in header: del header["N_PRM*"]
        ccdsum = np.array([ float(bin) for bin in header['CCDSUM'].split() ])
        point_ra = Angle(header['RA'], unit = u.hourangle).value*15
        point_dec = Angle(header['DEC'], unit = u.degree).value
        crpix1 = 1548/ccdsum[0]
        crpix2 = 1548/ccdsum[1]
        CDval = 4.012792e-05*ccdsum
        camrot = header['POSANGLE']
        header = iaf.goodman_saturate(header)
        header.set('MJD-OBS', Time(header['DATE-OBS']).mjd,"MDJ at start of observation")
        
    elif instrument == "SOI":
        ccdsum = np.array([ float(bin) for bin in header['CCDSUM'].split() ])
        point_ra = Angle(header['TELRA'], unit = u.hourangle).value*15
        point_dec = Angle(header['TELDEC'], unit = u.degree).value
        crpix1 = (2048+102)/ccdsum[0]
        crpix2 = 2048/ccdsum[1]
        CDval = 2.1306e-05*ccdsum
        camrot = header['DECPANGL']
        header["GAIN"] = 2.2
        header["RDNOISE"] = 4.7
        header["SATURATE"] = 55000

    header["CDELT1"] = CDval[0]
    header["CDELT2"] = CDval[1]
    header["CRVAL1"] = point_ra
    header["CRVAL2"] = point_dec
    header["CRPIX1"] = crpix1
    header["CRPIX2"] = crpix2

    cost = np.cos(camrot * np.pi/180)
    sent = np.sin(camrot * np.pi/180)
    header["CD1_1"] = CDval[0] * cost
    header["CD1_2"] = abs(CDval[1]) * np.sign(CDval[0]) * sent
    header["CD2_1"] = -abs(CDval[0]) * np.sign(CDval[1]) * sent
    header["CD2_2"] = CDval[1] * cost

    return header


def reduce_image(image_file, 
                 master_bias=None, 
                 master_flat=None,
                 shared_memory=False,
                 merge_amplifiers=True):

    #.opening image FITS file
    hdul = fits.open(image_file)
    image_exts = iaf.image_extensions(hdul, is_hdu=True)

    get_date = datetime.now().strftime("%b %d %Y %H:%M")
    fstr = os.path.basename(image_file)
    
    #.setting instrument
    inst = hdul[0].header['INSTRUME']
    
    #..preparing master bias data
    if master_bias is not None:
        if shared_memory:
            mbias_ccd = iaf.memory_to_ccddata(master_bias)
            mbias_buf = [SharedMemory(name=master_bias[i]['buffer']) for i in range(len(master_bias))]
        else: mbias_ccd = master_bias
        bstr = os.path.basename(mbias_ccd[0].header['FILENAME'])

    #.preparing master flats data
    if master_flat is not None:
        if shared_memory: 
            mflat_ccd = iaf.memory_to_ccddata(master_flat)
            mflat_buf = [SharedMemory(name=master_flat[i]['buffer']) for i in range(len(master_flat))]
        else: mflat_ccd = master_flat
        flstr = os.path.basename(mflat_ccd[0].header['FILENAME'])

    #.preparing to merge amplifiers
    can_merge = (len(image_exts) > 1) and merge_amplifiers and (hdul[0].header['OBSTYPE']=='OBJECT')
    if can_merge:
         #..collecting binning and size of the image
        ccdsum = hdul[image_exts[0]].header['CCDSUM']
        bin_x, bin_y = int(ccdsum[0]), int(ccdsum[2])
        imsz_x, imsz_y = iaf.iraf2python(hdul[image_exts[0]].header['DETSIZE'])
        imsz_x, imsz_y = int(imsz_x[1]/bin_x), int(imsz_y[1]/bin_y)
        #..SOI gap
        if inst.find('SOI') >= 0: 
            xgap = np.array([0,0,102/bin_x,102/bin_x],dtype=int)
            ygap = np.array([0,0,0,0], dtype=int)
        else: xgap, ygap = np.full(4,0), np.full(4,0)

        #..initializing output image
        img_merge = np.full((imsz_y+np.amax(ygap), imsz_x+np.amax(xgap)), np.nan, dtype=np.float32)

    #.Loop over the image extensions (amplifiers)
    proc_string = f".Processing {fstr:1s}"
    skip = np.full(len(image_exts),True)
    for ne,ext in enumerate(image_exts):

        #.Reading data for this extension
        proc_string+=f" [{ext:1.0f}]"
        img = CCDData(hdul[ext].data, meta=hdul[ext].header, unit="adu")
        
        #.OVERSCAN correction
        if ('BIASSEC' in img.header):
            proc_string+="o"

            biassec = img.header['BIASSEC']        
            img = subtract_overscan(img, 
                    fits_section=biassec, 
                    model=None, median=True,
                    add_keyword={'overscan': f"{get_date} Overscan is {biassec}; model=median"})

            #.SOI exception
            #.(wrong TRIMSEC keyword in header)
            trimsec = img.header['TRIMSEC']
            if (inst.find('SOI') >= 0):
                if (ext in [1,3]): trimsec = '[29:540,1:2048]'
                else: trimsec = '[28:539,1:2048]'

            img = trim_image(img, 
                    fits_section=trimsec,
                    add_keyword={'trim': f"{get_date} Trim is {trimsec}"})
            
            #.Updating header
            imsz = img.shape
            img.header['DATASEC'] = f"[1:{imsz[1]},1:{imsz[0]}]"
            del img.header['BIASSEC']
            del img.header['TRIMSEC']
            del img.header['BZERO']
            del img.header['BSCALE']

            skip[ne] = False

        #.Goodman exception
        #.(do not have an overscan section)
        elif inst.find('Goodman') >= 0:
            proc_string+="o"

            img.header['BITPIX']=-32
            del img.header['BZERO']
            del img.header['BSCALE']
            img.header['OVERSCAN']=f"{get_date} No overscan. Changing datatype to float"

            skip[ne] = False
        else: 
            proc_string+="-"


        #.BIAS correction
        if ('ZEROCOR' not in img.header) and (master_bias is not None): 
            proc_string+="z"
            img = subtract_bias(img, 
                mbias_ccd[ext],
                add_keyword={'ZEROCOR': f"{get_date} Zero is {bstr}[{ext}]"})

            skip[ne] = False
        else:
            proc_string+="-"

        #.FLAT FIELD correction
        if ('FLATCOR' not in img.header) and (master_flat is not None):
            proc_string+="f"
            fscl = mflat_ccd[ext].header['FLATNORM']
            img = flat_correct(img, 
                mflat_ccd[ext], 
                norm_value=1,
                add_keyword={'FLATCOR': f"{get_date} Flat is {flstr}[{ext}] (norm={fscl:.1f})"})

            skip[ne] = False
        else:
            proc_string+="-"

        #.Merging amplifiers
        if can_merge:
            proc_string+="m"
            #..collecting size and relative position of this amplifier
            ampos_x, ampos_y = iaf.iraf2python(img.header['DETSEC'])
            ampos_x = (np.array(ampos_x)/bin_x).astype(int) + xgap[ne]
            ampos_y = (np.array(ampos_y)/bin_y).astype(int) + ygap[ne]
            
            #..writing this amplifier data into the output CCDDATA object
            img_merge[ampos_y[0]:ampos_y[1],ampos_x[0]:ampos_x[1]] = img.data
        else:
            proc_string+="-"

        #.If any operation was done, update data and header in the HDU
        if not skip[ne]: 
            hdul[ext].data = img.data.astype(np.float32)
            hdul[ext].header = img.header
    
    #.Saving the processed image 
    if can_merge:
        
        #..preparing header of the output image
        if image_exts[0] != 0: 
            hdr = hdul[0].header
            hdr.extend(hdul[1].header, unique=True)
        else: hdr = hdul[1].header
        #..removing keywords no longer needed
        keystodel = ['DATASEC', 'CCDSEC', 'AMPSEC', 'DETSEC', 'NEXTEND', 'EXTNAME']
        for keyw in keystodel: del hdr[keyw]
        #..adjusting keywords to the new image format
        hdr['DETSIZE']=f"[1:{imsz_x},1:{imsz_y}]"
        hdr['CCDSIZE']=f"[1:{imsz_x},1:{imsz_y}]"
        hdr.insert('NAXIS', ('NAXIS1', imsz_x, "Axis length"), after=True)
        hdr.insert('NAXIS1', ('NAXIS2', imsz_y, "Axis length"), after=True)   
        hdr.append(('AMPMERGE', f"{get_date} Merged {len(image_exts)} amps"))

        #..creating output image CCDDATA object
        img = CCDData(img_merge, meta=hdr, unit='adu')
        #..writing to file (overwrite)
        img.write(image_file,hdu_mask=None,hdu_uncertainty=None,overwrite=True)
        
    else:
        if np.sum(skip) is not len(image_exts): 
            hdul.writeto(image_file, overwrite=True)
        hdul.close()
    print(proc_string, flush=True)


def reduce_image_mp(image_file, flat_idx, 
                    master_bias, 
                    master_flats_list,
                    merge_amplifiers=True):
    
    #.selecting the proper 'master_flat' (FILTER) for each science image
    master_flat = master_flats_list[flat_idx]

    #.reducing the image
    reduce_image(image_file, 
                  master_bias = master_bias,
                  master_flat = master_flat,
                  shared_memory = True,
                  merge_amplifiers = merge_amplifiers)


def process_bias(ifc, 
                 combined_bias='master_bias.fits',
                 delete_bias=True,
                 multiprocessing=True):
    
    #.skipping if master bias is found in folder
    with warnings.catch_warnings():
        warnings.simplefilter('error', AstropyUserWarning)
        try: 
            bias_ifc = ifc.filter(obstype='BIAS|ZERO', regex_match=True)
        except AstropyUserWarning:
            print(f".ABORTING 'master bias' creation: no BIAS images found")
            return

    #.grouping BIAS images into a list 
    file_list = bias_ifc.files
    bias_ifc = ifc.filter(obstype='BIAS|ZERO', regex_match=True)
    file_list = bias_ifc.files
    
    #.OVERSCAN correction
    if multiprocessing: 
        with Pool() as pool:
            pool.map(partial(reduce_image, master_bias=None, master_flat=None,
                             merge_amplifiers=False), file_list)
    else: 
        for file in file_list: 
            reduce_image(file)

    #.combining BIAS frames into a 'master_bias'
    combine_bias(file_list, delete_images=delete_bias, output=combined_bias, 
                 multiprocessing=multiprocessing)

    ifc.refresh()
    

def combine_bias(image_list, 
                 delete_images=True, 
                 output='master_bias.fits',
                 multiprocessing=True):
    
    #.aborting if there are no BIAS images or if a 'master_bias' is found
    nbias = len(image_list)
    if (not os.path.isfile(output)):
        if (nbias > 0): 
            print(f".Creating '{os.path.basename(output)}': {nbias} images")
        else: sys.exit(".ABORTING 'master bias' creation: no BIAS images found")
    else: 
        print(f".Using '{os.path.basename(output)}' image found in directory")
        return

    #.creating output image
    mbias = fits.open(image_list[0])

    #.getting image extensions
    image_exts = iaf.image_extensions(mbias, is_hdu=True)

    #.combining BIAS (per extension)
    if multiprocessing:
        with Pool() as pool:
            result = pool.map(partial(combine_bias_extension, bias_list=image_list), 
                              image_exts)
    else:
        result=[]
        for ext in image_exts:
            result.append(combine_bias_extension(ext, bias_list=image_list))
    
    #.joining combined extensions into the output image
    for res in result:
        ext = res[0]
        mbias[ext].data = res[1].data
        mbias[ext].header = res[1].header

    #.saving output image
    mbias[0].header['FILENAME']=output
    mbias.writeto(output, overwrite=True)
    mbias.close()

    #.deleting individual bias frames
    if delete_images: 
        for file in image_list: 
            try: os.remove(file)
            except OSError as e: print("Error: %s - %s." % (e.filename, e.strerror))


def combine_bias_extension(ext, bias_list):

    #..grouping BIAS images from this amplifier
    ccd_list = [ CCDData.read(bias_list[j], hdu=ext, unit="adu")
                 for j in np.arange(len(bias_list)) ]
    
    #..preparing combiner object
    comb = Combiner(ccd_list, dtype=np.float32)
    comb.sigma_clipping(low_thresh=3, high_thresh=3, func='median', dev_func='mad_std')
    #..average combining
    comb_bias = comb.average_combine()

    #..creating and updating output header
    comb_bias.header = fits.getheader(bias_list[0], ext=ext)
    for n,imgn in enumerate(bias_list, start=1): 
        fstr = os.path.basename(imgn)
        comb_bias.header.append((f"IMCMB{n:03}", f"{fstr}[{ext}]"))
    comb_bias.header.append(('NCOMBINE', len(ccd_list), '# images combined'))

    #..returning the combined image from this extension
    return (ext, comb_bias)


def process_flat(ifc, 
                 filter_keywords='filter1,filter2', 
                 master_bias='master_bias.fits',
                 combined_flats='master_flat.fits',
                 delete_flats=True,
                 multiprocessing=True):

    combined_flats = combined_flats.split('.fits')[0]

    #.skipping if master flats are found in folder
    with warnings.catch_warnings():
        warnings.simplefilter('error', AstropyUserWarning)
        try: 
            flat_ifc = ifc.filter(obstype='FLAT|SFLAT|DFLAT', regex_match=True)
        except AstropyUserWarning:
            print(f".ABORTING 'master flat' creation: no FLAT images found")
            return

    #.grouping FLAT-FIELD images in a list 
    flat_ifc = ifc.filter(obstype='FLAT|SFLAT|DFLAT', regex_match=True)
    file_list = flat_ifc.files

    #.applying OVERSCAN and BIAS corrections
    if multiprocessing: 
        #..loading 'master bias' in a memory buffer
        mbias_ccd = iaf.image_to_memory(master_bias)
        #..distributing images reduction to multiple processors
        with Pool() as pool:
            pool.map(partial(reduce_image, master_bias=mbias_ccd, 
                             master_flat=None, shared_memory=True,
                             merge_amplifiers=False), file_list)
        #..clearing memory buffers 
        iaf.unlink_memory(mbias_ccd)
    else: 
        #..loading 'master bias' in a CCDDATA object
        mbias_ccd = iaf.image_to_ccddata(master_bias)   
        #..reducing images one-by-one
        for file in file_list: 
            reduce_image(file, master_bias=mbias_ccd, 
                         master_flat=None, merge_amplifiers=False)

    #.combining FLAT-FIELDS by filter
    filters = iaf.ifc_filters(flat_ifc, filter_keywords=filter_keywords, 
                          obstype_selection='*')
    if isinstance(filter_keywords, str): keywds = filter_keywords.split(',')
    else: keywds = filter_keywords

    for fn,filt in enumerate(filters, start=1):

        #.grouping FLAT-FIELDS in this filter
        filt_list = list(flat_ifc.files_filtered(**{keywds[0]: filt}))
        for i in np.arange(1,len(keywds)): 
            filt_list.extend(list(flat_ifc.files_filtered(**{keywds[i]: filt})))
        filt_list = list(filter(None, filt_list))

        print(f"Creating '{os.path.splitext(os.path.basename(combined_flats))[0]}"+
            f"{fn}.fits': {len(filt_list)} images ({filt})")

        #.combining this filter FLAT-FIELDS into a 'master flat'
        combine_flat(filt_list, delete_images=delete_flats, 
                     output=f"{combined_flats}{fn}.fits", 
                     multiprocessing=multiprocessing)

    #.updating ImageFileCollection
    ifc.refresh()


def combine_flat(image_list, 
                 delete_images=True, 
                 output='master_flat.fits', 
                 multiprocessing=True):

    #.creating output master flat image
    mflat = fits.open(image_list[0])

    #.getting image extensions
    image_exts = iaf.image_extensions(mflat, is_hdu=True)

    #.calculating scaling factors for the flat-fields in the list
    fscales = iaf.flat_scale(image_list, normalize=True)

    #.combining flat-fields (per extension)
    if multiprocessing:
        with Pool() as pool:
            result = pool.map(partial(combine_flat_extension, flat_list=image_list, 
                                      scaling=fscales), image_exts)
    else:
        result=[]
        for ext in image_exts:
            result.append(combine_flat_extension(ext, flat_list=image_list, 
                                                 scaling=fscales))
    
    #.joining combined extensions into the output image 
    for res in result:
        ext = res[0]
        mflat[ext].data = res[1].data
        mflat[ext].header = res[1].header

    #.saving output image
    mflat[0].header['FILENAME']=output

    #..Goodman exception: creating a FLAT-FIELD mask
    if mflat[0].header['INSTRUME'].find("Goodman") >= 0:
        mask = np.ma.masked_less(mflat[0].data, 0.5)
        mask.fill_value = np.nan
        out = CCDData(mask.filled(), meta=mflat[0].header, mask=mask.mask, unit='adu')
        out.write(output, hdu_mask='MASK', 
                  hdu_uncertainty=None, hdu_flags=None, hdu_psf=None)
    else:
        mflat.writeto(output, overwrite=True)
        mflat.close()

    #.deleting individual flat images
    if delete_images: 
        for file in image_list: 
            try: os.remove(file)
            except OSError as e: print("Error: %s - %s." % (e.filename, e.strerror))


def combine_flat_extension(ext, flat_list, scaling=None):
    
    #.initializing scale factors
    if scaling is None: fscales = np.full(len(flat_list),1)
    else: fscales = np.array(scaling)
    
    #.grouping flat-field images from this amplifier
    ccd_list = [ CCDData.read(flat_list[j], hdu=ext, unit="adu")
                 for j in np.arange(len(flat_list)) ]
    #.(manually) scaling flats by the supplied factors
    # (comb.sigma_clipping does not work properly with comb.scaling)
    for k in np.arange(len(flat_list)): ccd_list[k].data /= fscales[k]
    #.preparing combiner object
    comb = Combiner(ccd_list, dtype=np.float32)
    comb.sigma_clipping(low_thresh=2., high_thresh=2., func='median', dev_func='mad_std')
    #.median combining the images
    comb_flat = comb.median_combine()

    #..creating and updating output header
    comb_flat.header = fits.getheader(flat_list[0], ext=ext)
    for n,imgn in enumerate(flat_list, start=1): 
        fstr = os.path.basename(imgn)
        comb_flat.header.append((f"IMCMB{n:03}", f"{fstr}[{ext}] (scale={fscales[n-1]:.1f})"))
    comb_flat.header.append(('NCOMBINE', len(ccd_list), '# images combined'))
    comb_flat.header.append(('FLATNORM', fscales[0], '# normalization scale'))

    #..returning the combined image for this extension
    return (ext, comb_flat)


def process_images(ifc, 
                  master_bias='master_bias.fits',
                  master_flat='master_flat.fits',
                  filter_keywords='filter1,filter2', 
                  multiprocessing=True):
    
    master_flat = master_flat.split('.fits')[0]

    #.identifing 'master flats' in the folder
    mflat_list = np.sort(glob.glob(master_flat+'*.fits'))
    flat_filters = iaf.image_filters(mflat_list, filter_keywords=filter_keywords)

    #.grouping SCIENCE images in a list 
    object_ifc = ifc.filter(obstype='OBJECT', regex_match=True)
    file_list = object_ifc.files

    #.selecting the proper 'master flat' for each image (FILTERS)
    obj_filters = iaf.find_filter(file_list, flat_filters, filter_keywords)

    #.reducing SCIENCE images
    if multiprocessing:
        #..loading 'master bias' and 'master flat' in a memory buffer
        mbias_ccd = iaf.image_to_memory(master_bias, name='mbias')
        mflat_ccd = [iaf.image_to_memory(flat, name='mflat'+flat.split('.fits')[0][-1]) 
                     for flat in mflat_list]
        #..distributing images reduction to multiple processors
        with Pool() as pool:
            pool.starmap(partial(reduce_image_mp, master_bias=mbias_ccd,
                         master_flats_list=mflat_ccd), 
                         zip(file_list,obj_filters))
        #..clearing memory buffers 
        iaf.unlink_memory(mbias_ccd)
        for flat in mflat_ccd: iaf.unlink_memory(flat)
    else:
        #.loading 'master bias' and 'master flat' in CCDDATA objects
        mbias_ccd = iaf.image_to_ccddata(master_bias)
        mflat_ccd = [iaf.image_to_ccddata(image) for image in mflat_list]
        #.reducing images one-by-one
        for i,file in enumerate(file_list):
            reduce_image(file, 
                         master_bias=mbias_ccd, 
                         master_flat=mflat_ccd[obj_filters[i]],
                         merge_amplifiers=True)

    #.updating ImageFileCollection
    ifc.refresh()


def reject_cosmicrays(dataset, **kwargs):
    
    #.If input dataset is a folder, expand it into a ImageFileCollection
    if isinstance(dataset,str): 
        ifc = ImageFileCollection(dataset, ext=0,
         keywords=['obstype','ut','ccdsum','airmass','exptime','object'],
         glob_exclude="*master*.fits, bpm*.fits", glob_include="*.fits")
    
    #.Otherwise the input dataset is already a ImageFileCollection
    else: ifc = dataset

    #.grouping SCIENCE images in a list 
    obj_ifc = ifc.filter(obstype='OBJECT', regex_match=True)
    obj_list = obj_ifc.files

    #.Looping over image list to clean cosmic rays 
    for file in obj_list: 
        print(f".Cleaning cosmic rays: {file}", end=" - ")
        lacosmic_image(file)
        

def lacosmic_image(image, **kwargs):

    #.If input image is a string, treat it as an image file path
    if isinstance(image, str): 
        hdul = fits.open(image, mode='update')
    #.else treat it as an HDU
    else: 
        hdul = image

    img = CCDData(hdul[0].data, meta=hdul[0].header, unit="adu")

    #.Abort if the image has already been cleaned from cosmic rays
    if ('LACOSMIC' in img.header): 
        print('Already cleaned')
        return

    #.Expanding kwargs
    if 'sigclip' not in kwargs: kwargs['sigclip'] = 6
    if 'sigfrac' not in kwargs: kwargs['sigfrac'] = 0.25
    if 'objlim' not in kwargs:  kwargs['objlim'] = 5
    if 'fsmode' not in kwargs:  kwargs['fsmode'] = 'median'
    if 'niter' not in kwargs:
        if ('EXPTIME' in img.header): 
            expt = img.header['EXPTIME']
            kwargs['niter'] = 1+(expt>=100)+(expt>=300)
        else: kwargs['niter'] = 2

    #.Getting essential info from header
    get_date = datetime.now().strftime("%b %d %Y %H:%M")
    if ('FWHM' in img.header): fwhm = img.header['FWHM']
    else: fwhm = 10.
    
    #.Cleaning cosmic rays with LACOSMIC
    img_cln = cosmicray_lacosmic( img, gain_apply=False, 
                                    gain=img.header['GAIN'],
                                    readnoise=img.header['RDNOISE'],
                                    satlevel=img.header['SATURATE'],
                                    verbose=False, **kwargs )

    #.Updating header and saving processed image (overwrite)
    print(f"{kwargs['niter']} passes")
    img_cln.header.set('LACOSMIC', 
        f"{get_date} {kwargs['fsmode']} Nit={kwargs['niter']} sigclip={kwargs['sigclip']} "+
        f"sigfrac={kwargs['sigfrac']} objlim={kwargs['objlim']}")

    hdul[0].data = img_cln.data
    hdul[0].header = img_cln.header
    hdul.close()


def fwhm_estimate(dataset, multiprocessing=False, **kwargs):

    #.If input dataset is a folder, expand it into a ImageFileCollection
    if isinstance(dataset,str): 
        ifc = ImageFileCollection(dataset, ext=0,
         keywords=['obstype','ut','ccdsum','airmass','exptime','object'],
         glob_exclude="*master*.fits, bpm*.fits", glob_include="*.fits")
    
    #.Otherwise the input dataset is already a ImageFileCollection
    else: ifc = dataset

    #.grouping SCIENCE images in a list 
    obj_ifc = ifc.filter(obstype='OBJECT', regex_match=True)
    obj_list = obj_ifc.files

    #.estimating FHWM for each image
    if multiprocessing: 
        with Pool() as pool:
            pool.map(partial(fwhm_image, **kwargs), obj_list)
    else: 
        for file in obj_list: 
            fwhm_image(file, **kwargs)


def fwhm_image(file, image_area=0.33, is_hdu=False, min_fwhm=1.5, **kwargs):

    if is_hdu: hdul=file
    else: hdul = fits.open(file, mode='update')

    print(f".Moffat fitting {file}:", end=" ")

    #.aborting if FWHM is found in header
    ifwhm = hdul[0].header.get('FWHM')
    if ifwhm:
        print(f".FWHM found in header = {ifwhm:.2f}")
        return

    #.looping over the image extensions
    image_indices = iaf.image_extensions(file, is_hdu=is_hdu)
    n_ext = len(image_indices)
    table = None
    for ext in image_indices:
    
    #.initializing arguments
        img = hdul[ext].data
        hdr = hdul[ext].header

        if n_ext == 1:
            imsz = img.shape
            border = (1-image_area)/2
            data = img[round(border*imsz[0]):round((1-border)*imsz[0]),
                       round(border*imsz[1]):round((1-border)*imsz[1])]
        else: data = img
    
    #.getting header keywords
        if 'saturation' not in kwargs:
            kwargs['saturation'] = hdr.get('SATURATE', default=52430)
        if 'gain' not in kwargs:
            kwargs['gain'] = hdr.get('GAIN', default=1)

    #.calculating FWHM for this extension stars
        tab = fwhm_fit(data, **kwargs)
    #..stacking tables from multiple extensions
        if tab is None: continue
        elif table is None: table = tab
        else: table = vstack([table,tab], metadata_conflicts='silent')
        
    #.if there are enough stars, skip to the end 
        good_data = np.isfinite(table['fwhm']) & (table['fwhm'] > min_fwhm)
        n_good = np.sum(good_data)
        if n_good >= 10: break

    #.getting best value for FWHM and storing in header
    if table is None: n_good = 0
    
    if n_good > 0:
        fwhm = np.median(table['fwhm'][good_data])
        beta = np.median(table['beta'][good_data])
    else: fwhm, beta = 0, 0
    print(f"median value FWHM = {fwhm} ({n_good} stars)")

    for ext in np.append(image_indices, 0):
        hdul[ext].header.set("FWHM",fwhm,f"Moffat FWHM (median of {n_good} values)")
        hdul[ext].header.set("BETA",beta,f"Moffat Beta (median of {n_good} values)")
        hdul[ext].header.set("BACK",table.meta['back'],f"Median background value")
        hdul[ext].header.set("BACK_RMS",table.meta['back_rms'],f"Background std-dev (from MAD)")

    #.returning
    if is_hdu: return hdul
    else: 
        hdul.close()


def fwhm_fit(image, n_max=50, saturation=52430., gain=1):

    initial_fwhm = 10
    half_box = initial_fwhm

    #.Using sigma-clipping to model the background
    # (actually, replaced by simple median and MAD for speed)
    back_median = np.nanmedian(image)
    back_std = 1.4826*np.nanmedian(np.abs(image-back_median))

    #.Detecting sources using IRAFStarfinder method
    # (actually, DAOFinder is faster than IRAFStarFinder)
    # (find_peaks is even faster, but unreliable)
    daofinder = DAOStarFinder(3*back_std, 1.5*initial_fwhm,
                            roundlo=-2.0, roundhi=2.0, sharplo=0.01, sharphi=10.0,
                            brightest=n_max, exclude_border=False, peakmax=saturation)
    
    #..catching zero objects warning and aborting
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=NoDetectionsWarning)
        try:
            tab = daofinder.find_stars(image[half_box:-half_box,half_box:-half_box]-back_median)
        except NoDetectionsWarning:
            return None
    
    tab['xcentroid'] += half_box
    tab['ycentroid'] += half_box
    tab.sort('mag')

    #.Building a global pixel grid with 1-FWHM size
    pos_xy = np.arange(-half_box, half_box, 1)
    global_x, global_y = np.meshgrid(pos_xy, pos_xy)
    params, parerr = np.full((n_max,5), np.nan), np.full((n_max,5), np.nan)

    for i,row in enumerate(tab):

        #.Adjusting grid for this star
        xcen, ycen = round(row['xcentroid']), round(row['ycentroid'])
        grid_x = global_x + xcen
        grid_y = global_y + ycen

        #.Fitting 2D model with curve_fit
        datax = np.vstack((grid_x.ravel(),grid_y.ravel())).astype(float)
        datay = image[ycen-half_box:ycen+half_box, xcen-half_box:xcen+half_box].ravel()-back_median
        datay = datay.clip(min=1)
        erroy = np.sqrt(datay/gain)
        p0 = [row['peak'],xcen,ycen,10,3.5]
        #..error cactching the fitting
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=RuntimeWarning)
            warnings.filterwarnings("error", category=OptimizeWarning)
            try:
                popt, pcov = curve_fit(iaf.moffat2d, datax[:,:], datay[:],p0=p0,
                                    sigma=erroy[:], absolute_sigma=True)
            except (RuntimeError, RuntimeWarning, OptimizeWarning):
                continue

        #..gathering resulting coefficients
        perr = np.sqrt(np.diag(pcov))
        params[i,:] = popt
        parerr[i,:] = perr/popt
    
    #.composing output table
    mask = (parerr[:,3] >= 1) | (parerr[:,4] >= 1) | (params[:,4] <= 1)
    params[mask,:] = np.nan
    parerr[mask,:] = np.nan

    tab['fwhm'] = 2*abs(params[:,3])*np.sqrt(2**(1/params[:,4])-1)
    tab['beta'] = params[:,4]
    tab.meta = {'back': back_median, 'back_rms': back_std}
    tab.remove_column('id')

    return tab


