import os, sys, glob
from datetime import datetime
import numpy as np
from astropy.io import fits
from astropy.nddata import CCDData
from functools import partial
from multiprocessing import Pool
from multiprocessing.shared_memory import SharedMemory
from ccdproc import subtract_overscan, trim_image, subtract_bias, flat_correct, Combiner
import imageauxiliary_functions as iaf

import warnings
from astropy.wcs.wcs import FITSFixedWarning
warnings.simplefilter('ignore', category=FITSFixedWarning)
warnings.simplefilter('ignore', category=UserWarning)


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
    
    #.grouping BIAS images into a list 
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

