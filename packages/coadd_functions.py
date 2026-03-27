import sys, os, re
import numpy as np
import json
from ccdproc import ImageFileCollection
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import NDData,Cutout2D
from astropy.wcs import WCS
import astropy.units as u
from functools import partial
from multiprocessing import Pool
from parallelbar import progress_starmap, progress_map
from reproject.mosaicking import find_optimal_celestial_wcs, reproject_and_coadd
from reproject import reproject_interp, reproject_exact, reproject_adaptive

from packages.imageauxiliary_functions import get_filter_label


def table_expr_parser(column_expression, column_names, table_name='table'):

    #.Splitting expression using common operators
    split_expr = re.split(r'\s+[+\-<>=!&|]+\s+', column_expression)
    for side in split_expr:
        #..identifying variables in each expression side
        variables = set(re.findall(r'\d+\.\d+|\w+', side))
        #..replacing variables by column names
        for var in variables: 
            if var in column_names: 
                column_expression = column_expression.replace(var,table_name+"['"+var+"']")

    #.Enclosing both sides of boolean operators in parenthesis ()
    for sep in [' & ',' \| ']:
        side = re.split(sep, column_expression)
        if len(side) > 1:
            for j in range(len(side)): side[j] = '('+side[j]+')'
            sep = sep.replace('\\','')
            column_expression = sep.join(side)

    #.Return expression ready for evaluation
    return column_expression


def table_expr_keywords(column_expression):

    expression_keywords, possible_keywords = [], []
    
    for string_expr in column_expression:
        #.splitting conditions in expression by boolean operators
        joint_condition = re.split(r'[\s]*[&|][\s]*', string_expr)
        for expression in joint_condition:
            #..removing parenthesis and braces
            expression = re.sub(r'[()\[\]]','',expression)
            #..splitting left-hand-side (LHS) and right-hand-side (RHS) using common operators
            expression_side = re.split(r'\s+[<>=!?]+\s+', expression)
            for i,side in enumerate(expression_side):
                #...getting operands at the LHS
                if i == 0:
                    operands = re.split(r'\s+[\+\*-/]+\s+', side)     
                    expression_keywords.extend(operands)
                #...getting operands at the RHS
                else:
                    operands = re.findall(r'[a-zA-Z_][a-zA-Z0-9_\-\.]*',side)
                    if operands: possible_keywords.extend(operands)

    return list(set(expression_keywords)), list(set(possible_keywords))


def validate_header_keywords(header,keyword_list):

    valid = [keyw in header for keyw in keyword_list]

    for exception in ['offset','sequence']:
        if exception in keyword_list: 
            valid[keyword_list.index(exception)] = True

    return valid


class NpEncoder(json.JSONEncoder):
# Class to extend the JSONEncoder class

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    

def create_image_sets(dataset, 
                      group_images_by=['object','filter1','filter2','sequence'],
                      separate_group_by = ['exptime <= 30 & offset < 0.02','exptime >= 100 & offset < 0.02'],
                      coordinate_keywords = ['ra','dec'],
                      imsets_file='imsets.json'):
    
    #.Checking if the Image sets file (the output) already exists
    if isinstance(dataset,str):
        imsets_path = os.path.join(dataset,imsets_file)
    else: 
        imsets_path = os.path.join(dataset.location,imsets_file)

    if os.path.isfile(imsets_path): 
        print(f".Image sets file found: {imsets_path}")
        with open(imsets_path,'r') as input_file:
            return json.load(input_file)
    else:         
        print(f".Creating Image sets file: {imsets_path}")


    #.Processing input for demanded keywords
    ifc_keywords = ['obstype','object','airmass','exptime']
    group_keywords = group_images_by.copy()

    separate_keywords, possible_keywords = table_expr_keywords(separate_group_by)

    demanded_keywords = list(set(group_keywords+separate_keywords))
    if 'offset' in separate_keywords: demanded_keywords += coordinate_keywords

    mask = [re.match('filter.*',keyw) != None for keyw in demanded_keywords]
    filter_keywords = list(np.asarray(demanded_keywords)[mask])

    #----------------------------------------------------------------------------------
    #.If input dataset is a folder, expand it into a ImageFileCollection
    ifc_keywords = list(set(ifc_keywords))
    if isinstance(dataset,str): 
        ifc = ImageFileCollection(dataset, ext=0, keywords=ifc_keywords,
            glob_exclude="*master*.fits", glob_include="*.fits")
        
    else: ifc = dataset    

    #.Validating demanded keywords in header
    header = next(ifc.headers())
    valid = validate_header_keywords(next(ifc.headers()),possible_keywords)
    for val,key in zip(valid,possible_keywords):
        if val and key not in demanded_keywords: demanded_keywords.append(key)

    valid = validate_header_keywords(next(ifc.headers()),demanded_keywords)
    if ~np.all(valid): 
        sys.exit(f'ABORTING: keywords not found: {list(np.array(demanded_keywords)[~np.array(valid)])}')
        
    ifc.keywords += demanded_keywords
    
    #.Generating main images table
    table = ifc.summary
    table['file'] = [os.path.join(ifc.location,file) for file in table['file']]

    #..adding sequence counter
    obj_list = np.array(table['object'])
    seq_number = 0
    seq_list = [seq_number]
    for i in np.arange(1,len(obj_list)):
        if obj_list[i] != obj_list[i-1]: seq_number += 1
        seq_list.append(seq_number)
        
    table['sequence'] = seq_list

    #----------------------------------------------------------------------------------
    #.Grouping exposures by keywords
    grouped = table.group_by(group_images_by)
    grouped_sizes = np.diff(grouped.groups.indices)

    #.Initiating image sets
    imsets = {}
    for i,group in enumerate(grouped.groups):

        #..calculating offsets for images in each group
        if grouped_sizes[i] > 1:
            coords = SkyCoord(ra= group[coordinate_keywords[0]], 
                              dec=group[coordinate_keywords[1]], 
                              unit=(u.hour,u.degree), frame='fk5')
            group['offset'] = np.round(coords[0].separation(coords).value, 7)

        #..separating groups by keyword expressions
        for separate in separate_group_by:

            #...evaluating expression
            eval_str = table_expr_parser(separate, demanded_keywords, table_name='group')
            mask = eval(eval_str)
            #...generating image set separated by this keyword expression
            if np.any(mask):
                imset_dict = {}
                for kwd in group_keywords:
                    imset_dict[kwd] = group[0][kwd]
                imset_dict['images'] = list(group['file'][mask])
                for kwd in separate_keywords:
                    imset_dict[kwd] = list(group[kwd][mask])

                imset_id = ''.join(group[0]['object'].split())
                imset_id+= '_'+get_filter_label(group[0][filter_keywords])
                seq = 1
                while imset_id+str(seq) in imsets: seq+=1
                imset_dict['output'] = str(os.path.join(ifc.location,imset_id+str(seq)+'.fits'))

                #....adding this imset to output dictionary
                imsets[imset_id+str(seq)] = imset_dict

    #----------------------------------------------------------------------------------
    #.Saving resulting image sets to output file (.json)
    with open(imsets_path,'w') as output_file:
        json.dump(imsets, output_file, indent=4, cls=NpEncoder)
        
    return imsets


def trim_image(image, footprint=None, header=None, image_ext=0, footprint_ext=1):

    #.Opening image
    if isinstance(image, np.ndarray):
        data = image.copy()
    elif isinstance(image, fits.HDUList):
        header = image[image_ext].header.copy()
        data = image[image_ext].data.copy()
    elif isinstance(image,str):
        hdul = fits.open(image, mode='update')
        header = hdul[image_ext].header.copy()
        data = hdul[image_ext].data.copy()

    #.Initializing footprint
    if footprint is None: mask=data.copy()
    elif isinstance(footprint, np.ndarray):
        mask = footprint
    elif isinstance(footprint, fits.HDUList):
        mask = footprint[footprint_ext].data
    elif isinstance(footprint,str):
        with fits.open(footprint, mode='read') as hdu2:
            mask = hdu2[footprint_ext].data

    #.Setting up WCS
    wcs = WCS(header, fix=False)        

    #.Trimming image
    y,x = np.where(mask > 0)
    min_x, max_x = min(x),max(x)
    min_y, max_y = min(y),max(y)
    trim_image = Cutout2D(data, ((min_x+max_x)/2, (min_y+max_y)/2), 
                        (max_y-min_y+1, max_x-min_x+1), wcs=wcs)
    header.update(trim_image.wcs.to_header())

    #.Returning trimmed image
    if isinstance(image, np.ndarray):
        return trim_image.data, header
    elif isinstance(image, fits.HDUList):
        image[image_ext].data = trim_image.data
        image[image_ext].header = header
        return image
    elif isinstance(image,str):
        hdul[image_ext].header = header
        hdul[image_ext].data = trim_image.data
        hdul.writeto(image, overwrite=True)
        hdul.close()


def coadd_images(images_list, output_file,
                 instrument='SAMI',
                 coverage_mask=None,
                 weight_map=None,
                 scale_keyword=None, 
                 weight_keyword='1/FWHM**2',
                 output_maps=False,
                 auto_rotate=True):

    #.Checking if output image already exists
    if os.path.isfile(output_file): 
        print(f"  {output_file} already exists")
        return
    else:
        print(f".coadding {[os.path.basename(img) for img in images_list]} to {output_file}")

    #.Checking scale and weight keyword expressions
    n_images = len(images_list)
    with fits.open(images_list[0]) as hdul:
        header = hdul[0].header
        imsize = hdul[0].data.shape
        ccdsum = np.array(header['CCDSUM'].split(), dtype=np.single)
        flat_image = re.findall('master_flat\d\.fits', header['FLATCOR'])[0]
        
        if scale_keyword is None: 
            base_scale = None
        else:
            scale_expr = table_expr_parser(scale_keyword, list(header.keys()), table_name='header')
            try: 
                base_scale = eval(scale_expr)
            except NameError:
                base_scale = None 
                print('.KEYWORD NOT FOUND IN SCALE EXPRESSION:', scale_expr)

        if weight_keyword is None:
            base_weight = None
        else:
            weight_expr = table_expr_parser(weight_keyword, list(header.keys()), table_name='header')
            try: 
                base_weight = eval(weight_expr)
            except NameError:
                base_weight = None
                print('.KEYWORD NOT FOUND IN WEIGHT EXPRESSION:', weight_expr)

    #.Setting default weight map and coverage mask based on instrument
    flat_path = os.path.join(os.path.dirname(images_list[0]),flat_image)
    if instrument.lower().find('goodman') >= 0:
        with fits.open(flat_path) as hdul:
            weight_map = np.clip(hdul[0].data**60,0.,1.).astype(np.single)
            coverage_mask = (1-hdul[1].data).astype(bool)
            weight_map[:,int(2880/ccdsum[0]):] *= 0.
            coverage_mask[:,int(2880/ccdsum[0]):] = False

    elif instrument.lower().find('sam') >= 0:
        with fits.open(flat_path.replace('_flat','_flat_merged')) as hdul:
            weight_map = np.clip(hdul[0].data,0.,1.).astype(np.single)
            coverage_mask = None
            auto_rotate = True

    #.Initializing null coverage mask and weight map if none are provided
    if coverage_mask is None: 
        coverage_mask = np.ones(imsize, dtype=bool)
    wgt_hdus=[]
    if weight_map is None:
        if base_weight is None:  wgt_hdus = None
        else:  weight_map = np.ones(imsize, dtype=np.single) 

    #.Setting up coverage masks list
    if isinstance(coverage_mask, list):
        coverage_list = coverage_mask
        mask_index = list(range(len(coverage_mask)))
    else:
        coverage_list = [coverage_mask]
        mask_index = [0]*n_images

    #.Setting up weight maps list
    if isinstance(weight_map, list):
        weight_list = weight_map
        weight_index = list(range(len(weight_map)))
    else:
        weight_list = [weight_map]
        weight_index = [0]*n_images

    #.Generating saturation, weight and gain maps 
    mosaic_hdus, sat_hdus, gain_hdus = [], [], []
    for i,file in enumerate(images_list):
        with fits.open(file) as hdul:
            header = hdul[0].header
            wcs = WCS(header)

            if base_scale is None: scale_factor = 1
            else: scale_factor = eval(scale_expr)/base_scale
            if base_weight is None: weight_factor = 1
            else: weight_factor = eval(weight_expr)/base_weight

            footprint = coverage_list[mask_index[i]]

            data = hdul[0].data/header['exptime']/scale_factor
            data[~footprint] = np.nan
            mosaic_hdus.append(NDData(data, wcs=wcs, meta=header))

            if output_maps:
                sat_data = footprint*header['saturate']/header['exptime']
                sat_data[~footprint] = np.nan
                sat_hdus.append(NDData(sat_data, wcs=wcs))       

                gain_data = footprint*header['gain']
                gain_data[~footprint] = np.nan
                gain_hdus.append(NDData(gain_data, wcs=wcs))

            if wgt_hdus is not None:
                weight = weight_list[weight_index[i]]
                wgt_hdus.append(weight*weight_factor)

    #.Calculating output mosaic shape and WCS
    wcs_mosaic, mosaic_shape = find_optimal_celestial_wcs(mosaic_hdus, projection='TAN', auto_rotate=auto_rotate)

    #.Composing output header with the updated mosaic WCS
    out_header = mosaic_hdus[0].meta.copy()
    del out_header['CD?_?']
    del out_header['A_?_?']
    del out_header['B_?_?']
    out_header.remove('A_ORDER', ignore_missing=True, remove_all=True)
    out_header.remove('B_ORDER', ignore_missing=True, remove_all=True)

    out_header.set('ori_gain', out_header['gain'], after='gain', 
                comment='gain of an individual exposure [e-/adu]')
    out_header.set('ori_expt', out_header['exptime'], after='exptime',
                comment='exposure time of the original image')
    out_header.set('ori_satu', out_header['saturate'], after='saturate',
                comment='saturation level of the original image')
    out_header.set('exptime', 1.)
    out_header.set('extname', 'MOSAIC_DATA')
    out_header.set('ncombine', n_images)
    for i,file in enumerate(images_list): out_header.set(f'imcmb{i:0>3.0f}', file)
    out_header.update(wcs_mosaic.to_header(relax=True))

    #.Combining images into output mosaics
    output_hdus = []
    if output_maps:
        gain_mosaic, _ = reproject_and_coadd(gain_hdus, wcs_mosaic,
                                        shape_out=mosaic_shape,
                                        input_weights=None,
                                        combine_function='sum',
                                        match_background=False,
                                        background_reference=None,
                                        reproject_function=reproject_interp,
                                        blank_pixel_value=np.nan)

        trimmed_img, _ = trim_image(gain_mosaic, header=out_header.copy())
        # trimmed_img, trim_header = gain_mosaic, out_header
        output_hdus.append(fits.ImageHDU(trimmed_img.astype(np.single), name='effective_gain', 
                                        do_not_scale_image_data=True, uint=False))

        sat_mosaic, _  = reproject_and_coadd(sat_hdus, wcs_mosaic,
                                        shape_out=mosaic_shape,
                                        input_weights=None,
                                        combine_function='min',
                                        match_background=False,
                                        background_reference=None,
                                        reproject_function=reproject_interp,
                                        blank_pixel_value=np.nan)   

        trimmed_img, _ = trim_image(sat_mosaic, header=out_header.copy())
        # trimmed_img, trim_header = sat_mosaic, out_header
        output_hdus.append(fits.ImageHDU(trimmed_img.astype(np.single), name='saturation', 
                                        do_not_scale_image_data=True, uint=False))

        out_header.set('gain', 1.)
        out_header.set('saturate', np.nanmedian(sat_mosaic).astype(np.single))
        out_footprint = sat_mosaic

    else: 
        out_header['gain'] = np.float32(header['gain']*n_images)
        out_header['saturate'] = np.float32(out_header['saturate']/out_header['ori_expt'])
        out_footprint = None

    img_mosaic, _ = reproject_and_coadd(mosaic_hdus, wcs_mosaic,
                                        shape_out=mosaic_shape,
                                        input_weights=wgt_hdus,
                                        combine_function='mean',
                                        match_background=True,
                                        background_reference=1,
                                        reproject_function=reproject_adaptive,
                                        blank_pixel_value=np.nan,
                                        conserve_flux=True, kernel='hann',
                                        parallel=False)

    trimmed_img, trim_header = trim_image(img_mosaic, footprint=out_footprint, header=out_header)
    # trimmed_img, trim_header = img_mosaic, out_header
    output_hdus.insert(0,fits.PrimaryHDU(data=trimmed_img.astype(np.single), header=trim_header,
                                         do_not_scale_image_data=True, uint=False))

    #.Saving output mosaic (and maps) into a HDUList object
    hdul = fits.HDUList(output_hdus)

    #.Returning the mosaic
    if output_file is None: return hdul
    else: hdul.writeto(output_file, overwrite=True)


def coadd_imageset(imageset, multiprocessing=True, **kwargs):

    setimages = [setdict['images'] for _,setdict in imageset.items()]
    outimages = [setdict['output'] for _,setdict in imageset.items()]

    if multiprocessing:
        # result = progress_starmap(partial(coadd_images, **kwargs), list(zip(setimages,outimages)), 
        #                           chunk_size=1, n_cpu=16)
        with Pool(16) as pool:
            pool.starmap(partial(coadd_images, **kwargs), zip(setimages,outimages),
                         chunksize=1)

    else:
        for imglist,outimage in zip(setimages,outimages):
            coadd_images(imglist, outimage, **kwargs)


