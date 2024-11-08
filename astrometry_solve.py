from ccdproc import ImageFileCollection
from functools import partial
from multiprocessing import Pool
from wcs_functions import wcs_solve

def astrometry_solve(dataset, multiprocessing=False, **kwargs):

    #.If input dataset is a folder, expand it into a ImageFileCollection
    if isinstance(dataset,str): 
        ifc = ImageFileCollection(dataset, ext=0,
         keywords=['obstype','ut','ccdsum','airmass','exptime','object'],
         glob_exclude="*master*.fits, bpm*.fits", glob_include="*.fits")
    
    #.Otherwise the input dataset is already a ImageFileCollection
    else: ifc = dataset

    #.grouping SCIENCE images in a list 
    object_ifc = ifc.filter(obstype='OBJECT', regex_match=True)
    file_list = object_ifc.files

    #.Calculating astrometric solution for each image
    if multiprocessing:
        with Pool() as pool:
            pool.map(partial(wcs_solve,**kwargs), file_list)
    else:
        for file in file_list:
            wcs_solve(file, **kwargs)
