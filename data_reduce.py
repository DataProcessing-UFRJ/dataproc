#!/usr/bin/env python3
from argparse import ArgumentParser
from ccdproc import ImageFileCollection
from dataprocessing_functions import header_setup, process_bias, process_flat, process_images, reject_cosmicrays, fwhm_estimate
from astrometry_solve import astrometry_solve

def parse_arguments():

    #.parsing command line elements to function arguments.
    parser = ArgumentParser(
        prog="data_reduce", 
        description="Main script to perform data reduction of imaging exposures"
    )
    parser.add_argument(
        "folder", type=str,
        help="folder containing the science and calibration images to reduce"
    )
    parser.add_argument(
        "--filter_keywords", nargs="+", default=['filter'],
        metavar="keyword1 keyword2",
        help="header keywords containg the photometric filters used (default: %(default)s)"
    )
    parser.add_argument(
        "--summary_file", type=str, default="observations.log",
        help="output table: summary of main header keywords of each image (default: %(default)s)"
    )
    parser.add_argument(
        "--summary_keywords", nargs="+", action="append",
        default=['obstype','time-obs','ccdsum','airmass','exptime','object'],
        metavar="keyword1 keyword2",
        help="keywords used to build the 'summary_file' output table (default: %(default)s)"
    )
    parser.add_argument(
        "--logfile", type=str, default="data_reduce.log",
        help="file to log the opperations performed over the images (default: %(default)s)"
    )
    parser.add_argument(
        "--multiprocessing", action="store_true", default=False,
        help="enable splitting image processing over all CPU cores (default: %(default)s)"
    )
    parser.add_argument(
        "--instrument", type=str, default="SAMI",
        help="instrument name designation or configuration file (default: %(default)s)"
    )

    
    args=parser.parse_args()
    
    #..checking that folder argument ends with '/'
    args.folder = args.folder.strip()
    if args.folder[-1] != '/': args.folder += '/'

    return args


def data_reduce(folder,
                filter_keywords=['filter'],
                summary_keywords=['obstype','object','airmass','exptime'],
                summary_file='observations.log',
                logfile='data_reduce.log',
                instrument='SAMI',
                multiprocessing=False):
    
    #.processing filter keywords
    if isinstance(filter_keywords,str): 
        filter_keywords=filter_keywords.split(',')
    for keyw in filter_keywords: 
        if keyw not in summary_keywords: summary_keywords.append(keyw)

    #.initiating ImageFileCollection
    ifc = ImageFileCollection(
        folder, keywords=summary_keywords, ext=0, 
        glob_exclude='*master*.fits', glob_include='*.fits')
    
    #.writing data summary table
    if summary_file:
        tab=ifc.summary
        tab.write(folder+summary_file,
                  format='ascii.fixed_width_two_line',overwrite=True)

    #.configuring HEADERS
    header_setup(ifc, instrument=instrument,
                 multiprocessing=multiprocessing)

    #.processing BIAS
    process_bias(ifc, 
                 combined_bias=folder+'master_bias.fits', 
                 multiprocessing=multiprocessing)
    
    #.processing FLATS
    process_flat(ifc, filter_keywords=filter_keywords, 
                 master_bias=folder+'master_bias.fits', 
                 combined_flats=folder+'master_flat.fits', 
                 multiprocessing=multiprocessing)
    
    #.processing science images
    process_images(ifc, filter_keywords=filter_keywords, 
                   master_bias=folder+'master_bias.fits', 
                   master_flat=folder+'master_flat.fits', 
                   multiprocessing=multiprocessing)
    
    #.solving astrometry
    astrometry_solve(ifc, catalog='I/350', cat_magnitude='Gmag',
                     cat_constraints={'Gmag': '< 20.5'},
                     multiprocessing=multiprocessing)
    
    #.removing cosmic rays 
    reject_cosmicrays(ifc)

    #.estimating FWHM 
    fwhm_estimate(ifc, multiprocessing=multiprocessing)


#.Initializing main function from the command line
if __name__ == '__main__':
    
    args = parse_arguments()
    data_reduce(args.folder,
                filter_keywords=args.filter_keywords,
                summary_keywords=args.summary_keywords,
                summary_file=args.summary_file,
                logfile=args.logfile,
                multiprocessing=args.multiprocessing,
                instrument=args.instrument)