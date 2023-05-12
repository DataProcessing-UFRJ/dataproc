from argparse import ArgumentParser
from ccdproc import ImageFileCollection
from dataprocessing_functions import process_bias, process_flat, process_images

def parse_arguments():

    #.list flattening generator
    flatten = lambda *n: (e for a in n
    for e in (flatten(*a) if isinstance(a, (tuple, list)) else (a,)))

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
    
    args=parser.parse_args()
    
    #..checking that folder argument ends with '/'
    args.folder = args.folder.strip()
    if args.folder[-1] != '/': args.folder += '/'

    #..fixing nested listing inside 'keywords' argument
    keylist = args.summary_keywords
    keylist += args.filter_keywords
    args.summary_keywords = list(flatten(keylist))

    return args


def data_reduce(folder,
                filter_keywords=['filter'],
                summary_keywords=['obstype','exptime','object'],
                summary_file='observations.log',
                logfile='data_reduce.log',
                multiprocessing=False):
    
    #.initiating ImageFileCollection
    ifc = ImageFileCollection(
        folder, keywords=summary_keywords, ext=0, 
        glob_exclude='*master*.fits', glob_include='*.fits')
    
    #.writing data summary table
    if summary_file:
        tab=ifc.summary
        tab.write(folder+summary_file,
                  format='ascii.fixed_width_two_line',overwrite=True)

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


#.Initializing main function from the command line
if __name__ == '__main__':
    
    args = parse_arguments()
    data_reduce(args.folder,
                filter_keywords=args.filter_keywords,
                summary_keywords=args.summary_keywords,
                summary_file=args.summary_file,
                logfile=args.logfile,
                multiprocessing=args.multiprocessing)
    