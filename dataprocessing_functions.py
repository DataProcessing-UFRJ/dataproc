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


def reduce_image(image_file, 
                 master_bias=None, 
                 master_flat=None,
                 shared_memory=False,
                 merge_amplifiers=True):

    #.Abrindo o arquivo fits da imagem
    hdul = fits.open(image_file)
    image_exts = iaf.image_extensions(hdul, is_hdu=True)

    get_date = datetime.now().strftime("%b %d %Y %H:%M")
    fstr = os.path.basename(image_file)
    
    #..preparando dados do master bias
    if master_bias is not None:
        if shared_memory:
            mbias_ccd = iaf.memory_to_ccddata(master_bias)
            mbias_buf = [SharedMemory(name=master_bias[i]['buffer']) for i in range(len(master_bias))]
        else: mbias_ccd = master_bias
        bstr = os.path.basename(mbias_ccd[0].header['FILENAME'])

    #.preparando dados dos master flats
    if master_flat is not None:
        if shared_memory: 
            mflat_ccd = iaf.memory_to_ccddata(master_flat)
            mflat_buf = [SharedMemory(name=master_flat[i]['buffer']) for i in range(len(master_flat))]
        else: mflat_ccd = master_flat
        flstr = os.path.basename(mflat_ccd[0].header['FILENAME'])

    #.preparando para juntar os amplificadores
    can_merge = (len(image_exts) > 1) and merge_amplifiers and (hdul[0].header['OBSTYPE']=='OBJECT')
    if can_merge:
         #..coletando binagem e dimensoes da imagem
        ccdsum = hdul[image_exts[0]].header['CCDSUM']
        bin_x, bin_y = int(ccdsum[0]), int(ccdsum[2])
        imsz_x, imsz_y = iaf.iraf2python(hdul[image_exts[0]].header['DETSIZE'])
        imsz_x, imsz_y = int(imsz_x[1]/bin_x), int(imsz_y[1]/bin_y)
        #..SOI gap
        inst = hdul[0].header['INSTRUME']
        if inst.find('SOI') >= 0: 
            xgap = np.array([0,0,102/bin_x,102/bin_x],dtype=int)
            ygap = np.array([0,0,0,0], dtype=int)
        else: xgap, ygap = np.full(4,0), np.full(4,0)

        #..inicializando imagem final
        img_merge = np.full((imsz_y+np.amax(ygap), imsz_x+np.amax(xgap)), np.nan, dtype=np.float32)

    #.Loop ao longo das extensoes (amplificadores) da imagem
    proc_string = f".Processing {fstr:1s}"
    skip = np.full(len(image_exts),True)
    for ne,ext in enumerate(image_exts):

        #.Lendo dados desta extensao
        proc_string+=f" [{ext:1.0f}]"
        img = CCDData(hdul[ext].data, meta=hdul[ext].header, unit="adu")
        
        #.Realizando correcao de OVERSCAN
        if ('BIASSEC' in img.header):
            proc_string+="o"
            biassec = img.header['BIASSEC']        
            img = subtract_overscan(img, 
                        fits_section=biassec, 
                        model=None, median=True,
                        add_keyword={'overscan': f"{get_date} Overscan is {biassec}; model=median"})

            #.SOI exception
            #.(keyword TRIMSEC errada no header)
            trimsec = img.header['TRIMSEC']
            if (hdul[0].header['INSTRUME'].find('SOI') >= 0):
                if (ext in [1,3]): trimsec = '[29:540,1:2048]'
                else: trimsec = '[28:539,1:2048]'

            img = trim_image(img, 
                        fits_section=trimsec,
                        add_keyword={'trim': f"{get_date} Trim is {trimsec}"})
            
            #.Atualizando o header
            imsz = img.shape
            img.header['DATASEC'] = f"[1:{imsz[1]},1:{imsz[0]}]"
            del img.header['BIASSEC']
            del img.header['TRIMSEC']
            del img.header['BZERO']
            del img.header['BSCALE']

            skip[ne] = False
        else: 
            proc_string+="-"


        #.Realizando correcao de BIAS
        if ('ZEROCOR' not in img.header) and (master_bias is not None): 
            proc_string+="z"
            img = subtract_bias(img, 
                        mbias_ccd[ext],
                        add_keyword={'ZEROCOR': f"{get_date} Zero is {bstr}[{ext}]"})

            skip[ne] = False
        else:
            proc_string+="-"

        #.Realizando correcao de FLAT
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

        #.Juntando amplificadores
        if can_merge:
            proc_string+="m"
            #..coletando dimensoes e posicao relativa deste amplificador
            ampos_x, ampos_y = iaf.iraf2python(img.header['DETSEC'])
            ampos_x = (np.array(ampos_x)/bin_x).astype(int) + xgap[ne]
            ampos_y = (np.array(ampos_y)/bin_y).astype(int) + ygap[ne]
            
            #..escrevendo dados deste amplificador no objeto CCDDdata da imagem final
            img_merge[ampos_y[0]:ampos_y[1],ampos_x[0]:ampos_x[1]] = img.data
        else:
            proc_string+="-"

        #.Se alguma operacao foi realizada sobre a imagem, atualizar dados e cabecalho no HDU
        if not skip[ne]: 
            hdul[ext].data = img.data.astype(np.float32)
            hdul[ext].header = img.header
    

    #.Salvando a imagem processada 
    if can_merge:
        
        #..preparando o header da imagem combinada
        if image_exts[0] != 0: 
            hdr = hdul[0].header
            hdr.extend(hdul[1].header, unique=True)
        else: hdr = hdul[1].header
        #..removendo keywords que nao serao mais necessarias
        keystodel = ['DATASEC', 'CCDSEC', 'AMPSEC', 'DETSEC', 'NEXTEND', 'EXTNAME']
        for keyw in keystodel: del hdr[keyw]
        #..ajustando keywords para refeletir o novo estado da imagem
        hdr['DETSIZE']=f"[1:{imsz_x},1:{imsz_y}]"
        hdr['CCDSIZE']=f"[1:{imsz_x},1:{imsz_y}]"
        hdr.insert('NAXIS', ('NAXIS1', imsz_x, "Axis length"), after=True)
        hdr.insert('NAXIS1', ('NAXIS2', imsz_y, "Axis length"), after=True)   
        hdr.append(('AMPMERGE', f"{get_date} Merged {len(image_exts)} amps"))

        #..construindo o objeto CCDData para armazenar a imagem final
        img = CCDData(img_merge, meta=hdr, unit='adu')
        #..escrevendo para o arquivo (overwrite)
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
    
    master_flat = master_flats_list[flat_idx]

    #.executando a reducao da imagem
    reduce_image(image_file, 
                  master_bias = master_bias,
                  master_flat = master_flat,
                  shared_memory = True,
                  merge_amplifiers = merge_amplifiers)


def process_bias(ifc, 
                 combined_bias='master_bias.fits',
                 delete_bias=True,
                 multiprocessing=True):
    
    #.separando imagens de BIAS em uma lista 
    bias_ifc = ifc.filter(obstype='BIAS|ZERO', regex_match=True)
    file_list = bias_ifc.files
    
    #.corrigindo OVERSCAN nas imagens de BIAS
    if multiprocessing: 
        with Pool() as pool:
            pool.map(partial(reduce_image, master_bias=None, master_flat=None,
                             merge_amplifiers=False), file_list)
    else: 
        for file in file_list: 
            reduce_image(file)

    #.combinando imagens de BIAS em um 'master_bias'
    combine_bias(file_list, delete_images=delete_bias, output=combined_bias, 
                 multiprocessing=multiprocessing)

    ifc.refresh()
    

def combine_bias(image_list, 
                 delete_images=True, 
                 output='master_bias.fits',
                 multiprocessing=True):
    
    #.tratando excessoes
    nbias = len(image_list)
    if (not os.path.isfile(output)):
        if (nbias > 0): 
            print(f".Creating '{os.path.basename(output)}': {nbias} images")
        else: sys.exit(".ABORTING 'master bias' creation: no BIAS images found")
    else: 
        print(f".Using '{os.path.basename(output)}' image found in directory")
        return

    #.criando imagem de saida (a partir da 1a da lista)
    mbias = fits.open(image_list[0])

    #.obtendo extensoes com imagem
    image_exts = iaf.image_extensions(mbias, is_hdu=True)

    #.combinando bias (por extensao)
    if multiprocessing:
        with Pool() as pool:
            result = pool.map(partial(combine_bias_extension, bias_list=image_list), 
                              image_exts)
    else:
        result=[]
        for ext in image_exts:
            result.append(combine_bias_extension(ext, bias_list=image_list))
    
    #.colocando as extensoes combinadas na imagem de saida 
    for res in result:
        ext = res[0]
        mbias[ext].data = res[1].data
        mbias[ext].header = res[1].header

    #.salvando imagem de saida
    mbias[0].header['FILENAME']=output
    mbias.writeto(output, overwrite=True)
    mbias.close()

    if delete_images: 
        for file in image_list: 
            try: os.remove(file)
            except OSError as e: print("Error: %s - %s." % (e.filename, e.strerror))


def combine_bias_extension(ext, bias_list):

    #..agrupando as imagens de bias deste amplificador
    ccd_list = [ CCDData.read(bias_list[j], hdu=ext, unit="adu")
                 for j in np.arange(len(bias_list)) ]

    #..preparando objeto combinador
    comb = Combiner(ccd_list, dtype=np.float32)
    #..processando opcoes do combinador
    comb.sigma_clipping(low_thresh=3, high_thresh=3, func='median', dev_func='mad_std')
    #..realizando combinacao
    comb_bias = comb.average_combine()

    #..pegando o cabecalho da primeira imagem e atualizando
    comb_bias.header = fits.getheader(bias_list[0], ext=ext)
    for n,imgn in enumerate(bias_list, start=1): 
        fstr = os.path.basename(imgn)
        comb_bias.header.append((f"IMCMB{n:03}", f"{fstr}[{ext}]"))
    comb_bias.header.append(('NCOMBINE', len(ccd_list), '# images combined'))

    #..retornando imagem processada desta extensao
    return (ext, comb_bias)


def process_flat(ifc, 
                 filter_keywords='filter1,filter2', 
                 master_bias='master_bias.fits',
                 combined_flats='master_flat.fits',
                 delete_flats=True,
                 multiprocessing=True):

    combined_flats = combined_flats.split('.fits')[0]
  
    #.separando imagens de FLAT em uma lista 
    flat_ifc = ifc.filter(obstype='FLAT|SFLAT|DFLAT', regex_match=True)
    file_list = flat_ifc.files

    #.aplicando correcao de OVERSCAN e BIAS nas imagens de FLAT
    if multiprocessing: 
        #..carregando 'master bias' em um buffer de memoria
        mbias_ccd = iaf.image_to_memory(master_bias)
        #..executando a reducao com multiprocessamento
        with Pool() as pool:
            pool.map(partial(reduce_image, master_bias=mbias_ccd, 
                             master_flat=None, shared_memory=True,
                             merge_amplifiers=False), file_list)
        #..liberando buffers de memoria 
        iaf.unlink_memory(mbias_ccd)
    else: 
        #..carregando imagem 'master bias' em um objeto CCDDATA
        mbias_ccd = iaf.image_to_ccddata(master_bias)   
        #..executando a reducao imagem por imagem
        for file in file_list: 
            reduce_image(file, master_bias=mbias_ccd, 
                         master_flat=None, merge_amplifiers=False)

    #.combinando FLATS por filtro
    filters = iaf.ifc_filters(flat_ifc, filter_keywords=filter_keywords, 
                          obstype_selection='*')
    if isinstance(filter_keywords, str): keywds = filter_keywords.split(',')
    else: keywds = filter_keywords

    for fn,filt in enumerate(filters, start=1):

        #.separando lista de imagens de FLAT neste filtro
        filt_list = list(flat_ifc.files_filtered(**{keywds[0]: filt}))
        for i in np.arange(1,len(keywds)): 
            filt_list.append(list(flat_ifc.files_filtered(**{keywds[i]: filt})))
        filt_list = list(filter(None, filt_list))

        print(f"Creating '{os.path.splitext(os.path.basename(combined_flats))[0]}"+
            f"{fn}.fits': {len(filt_list)} images ({filt})")

        #.combinando imagens deste filtro em um 'master flat'
        combine_flat(filt_list, delete_images=delete_flats, 
                     output=f"{combined_flats}{fn}.fits", 
                     multiprocessing=multiprocessing)

    #.atualizando ImageFileCollection
    ifc.refresh()


def combine_flat(image_list, 
                 delete_images=True, 
                 output='master_flat.fits', 
                 multiprocessing=True):

    #.criando imagem de saida (a partir da 1a da lista)
    mflat = fits.open(image_list[0])

    #.obtendo extensoes com imagem
    image_exts = iaf.image_extensions(mflat, is_hdu=True)

    #.gerando fatores de escala entre os flats da lista
    fscales = iaf.flat_scale(image_list, normalize=True)

    #.combinando bias (por extensao)
    if multiprocessing:
        with Pool() as pool:
            result = pool.map(partial(combine_flat_extension, flat_list=image_list, 
                                      scaling=fscales), image_exts)
    else:
        result=[]
        for ext in image_exts:
            result.append(combine_flat_extension(ext, flat_list=image_list, 
                                                 scaling=fscales))
    
    #.colocando as extensoes combinadas na imagem de saida 
    for res in result:
        ext = res[0]
        mflat[ext].data = res[1].data
        mflat[ext].header = res[1].header

    #.salvando imagem de saida
    mflat[0].header['FILENAME']=output
    mflat.writeto(output, overwrite=True)
    mflat.close()

    if delete_images: 
        for file in image_list: 
            try: os.remove(file)
            except OSError as e: print("Error: %s - %s." % (e.filename, e.strerror))


def combine_flat_extension(ext, flat_list, scaling=None):
    
    if scaling is None: fscales = np.full(len(flat_list),1)
    else: fscales = np.array(scaling)
    
    #.agrupando as imagens de flat deste amplificador
    ccd_list = [ CCDData.read(flat_list[j], hdu=ext, unit="adu")
                 for j in np.arange(len(flat_list)) ]
    #.escalonando (manualmente) flats pelo valor mediano
    # (o metodo comb.sigma_clipping nao funciona se a escala for feita pelo comb.scaling)
    for k in np.arange(len(flat_list)): ccd_list[k].data /= fscales[k]
    #.preparando objeto combinador
    comb = Combiner(ccd_list, dtype=np.float32)
    #.processando opcoes do combinador
    comb.sigma_clipping(low_thresh=2., high_thresh=2., func='median', dev_func='mad_std')
    #.realizando combinacao
    comb_flat = comb.median_combine()

    #..pegando o cabecalho da primeira imagem e atualizando
    comb_flat.header = fits.getheader(flat_list[0], ext=ext)
    for n,imgn in enumerate(flat_list, start=1): 
        fstr = os.path.basename(imgn)
        comb_flat.header.append((f"IMCMB{n:03}", f"{fstr}[{ext}] (scale={fscales[n-1]:.1f})"))
    comb_flat.header.append(('NCOMBINE', len(ccd_list), '# images combined'))
    comb_flat.header.append(('FLATNORM', fscales[0], '# normalization scale'))

    #..retornando imagem processada desta extensao
    return (ext, comb_flat)


def process_images(ifc, 
                  master_bias='master_bias.fits',
                  master_flat='master_flat.fits',
                  filter_keywords='filter1,filter2', 
                  multiprocessing=True):
    
    master_flat = master_flat.split('.fits')[0]

    #.identificando 'master flats' presentes na pasta
    mflat_list = np.sort(glob.glob(master_flat+'*.fits'))
    flat_filters = iaf.image_filters(mflat_list)

    #.separando imagens de CIENCIA em uma lista 
    object_ifc = ifc.filter(obstype='OBJECT', regex_match=True)
    file_list = object_ifc.files

    #.selecionando o 'master flat' correto para cada imagem
    obj_filters = iaf.find_filter(file_list, flat_filters, filter_keywords)

    #.reduzindo imagens de ciencia
    if multiprocessing:
        #..carregando 'master bias' e 'master flat' em um buffer de memoria
        mbias_ccd = iaf.image_to_memory(master_bias, name='mbias')
        mflat_ccd = [iaf.image_to_memory(flat, name='mflat'+flat.split('.fits')[0][-1]) 
                     for flat in mflat_list]
        #.executando a reducao com multiprocessamento
        with Pool() as pool:
            pool.starmap(partial(reduce_image_mp, master_bias=mbias_ccd,
                         master_flats_list=mflat_ccd), 
                         zip(file_list,obj_filters))
        #..liberando buffers de memoria 
        iaf.unlink_memory(mbias_ccd)
        for flat in mflat_ccd: iaf.unlink_memory(flat)
    else:
        #.carregando 'master bias' e 'master flat' em objetos CCDDATA
        mbias_ccd = iaf.image_to_ccddata(master_bias)
        mflat_ccd = [iaf.image_to_ccddata(image) for image in mflat_list]
        #.executando a reducao imagem a imagem
        for i,file in enumerate(file_list):
            reduce_image(file, 
                         master_bias=mbias_ccd, 
                         master_flat=mflat_ccd[obj_filters[i]],
                         merge_amplifiers=True)

    #.atualizando ImageFileCollection
    ifc.refresh()

