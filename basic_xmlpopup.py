import numpy as np
from PIL import Image
from pathlib import Path
import tiffile as tf
import time
import cv2
import napari
import os
import PySimpleGUI as sg
from datetime import datetime
from apeer_ometiff_library import io, processing, omexmlClass
#import imgfileutils as imf
import zarr
import xmltodict
import webbrowser

# 16>8
def map_uint16_to_uint8(img, lower_bound=None, upper_bound=None):
    '''
    Map a 16-bit image trough a lookup table to convert it to 8-bit.

    Parameters
    ----------
    img: numpy.ndarray[np.uint16]
        image that should be mapped
    lower_bound: int, optional
        lower bound of the range that should be mapped to ``[0, 255]``,
        value must be in the range ``[0, 65535]`` and smaller than `upper_bound`
        (defaults to ``numpy.min(img)``)
    upper_bound: int, optional
       upper bound of the range that should be mapped to ``[0, 255]``,
       value must be in the range ``[0, 65535]`` and larger than `lower_bound`
       (defaults to ``numpy.max(img)``)

    Returns
    -------
    numpy.ndarray[uint8]
    '''
    if lower_bound is None:
        lower_bound = np.min(img)
    if upper_bound is None:
        upper_bound = np.max(img)
    if lower_bound >= upper_bound:
        raise ValueError(
            '"lower_bound" must be smaller than "upper_bound"')
    lut = np.concatenate([
        np.zeros(lower_bound, dtype=np.uint16),
        np.linspace(0, 255, upper_bound - lower_bound).astype(np.uint16),
        np.ones(2**16 - upper_bound, dtype=np.uint16) * 255
    ])
    return lut[img].astype(np.uint8)

# text popup
def popup_text(filename, text):
    layout = [
        [sg.Multiline(text, size=(80, 25)),],
    ]
    win = sg.Window('Metadata', layout, modal=True, finalize=True)
    while True:
        event, values = win.read()
        if event == sg.WINDOW_CLOSED:
            break
    win.close()

# Theme color
sg.theme('LightGrey1')


# flayout sizes
fs_w = 20
fs_h = 1

# symbols
SYMBOL_UP =    '▲'
SYMBOL_DOWN =  '▼'

# collapse function
def collapse(layout, key):
    """
    Helper function that creates a Column that can be later made hidden, thus appearing "collapsed"
    :param layout: The layout for the section
    :param key: Key used to make this seciton visible / invisible
    :return: A pinned column that can be placed directly into your layout
    :rtype: sg.pin
    """
    return sg.pin(sg.Column(layout, key=key))

# extra meta data window
optional_meta = [
    [sg.Text('Name', size =(fs_w, fs_h)), sg.InputText(default_text=None,key='-Name-')],
    [sg.Text('Description', size =(fs_w, fs_h)), sg.InputText(default_text=None,key='-Description-')],
    [sg.Text('Acquisition date', size =(fs_w, fs_h)), sg.InputText(key='-AcquisitionDate-')],  
    #default_text=datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
    ]


# layout
layout = [
 	[sg.Text('Choose a file to convert:')],
	[sg.Input(key='-image path-',enable_events=True),sg.FileBrowse(target='-image path-')],
    [sg.Text('Choose a metadata file (optional):')],
    [sg.Input(key='-xml path-',enable_events=True),sg.FileBrowse(target='-xml path-'),sg.Button('View'),sg.Button('Autofill')],
	[sg.Text('Output file name and location:')],
    [sg.InputText(key='-output name-'),sg.FileBrowse(target='-output name-')],
	[sg.HorizontalSeparator()],
    [sg.Text('Enter image dimensions, bit depth, and pixel size:')],
    [sg.Text('Width', size =(fs_w, fs_h)), sg.InputText(key='-image width-')],
    [sg.Text('Height', size =(fs_w, fs_h)), sg.InputText(key='-image height-')],
    [sg.Text('Pixel size:', size =(fs_w, fs_h)), sg.InputText(key='-pixel size-')],
    [sg.Text('Pixel size unit:', size =(fs_w, fs_h)),sg.Combo(['m','µm','nm'],default_value='m',key='-pixel size unit-')],
    [sg.Text('Bit depth:',size =(fs_w, fs_h)),sg.Combo([8,16,'16>8'],default_value=8,key='-bit depth-')],
    [sg.HorizontalSeparator()],
    [sg.Text('Enter pyramidal OME-TIFF parameters:')],
    [sg.Text('Downsample factor:', size =(fs_w, fs_h)), sg.InputText(default_text='2',key='-Downsample-')],
    [sg.Text('Number of levels:', size =(fs_w, fs_h)), sg.InputText(default_text='5',key='-levels-')],
    [sg.Text('Tile size (multiple of 16):', size =(fs_w, fs_h)), sg.InputText(default_text='256',key='-tile size-')],
    [sg.Text('Compression:',size =(fs_w, fs_h)),sg.Combo(['Uncompressed','jpeg'],default_value='Uncompressed',key='-Compress-')],
    [sg.Text(SYMBOL_DOWN, enable_events=True, key='-open extra-'),sg.Text('Additional OME-TIFF metadata', enable_events=True, k='-open extra text-')],[collapse(optional_meta, '-extra-')],
    [sg.Submit(), sg.Cancel()],
    #[sg.ProgressBar(1000, orientation='h', size=(20, 20), key='-progressbar-')],  # cool option
    [sg.HorizontalSeparator()],
    [sg.Text('View the converted (or any) OME-TIFF:')],
    [sg.InputText(key='-input view-'),sg.FileBrowse()],
    [sg.Button("View",key='-view-'),sg.Button("View ome-xml",key='-view omexml-')],
    [sg.Text('nanotomy.org', click_submits=True,enable_events=True,key='-nanotomy-',font=("Helvetica", 10), text_color='black')]
]


# initialize extra section state
opened_extra = False

# get user entered values
window = sg.Window('RAW to OME-TIFF', layout)

while True:
    event, values = window.read()
    if event == sg.WINDOW_CLOSED or event == 'Cancel':
        break
    elif event == '-nanotomy-':
        webbrowser.open('https://www.nanotomy.org')
    elif event == 'View':
        filename = values['-xml path-']
        if Path(filename).is_file() and filename.endswith('.xml'):
            try:
                with open(filename, "rt", encoding='utf-8') as f:
                    text = f.read()
                popup_text(filename, text)
            except:
                sg.popup('XML file not found')
        else:
            sg.popup('XML file not found')

    # output name and xml path based on input file
    elif event == '-image path-':
        if values['-image path-'].endswith('raw') is False:
            sg.popup('Not a raw image.')
        else:
            window.Element('-output name-').Update(values['-image path-'][:-3] + 'ome.tiff')
            if os.path.exists(values['-image path-'][:-3] + 'xml'):
                window.Element('-xml path-').Update(values['-image path-'][:-3] + 'xml') 

    # Autofill
    elif event == '-xml path-' or event == 'Autofill':
        if values['-xml path-'].endswith('.xml') is False:
            sg.popup('Not an XML file. Select another.')
        else:
            try:
                # parse the XML (ATLAS type only now)
                with open(values['-xml path-']) as fd:
                    dic = xmltodict.parse(fd.read())
                auto_width = dic['RawExport']['Width']
                auto_height = dic['RawExport']['Height']
                auto_pixelsize = dic['RawExport']['PixelSize']['Value']
                auto_pixelsizeunit = dic['RawExport']['PixelSize']['Unit']
                auto_bitdepth = dic['RawExport']['BitPerSample']


                # update the corresponding fields
                window.Element('-image width-').Update(auto_width)
                window.Element('-image height-').Update(auto_height)
                window.Element('-pixel size-').Update(auto_pixelsize)
                window.Element('-bit depth-').Update(auto_bitdepth)
                window.Element('-pixel size unit-').Update(auto_pixelsizeunit)
            except:
                sg.popup('XML file not found or incorrect')

    elif event.startswith('-open extra-'):
        opened_extra = not opened_extra
        window['-open extra-'].update(SYMBOL_DOWN if opened_extra else SYMBOL_UP)
        window['-extra-'].update(visible=opened_extra)

    elif event == 'Submit':
        # input file (RAW) and output
        input = values['-image path-']
        output = values['-output name-']

        # time the conversion
        start_time_1 = time.time()


        # read raw file
        ROWS = int(values['-image height-'])
        COLS = int(values['-image width-'])
        fin = open(input)

        # Loading the input image
        if values['-bit depth-'] == 8:
        	image = np.fromfile(fin, dtype = np.uint8, count = ROWS*COLS)
        elif values['-bit depth-'] == 16:
        	image = np.fromfile(fin, dtype = np.uint16, count = ROWS*COLS)
        elif values['-bit depth-'] == '16>8':
            image = np.fromfile(fin, dtype = np.uint16, count = ROWS*COLS)


        # Conversion from 1D to 2D array
        image.shape = (image.size // COLS, COLS)

        if values['-bit depth-'] == '16>8':
            #convert 16bit to 8bit
            image = map_uint16_to_uint8(image)

        end_time_1 = time.time()

        ##################################### OME TIFF #########################################
        # parameters 
        downsample_factor = int(values['-Downsample-'])     # downsample factor of pyramid levels
        lvl = int(values['-levels-'])                   	# pyramid levels
        tile_size = int(values['-tile size-'])            	# Multiples of 16

        compression = None if values['-Compress-'] == 'Uncompressed' else values['-Compress-']


        # get pixel size
        pixunit = values['-pixel size unit-']
        unit_factor = 1e-6 if pixunit == 'µm' else 1e-9 if pixunit == 'nm' else 1

        pixel_size =  float(values['-pixel size-'])*unit_factor
        pixel_size_x = pixel_size
        pixel_size_y = pixel_size
        pixel_size_z = pixel_size

        # Write tiff file
        start_time_2 = time.time()
        with tf.TiffWriter(output, bigtiff=True) as tif:
            # use tiles and JPEG compression. 
            options = {'tile': (tile_size, tile_size),
                       'compress': compression,
                       'metadata':{'PhysicalSizeX': pixel_size_x*1e9, 'PhysicalSizeXUnit': 'nm',
                                   'PhysicalSizeY': pixel_size_y*1e9, 'PhysicalSizeYUnit': 'nm',
                                   'axes': 'YX','Description':values['-Description-'],
                                   'AcquisitionDate':values['-AcquisitionDate-'],
                                   'Name':values['-Name-']}}
            # save the base image (the original resolution)
            tif.write(image, subifds=lvl, **options)
            
            # iteratively generate and save the pyramid levels to the SubIFDs
            image2 = image
            for _ in range(lvl):
                image2 = cv2.resize(
                    image2,
                    (image2.shape[1] // downsample_factor, image2.shape[0] // downsample_factor),
                    interpolation=cv2.INTER_LINEAR
                )
                tif.write(image2, **options)

        # end timer and print info      
        end_time_2 = time.time()
        size_gb = os.path.getsize(output)*10e-10

        # update the path to visualize to the converted file
        window.Element('-input view-').Update(values['-output name-']) 

        sg.popup('Your conversion was completed in %02d seconds. File size: %.2f GB.' % (end_time_2-start_time_1,size_gb))

    elif event == '-view-':
        filename = values['-input view-']
        store = tf.imread(filename, aszarr=True)
        zgroup = zarr.open(store, mode='r')
        print(zgroup.info)
        print(zgroup[0].info)
        data = [zgroup[int(dataset['path'])] for dataset in zgroup.attrs['multiscales'][0]['datasets']]
        viewer = napari.view_image(data, rgb=False)  # contrast_limits=[0, 255]
        napari.run()
        store.close()

    elif event == '-view omexml-':
        filename = values['-input view-']
        with tf.TiffFile(filename) as tif:
            for page in tif.pages:
                print(page.tags)
                 #print(page.image_desription.name)
                 #print(page.image_desription.value)
         
         #with tf.TiffFile(filename) as tif:
         #   for page in tif.pages:
         #        for tag in page.tags:
         #           tag_name, tag_value = tag.name, tag.value
         #           print(tag_name,tag_value)
        
        #omexml_string = tif.image_description #.decode("utf-8")
        #metadata = omexmlClass.OMEXML(omexml_string)
        #print(omexml_string)
        #print(metadata)


window.close()