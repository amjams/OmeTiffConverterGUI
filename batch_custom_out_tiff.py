# batch mode but with a custom output location for the tiff (not raw) images

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
import zarr
import xmltodict
import webbrowser
from datetime import datetime



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
sg.theme('SandyBeach')     


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
    [sg.Text('Choose a source folder:')],
    [sg.Input(key='-folder path-',enable_events=True),sg.FolderBrowse(target='-folder path-',initial_folder=os.getcwd())],
    [sg.Text('Choose an output folder:')],
    [sg.Input(key='-output folder path-',enable_events=True),sg.FolderBrowse(target='-output folder path-',initial_folder=os.getcwd())],
    [sg.HorizontalSeparator()],
    [sg.Text('Enter pyramidal OME-TIFF parameters:')],
    [sg.Text('Downsample factor:', size =(fs_w, fs_h)), sg.InputText(default_text='2',key='-Downsample-')],
    [sg.Text('Number of levels:', size =(fs_w, fs_h)), sg.InputText(default_text='5',key='-levels-')],
    [sg.Text('Tile size (multiple of 16):', size =(fs_w, fs_h)), sg.InputText(default_text='256',key='-tile size-')],
    [sg.Text('Compression:',size =(fs_w, fs_h)),sg.Combo(['Uncompressed','jpeg'],default_value='Uncompressed',key='-Compress-')],
    [sg.HorizontalSeparator()],
    [sg.Text(SYMBOL_DOWN, enable_events=True, key='-open extra-'),sg.Text('Additional OME-TIFF metadata', enable_events=True, k='-open extra text-')],[collapse(optional_meta, '-extra-')],
    [sg.Submit(), sg.Cancel()],
    [sg.Text('nanotomy.org', click_submits=True,enable_events=True,key='-nanotomy-',font=("Helvetica", 10), text_color='black')]
]


# initialize extra section state
opened_extra = False

# get user entered values
window = sg.Window('TIFF to OME-TIFF BATCH CUSTOM OUTPUT FOLDER', layout)

while True:
    event, values = window.read()
    if event == sg.WINDOW_CLOSED or event == 'Cancel':
        break
    elif event == '-nanotomy-':
        webbrowser.open('http://nanotomy.org')


    elif event.startswith('-open extra-'):
        opened_extra = not opened_extra
        window['-open extra-'].update(SYMBOL_DOWN if opened_extra else SYMBOL_UP)
        window['-extra-'].update(visible=opened_extra)

    elif event == 'Submit':
        batch_folder_path = values['-folder path-']
        output_folder_path = values['-output folder path-']

        for image_name in os.listdir(batch_folder_path):
            if not image_name.startswith('.'):
                
                # time the conversion and start it
                start_time_1 = time.time()
                image_folder_path = os.path.join(batch_folder_path,image_name)
                image_path = os.path.join(image_folder_path,image_name+'.tif')
                #xml_path = os.path.join(image_folder_path,image_name+'.xml')
                output_path = os.path.join(output_folder_path,image_name+'.ome.tiff')

                # input file (TIFF) and output
                input = image_path
                output = output_path

                # print out and timing
                size_gb_in = os.path.getsize(input)*10e-10
                now = datetime.now()
                print('Image %s is starting to convert. TIFF image size: %.2f GB. Time: %s' % (image_name,size_gb_in,now.strftime("%H:%M:%S")))

                # read lzw tiff and get one channel (grayscale)
                image = tf.imread(input)
                try:
                    image = image[:,:,0]
                except:
                    pass

                end_time_1 = time.time()

                ##################################### OME TIFF #########################################
                # parameters 
                downsample_factor = int(values['-Downsample-'])     # downsample factor of pyramid levels
                lvl = int(values['-levels-'])                       # pyramid levels
                tile_size = int(values['-tile size-'])              # Multiples of 16

                compression = None if values['-Compress-'] == 'Uncompressed' else values['-Compress-']


                # get pixel size
                pixunit = 'nm'
                unit_factor = 1e-6 if pixunit == 'µm' else 1e-9 if pixunit == 'nm' else 1


                pixel_size =  2.5*unit_factor
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

                print('Image %s was converted in %02d minutes. File size: %.2f GB.' % (image_name,(end_time_2-start_time_1)/60,size_gb))
                #sg.popup('Image %s in %02d seconds. File size: %.2f GB.' % (image_name,end_time_2-start_time_1,size_gb))



window.close()