import nibabel as nib
from nilearn import image, plotting
import nibabel.freesurfer as fs
from nilearn.image import resample_to_img
import matplotlib.colors as mcolors
from nibabel.freesurfer import read_annot

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import pandas as pd

import numpy as np
import matplotlib.image as mpimg
import re
import os
#from dotenv import load_dotenv
import flywheel
import logging
log = logging.getLogger(__name__)
# from reportlab.lib.pagesizes import letter
# from reportlab.pdfgen import canvas
# from reportlab.lib.utils import ImageReader
from tqdm import tqdm
from datetime import datetime
from IPython.display import display, clear_output


today = datetime.now()


import skimage
import plotly.express as px
from skimage.transform import resize
from ipywidgets import interact, FloatSlider, IntSlider


import os
import subprocess
# Source zshrc and get the env var you want
def get_env_from_zshrc(var_name):
    command = f"source ~/.zshrc && echo ${var_name}"
    result = subprocess.run(['zsh', '-c', command], stdout=subprocess.PIPE, text=True)
    return result.stdout.strip()

api_key = get_env_from_zshrc('FW_CLI_API_KEY')
# Connect to your Flywheel instance

fw = flywheel.Client(api_key=api_key)
display(f"User: {fw.get_current_user().firstname} {fw.get_current_user().lastname}")

def preprocess_nifti(nifti_path, target_height=300):
    # load data
    data = nib.load(nifti_path).get_fdata()
    data = np.nan_to_num(data)
    xyz = data.shape

    # Normalise entire data arrays
    min_val, max_val = np.min(data), np.max(data)
    if max_val > min_val:
        data = (255 * (data - min_val) / (max_val - min_val))
    else:
        data = np.zeros_like(data)  # Black image if constant
        
    # Determine orientation and choose slices correctly
    if 'SAG' in nifti_path.upper():
        orientation = "sag"
        plane = (1, 2)
        ax = 0
    elif 'COR' in nifti_path.upper():
        orientation = "cor"
        plane = (0, 2)
        ax = 1
    elif 'AXI' in nifti_path.upper():
        orientation = "axi"
        plane = (0, 1)
        ax = 2
    
    # Resize the image to target height
    w, h = xyz[plane[0]], xyz[plane[1]]
    scale_factor = target_height / h
    new_width = int(w * scale_factor)
    
    # Create new shape and resize
    new_shape = list(xyz)
    new_shape[plane[0]], new_shape[plane[1]] = new_width, target_height
    data = skimage.transform.resize(data, new_shape, mode='constant', preserve_range=True)
    
    return data, plane[0], ax
        
def nifti_overlay_animation(colored_data,native_path, seg_path, sub, outliers, orientation='axial', target_height=300, alpha=0.4,cmap="jet"):
    # Load NIfTI files
    native_img = nib.load(native_path)
    seg_img = nib.load(seg_path)

    native_data = native_img.get_fdata()
    seg_data = seg_img.get_fdata()

    # Rescale volumes to same shape (if needed)
    if native_data.shape != seg_data.shape:
        raise ValueError("Native and segmentation volumes must have the same shape.")

    # Normalize native scan to 0-255 (grayscale)
    native_data = native_data - np.min(native_data)
    native_data = (native_data / np.max(native_data)) * 255
    native_data = native_data.astype(np.uint8)

    # Resize to target height (optional)
    scale = target_height / native_data.shape[0]
    new_shape = (target_height, int(native_data.shape[1]*scale), native_data.shape[2])
    native_data_resized = resize(native_data, new_shape, preserve_range=True, order=1).astype(np.uint8)
    seg_data_resized = resize(seg_data, new_shape, preserve_range=True, order=0).astype(np.uint8)

    # Map orientation to slicing axes
    if orientation == 'axial':
        axis = 2
        rotate_times = 1
    elif orientation == 'coronal':
        axis = 1
        rotate_times = 1
    elif orientation == 'sagittal':
        axis = 0
        rotate_times = 1
    else:
        raise ValueError("Orientation must be 'axial', 'coronal', or 'sagittal'.")
    
    # Create overlay for each slice
    overlays = []
    num_slices = native_data_resized.shape[axis]
    for i in range(num_slices):
        if axis == 2:
            background = np.stack([native_data_resized[:, :, i]]*3, axis=-1)
            mask = seg_data_resized[:, :, i]
        elif axis == 1:
            background = np.stack([native_data_resized[:, i, :]]*3, axis=-1)
            mask = seg_data_resized[:, i, :]
        elif axis == 0:
            background = np.stack([native_data_resized[i, :, :]]*3, axis=-1)
            mask = seg_data_resized[i, :, :]

        # Rotate for display
        background = np.rot90(background, k=rotate_times)
        mask = np.rot90(mask, k=rotate_times)

        # Apply colormap to mask
        mask_rgb = plt.get_cmap(cmap)(mask)[:, :, :3]  # Drop alpha channel
        mask_rgb = (mask_rgb * 255).astype(np.uint8)

        # Alpha blend
        overlay = background.copy()
        overlay[mask > 0] = ((1 - alpha) * background[mask > 0] + alpha * mask_rgb[mask > 0]).astype(np.uint8)

        overlays.append(overlay)

    # Convert to 4D array for plotly
    overlays_np = np.stack(overlays, axis=0)

    # Plotly animation
    print(f'Loading overlaid animation for {sub}, this may take a few seconds...')

    # num_slices = overlays_np.shape[0]
    # middle_index = num_slices // 2
    # rolled_data = np.roll(overlays_np, -middle_index, axis=0)

    # fig = px.imshow(rolled_data, animation_frame=0, aspect='equal')

    fig = px.imshow(overlays_np, animation_frame=0, aspect='equal')
    fig.update_layout(
    title={
        'text': f"<b>⚠️ Outlier ROIs (z-score and cov):</b><br>{outliers}",
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': dict(size=10)
    },
    autosize=True,
    height=600,
    coloraxis_showscale=False
)
    
    fig.update_layout(autosize=True, height=600, coloraxis_showscale=False)
    fig.update_layout(xaxis=dict(showticklabels=False), yaxis=dict(showticklabels=False))
    fig.update_layout(dragmode=False)

    # num_slices = overlays_np.shape[0]
    # middle_index = num_slices // 2

    # fig.data[0].z = overlays_np[middle_index]
    # fig.layout.sliders[0]['active'] = middle_index
    # fig.layout.updatemenus[0]['buttons'][0]['args'][1]['frame']['redraw'] = True

    # Show the plot
    fig.show()
    return fig

def download_analysis_files(asys,sub_label,ses_label,str_pattern,download_dir):
    download_dir = Path(f'{download_dir}/{sub_label}/{ses_label}/')
    download_dir.mkdir(parents=True,exist_ok=True)
    input_file= asys.inputs[0]
    download_path = os.path.join(download_dir , input_file.name)
    fw.download_input_from_analysis( asys.id, input_file.name, download_path)
    print("Downloaded input file:",download_path)
    
    print([file.name for file in asys.files])
    for file in asys.files:            
        if file.name.endswith('nii.gz') : #re.search(str_pattern, file.name) and #or re.search(rf'{input_gear}.*\.nii.gz', file.name):

        #if file.name.endswith('aparc+aseg.nii.gz') or file.name.endswith('synthSR.nii.gz') or re.search('ResCNN.*\.nii.gz', file.name) or re.search('mrr_fast.*\.nii.gz', file.name) or re.search('mrr-axireg.*\.nii.gz', file.name) or re.search('.*\.zip', file.name):
            parcellation = file
            print("Found ", file.name)
            if file :
                
                download_path = os.path.join(download_dir , parcellation.name)
                
                parcellation.download(download_path)
                print('Downloaded parcellation ',download_path)
                

def get_data(sub_label, ses_label, seg_gear, input_gear, v,download_dir,project_label="Botswana-MOTHEO",api_key=api_key):
    
    from fw_client import FWClient
    #api_key = os.environ.get('FW_CLI_API_KEY')
    fw_ = FWClient(api_key=api_key)

    project = fw.projects.find_first(f'label={project_label}')
    display(f"Project: {project.label}")
    project = project.reload()

    subject = project.subjects.find_first(f'label="{sub_label}"')
    subject = subject.reload()
    sub_label = subject.label
    
    session = subject.sessions.find_first(f'label="{ses_label}"')
    session = session.reload()
    ses_label = session.label
    print(seg_gear, input_gear)
    analyses = session.analyses

    seg_parc_map = {"recon-all-clinical":"aparc+aseg","minimorph":"segmentation"}
    str_pattern = seg_parc_map[seg_gear]

    # If there are no analyses containers, we know that this gear was not run
    if len(analyses) == 0:
        run = 'False'
        status = 'NA'
        print('No analysis containers')
    else:
        try:
            if input_gear.startswith("gambas"):
                seg_gear = seg_gear + "_gambas"
                print("Looking for anaylyses from ", seg_gear)

            matches = [asys for asys in analyses if asys.label.startswith(seg_gear) and asys.job.get('state') == "complete"]
            print("Matches: ", len(matches),[asys.label for asys in matches] )
            # If there are no matches, the gear didn't run
            if len(matches) == 0:
                run = 'False'
                status = 'NA'
                print(f"Did not find any matched, {seg_gear} did not run.")
            # If there is one match, that's our target
            elif len(matches) == 1:
                run = 'True'
                #status = matches[0].job.get('state')
                #print(status)
                #print("Inputs ", matches[0])
                asys=matches[0]
                download_analysis_files(asys,sub_label,ses_label,str_pattern,download_dir)


            else:
                last_run_date = max([asys.created for asys in matches])
                last_run_analysis = [asys for asys in matches if asys.created == last_run_date and asys.job.get('state') == "complete"]

                # There should only be one exact match
                last_run_analysis = last_run_analysis[0]

                run = 'True'
                #status = last_run_analysis.job.get('state')
                asys=last_run_analysis
                download_analysis_files(asys,sub_label,ses_label,str_pattern,download_dir)

        except Exception as e:
            print(f"Exception caught for {sub_label} {ses_label}: ", e)
            


def load_ratings(RATINGS_FILE,metrics):
    if os.path.exists(RATINGS_FILE):
        return pd.read_csv(RATINGS_FILE)
    
    return pd.DataFrame(columns=["User", "Timestamp", "Project", "Subject", "Session"] + metrics)
    
# Function to simplify acquisition labels
def simplify_label(label):
    # Initialize empty result
    result = []
    
    # Check for orientation
    if 'AXI' in label.upper():
        result.append('AXI')
    elif 'COR' in label.upper():
        result.append('COR')
    elif 'SAG' in label.upper():
        result.append('SAG')
        
    elif 'Localizer' in label:
        result.append('LOC')
        
    # Check for T1/T2
    if 'T1' in label.upper():
        result.append('T1')
    elif 'T2' in label.upper():
        result.append('T2')
        
    # Check for fast vs. standard labels
    if 'FAST' in label.upper():
        result.append('FAST')
    elif 'STANDARD' in label.upper():
        result.append('STANDARD')
        
    # Check for Gray_White
    if 'Gray' in label.upper():
        result.append('GrayWhite')
        
    # Return combined result or original label if no matches
    return '_'.join(result) if result else label

def save_rating(ratings_file, responses,project,metrics):
    df = load_ratings(ratings_file,metrics)
    print(df.columns)
    print(responses, len(responses))
    new_entry = pd.DataFrame([responses], 
                              columns=df.columns)
    
    df = pd.concat([df, new_entry], ignore_index=True)
    # df.loc[:, 'Acquisition'] = df['Acquisition'].apply(simplify_label)
    
    df.to_csv(ratings_file, index=False)
    log.info(f"\nSaved rating: {ratings_file}")
    custom_name = ratings_file.split('/')[-1]
    #project.upload_file(ratings_file, filename=custom_name)
    log.info("QC responses have been uploaded to the project information tab.")

    
    
    
def find_csv_file(directory, username):
    username_cleaned = username.replace(" ", "")
    
    for root, _, files in os.walk(directory):
        for file in files:
            if username_cleaned in file and file.endswith(".csv"):
                return os.path.join(root, file)  # Return the first matching file found

    return None  # No matching file found