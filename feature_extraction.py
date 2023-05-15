#!/usr/bin/env python
# coding: utf-8

# # Feature extraction

# This example shows how to use the radiomics package and the feature extractor.
# The feature extractor handles preprocessing, and then calls the needed featureclasses to calculate the features.
# It is also possible to directly instantiate the feature classes. However, this is not recommended for use outside debugging or development. For more information, see `helloFeatureClass`.

# In[3]:


from __future__ import print_function
import sys
import os
import logging
import six
from radiomics import featureextractor, getFeatureClasses
import radiomics
import SimpleITK as sitk
import matplotlib.pyplot as plt
import pandas as pd
import traceback
from scipy.stats import pearsonr
import numpy as np
import seaborn as sns


# ## Define functions

# In[11]:


def get_settings():
    environment = os.environ.get('PYRADIOMICS_ENV', 'local')  # Default to 'local' if the environment variable is not set

    if environment == 'local':
        base_dir = "/Users/Gabriel/Desktop/MSc_Dissertation/pyRadiomics/Data/"
        params = "/Users/Gabriel/Desktop/MSc_Dissertation/pyRadiomics/Params.yaml"
        train_df = pd.read_csv("/Users/Gabriel/Desktop/MSc_Dissertation/pyRadiomics/train_data.csv") # Get data from training dataset
    elif environment == 'cluster':
        base_dir = "/cluster/project2/UCSF_PDGM_dataset/UCSF-PDGM-v3/"
        params = "/home/goliveir/pyRadiomics/Params.yaml"
        train_df = pd.read_csv("/home/goliveir/pyRadiomics/validation_data.csv") # Get data from training dataset -- change path to validation or test set to extract those features
    else:
        raise ValueError("Unknown environment: please set PYRADIOMICS_ENV to 'local' or 'cluster'")

    return base_dir, params, train_df, environment

def setup_logging(): # Create logfile called testLog.txt
    logger = radiomics.logger
    logger.setLevel(logging.DEBUG) # set level to DEBUG to include debug log messages in log file

    # Write out all log entries to a file
    handler = logging.FileHandler(filename='testLog.txt', mode='w')
    formatter = logging.Formatter('%(levelname)s:%(name)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def read_image_and_label(image_path, label_path, patient_id):
    Image = sitk.ReadImage(image_path)
    Label = sitk.ReadImage(label_path)

    if Image is None or Label is None:
        raise Exception(f"Error getting data for patient {patient_id}!")

    return Image, Label

def extract_features(params, Image, Label, patient_id):
    try:
        extractor = featureextractor.RadiomicsFeatureExtractor(params)
        result = extractor.execute(Image, Label)  
        return extractor, result
    except Exception as e:
        print(f"Could not extract features for patient {patient_id} retrying with label-value=2")
        settings = {'label': 2}
        extractor = radiomics.featureextractor.RadiomicsFeatureExtractor(additionalInfo=True, **settings)
        result = extractor.execute(Image, Label)  
        return extractor, result
    

def apply_filters_on_images(extractor):

    # By default, only 'Original' (no filter applied) is enabled. Optionally enable some image types:

    # extractor.enableImageTypeByName('Wavelet')
    # extractor.enableImageTypeByName('LoG', customArgs={'sigma':[3.0]})
    # extractor.enableImageTypeByName('Square')
    # extractor.enableImageTypeByName('SquareRoot')
    # extractor.enableImageTypeByName('Exponential')
    # extractor.enableImageTypeByName('Logarithm')

    # Alternative; set filters in one operation 
    # This updates current enabled image types, i.e. overwrites custom settings specified per filter. 
    # However, image types already enabled, but not passed in this call, are not disabled or altered.

    # extractor.enableImageTypes(Wavelet={}, LoG={'sigma':[3.0]})

    print('Filters applied on the image:')
    for imageType in extractor.enabledImagetypes.keys():
        print('\t' + imageType)
    return extractor

def calculate_features(extractor, Image, Label):
    extractor.disableAllFeatures() # These 2 lines will cause only firstorder features to be calculated
    extractor.enableFeatureClassByName('firstorder') 
    featureVector = extractor.execute(Image, Label)
    
    # Show calculated features
    print('Calculated features:')
    for featureName in featureVector.keys():
        print('Computed %s: %s' % (featureName, featureVector[featureName]))
    return featureVector

def fetch_data(base_dir, patient_id, img_type):
    folder_name = f"UCSF-PDGM-{patient_id:04d}_nifti"
    data_dir = os.path.join(base_dir, folder_name)
    
    image_types = {
        "DTI_eddy_FA": "_DTI_eddy_FA.nii.gz",
        "FLAIR": "_FLAIR_bias.nii.gz",
        "SWI": "_SWI.nii.gz",
        "T1": "_T1_bias.nii.gz",
        "T2": "_T2_bias.nii.gz",
        "DWI": "_DWI_bias.nii.gz",
        "T1_contrast": "_T1c_bias.nii.gz",
        "DTI_eddy_MD": "_DTI_eddy_MD.nii.gz",
        "ADC": "_ADC.nii.gz",
        "ASL": "_ASL.nii.gz"
    } 
    
    if img_type in image_types:
        print(f"Processing {img_type} image for patient {patient_id}")
        image_filename = f"UCSF-PDGM-{patient_id:04d}{image_types[img_type]}"
        image_path = os.path.join(data_dir, image_filename)
    else:
        print(f"Unknown image type: {img_type}")
        image_path = None
    
    label_filename = f"UCSF-PDGM-{patient_id:04d}_tumor_segmentation.nii.gz"
    label_path = os.path.join(data_dir, label_filename)
    
    return image_path, label_path

def view_data(Image, Label):
    plt.figure(figsize=(20,20))
    plt.subplot(2,2,1)
    plt.imshow(sitk.GetArrayFromImage(Image)[109,:,:], cmap="gray")
    plt.title("Brain #1")
    plt.subplot(2,2,2)
    plt.imshow(sitk.GetArrayFromImage(Label)[109,:,:])        
    plt.title("Segmentation #1")
    plt.show()
    
def main():
    logger = setup_logging()
    base_dir, params, train_df, environment = get_settings()
    aquisitions = ["T2"] # Adapt according to needs
    
    for cImg in aquisitions:
        img_type = cImg # "T1_contrast" # choose from: DTI_eddy_FA, FLAIR, SWI, T1, T2, DWI, T1_contrast, DTI_eddy_MD
        feature_data = pd.DataFrame() # Initialize feature data df
        errors_id = pd.DataFrame()
        count = 0

        for _, row in train_df.iterrows():
            patient_id_str = row[0]  # Assuming the patient ID is in the first column
            patient_id = int(patient_id_str.split("-")[-1])
            image_path, label_path = fetch_data(base_dir, patient_id, img_type)

            try:
                Image, Label = read_image_and_label(image_path, label_path, patient_id)
                if environment == 'local': view_data(Image, Label)
                extractor, result = extract_features(params, Image, Label, patient_id) # Here it goes wrong if label < 3
                extractor = apply_filters_on_images(extractor)
                featureVector = calculate_features(extractor, Image, Label)
                first_order_features = {key: value for key, value in featureVector.items() if key.startswith('original_firstorder_')}
                feature_data = feature_data.append(pd.DataFrame(first_order_features, index=[patient_id]))
            except Exception as e:
                count += 1
                tb_str = traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__)
                tb_str = ''.join(tb_str)
                print(f"Could not find data for patient {patient_id}: {str(e)}\nTraceback:\n{tb_str}")

        print(f'Number of errors occurred: {count}')
         # Save the feature_data DataFrame as a CSV file in the base_dir
        feature_data.to_csv(os.path.join(base_dir, f"extracted_firstorder_features_{img_type}.csv"))        
            
if __name__ == "__main__":
    main()


# In[ ]:




