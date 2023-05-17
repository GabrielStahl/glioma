from __future__ import print_function
import sys
import os
import six
from radiomics import featureextractor, getFeatureClasses
import radiomics
import SimpleITK as sitk
import pandas as pd
import traceback
from scipy.stats import pearsonr
import numpy as np

def get_settings():
    environment = os.environ.get('PYRADIOMICS_ENV', 'local')  # Default to 'local' if the environment variable is not set

    dataset = "train" # Choose from: ["train", "validation", "test"] -- this will determine which dataset to use
    Filter = "SquareRoot" # Choose from: ["Original", "SquareRoot"]
    
    # choose modalities from: ["T1","T2","DTI_eddy_FA", "FLAIR", "SWI", "DWI", "T1_contrast", "DTI_eddy_MD", "ADC", "ASL"]
    acquisitions = ["T1","T2","DTI_eddy_FA", "FLAIR", "SWI", "DWI", "T1_contrast", "DTI_eddy_MD", "ADC", "ASL"] 

    if environment == 'local':
        base_dir = "/Users/Gabriel/Desktop/MSc_Dissertation/pyRadiomics/Data/"
        params = "/Users/Gabriel/Desktop/MSc_Dissertation/pyRadiomics/Params.yaml"
        train_df = pd.read_csv("/Users/Gabriel/Desktop/MSc_Dissertation/pyRadiomics/Training/" + dataset + "_data.csv") 
    elif environment == 'cluster':
        base_dir = "/cluster/project2/UCSF_PDGM_dataset/UCSF-PDGM-v3/"
        params = "/home/goliveir/pyRadiomics/Params.yaml"
        train_df = pd.read_csv("/home/goliveir/pyRadiomics/" + dataset + "_data.csv") 
    else:
        raise ValueError("Unknown environment: please set PYRADIOMICS_ENV to 'local' or 'cluster'")

    return base_dir, params, train_df, environment, dataset, Filter, acquisitions

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
    

def apply_filters_on_images(extractor, Filter):

    # By default, only 'Original' (no filter applied) is enabled. Optionally enable some image types:
    if Filter == "SquareRoot":
        extractor.enableImageTypeByName('SquareRoot')

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
    
def main():
    base_dir, params, train_df, environment, dataset, Filter, acquisitions = get_settings()
    lost_patients = []
    
    for cImg in acquisitions:
        img_type = cImg 
        feature_data = pd.DataFrame() # Initialize feature data df
        errors_id = pd.DataFrame()
        count = 0

        for _, row in train_df.iterrows():
            patient_id_str = row[0]  # Assuming the patient ID is in the first column
            patient_id = int(patient_id_str.split("-")[-1])
            image_path, label_path = fetch_data(base_dir, patient_id, img_type)

            try:
                Image, Label = read_image_and_label(image_path, label_path, patient_id)
                extractor, result = extract_features(params, Image, Label, patient_id) # Here it goes wrong if label < 3
                extractor = apply_filters_on_images(extractor, Filter)
                featureVector = calculate_features(extractor, Image, Label)
                
                if Filter == "Original":
                    first_order_features = {key: value for key, value in featureVector.items() if key.startswith('original_firstorder_')}
                
                if Filter == "SquareRoot":
                    first_order_features = {key: value for key, value in featureVector.items() if key.startswith('squareroot_firstorder_')}

                feature_data = feature_data.append(pd.DataFrame(first_order_features, index=[patient_id]))
            except Exception as e:
                count += 1
                tb_str = traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__)
                tb_str = ''.join(tb_str)
                print(f"Error for patient {patient_id}: {str(e)}\nTraceback:\n{tb_str}")
                lost_patients.append(patient_id)

        print(f'Number of errors occurred: {count} --> patients affected by error {set(lost_patients)}')
         # Save the feature_data DataFrame as a CSV file in the base_dir
        feature_data.to_csv(os.path.join(base_dir, f"{dataset}_firstorder_{Filter}_{img_type}.csv"))        
            
if __name__ == "__main__":
    main()






