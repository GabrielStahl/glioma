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

# Configure Settings
dataset = "training" # Choose from: ["training", "validation", "test"] -- this will determine which dataset to use
# Filter = "Square" # Choose from: ["Original", "SquareRoot", "Wavelet", "Square", "Exponential", "Logarithm"]
Filter = sys.argv[1] # Pass the filter as an argument in the bash file
Store_Original = False # Choose from: [True, False] 
# choose modalities from: ["T1","T2","DTI_eddy_FA", "FLAIR", "SWI", "DWI", "T1_contrast", "DTI_eddy_MD", "ADC", "ASL"]
acquisitions = ["T1","T2","DTI_eddy_FA", "FLAIR", "SWI", "DWI", "T1_contrast", "DTI_eddy_MD", "ADC", "ASL"] 

def get_settings():
    environment = os.environ.get('PYRADIOMICS_ENV', 'local')  # Default to 'local' if the environment variable is not set

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

    return base_dir, params, train_df, environment

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
        print(f"Could not extract features for patient {patient_id} Reason: {str(e)} \nretrying with label-value=2")
        settings = {'label': 2}
        extractor = radiomics.featureextractor.RadiomicsFeatureExtractor(additionalInfo=True, **settings)
        result = extractor.execute(Image, Label)  
        return extractor, result
    

def apply_filters_on_images(extractor):

    # By default, only 'Original' (no filter applied) is enabled. Optionally enable some image types:

    filter_types = {
        "SquareRoot": 'SquareRoot',
        "Wavelet": 'Wavelet',
        "Square": 'Square',
        "Exponential": 'Exponential',
        "Logarithm": 'Logarithm'
    }

    for filter_type, filter_name in filter_types.items():
        if Filter == filter_type:
            extractor.enableImageTypeByName(filter_name)
    
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
    base_dir, params, train_df, environment = get_settings()
    lost_patients = []
    
    for cImg in acquisitions:
        img_type = cImg 
        
        feature_data = pd.DataFrame() # Initialize feature data df
        feature_data_sqrroot = pd.DataFrame() # Initialize feature data df
        feature_data_wavelet = pd.DataFrame() # Initialize feature data df
        feature_data_square = pd.DataFrame() # Initialize feature data df
        feature_data_exponential = pd.DataFrame() # Initialize feature data df
        feature_data_logarithm = pd.DataFrame() # Initialize feature data df

        errors_id = pd.DataFrame()
        count = 0

        for _, row in train_df.iterrows():
            patient_id_str = row[0]  # Assuming the patient ID is in the first column
            patient_id = int(patient_id_str.split("-")[-1])
            image_path, label_path = fetch_data(base_dir, patient_id, img_type)

            try:
                Image, Label = read_image_and_label(image_path, label_path, patient_id)
                extractor, result = extract_features(params, Image, Label, patient_id) # Here it goes wrong if label < 3
                extractor = apply_filters_on_images(extractor)
                featureVector = calculate_features(extractor, Image, Label)
                
                # Always store the original features in the feature_data DataFrame
                original_features = {key: value for key, value in featureVector.items() if key.startswith('original_firstorder_')}
                feature_data = feature_data.append(pd.DataFrame(original_features, index=[patient_id]))

                # Additionally, store the filtered features in separate DataFrames
                if Filter == "SquareRoot":
                    sqrroot_features = {key: value for key, value in featureVector.items() if key.startswith('squareroot_firstorder_')}
                    feature_data_sqrroot = feature_data_sqrroot.append(pd.DataFrame(sqrroot_features, index=[patient_id]))

                if Filter == "Wavelet":
                    wavelet_features = {key: value for key, value in featureVector.items() if key.startswith('wavelet-LLH_firstorder_')}
                    feature_data_wavelet = feature_data_wavelet.append(pd.DataFrame(wavelet_features, index=[patient_id]))

                if Filter == "Square":
                    square_features = {key: value for key, value in featureVector.items() if key.startswith('square_firstorder_')}
                    feature_data_square = feature_data_square.append(pd.DataFrame(square_features, index=[patient_id]))

                if Filter == "Exponential":
                    exponential_features = {key: value for key, value in featureVector.items() if key.startswith('exponential_firstorder_')}
                    feature_data_exponential = feature_data_exponential.append(pd.DataFrame(exponential_features, index=[patient_id]))

                if Filter == "Logarithm":
                    logarithm_features = {key: value for key, value in featureVector.items() if key.startswith('logarithm_firstorder_')}
                    feature_data_logarithm = feature_data_logarithm.append(pd.DataFrame(logarithm_features, index=[patient_id]))
    
            except Exception as e:
                count += 1
                tb_str = traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__)
                tb_str = ''.join(tb_str)
                print(f"Error for patient {patient_id}: {str(e)}\nTraceback:\n{tb_str}")
                lost_patients.append(patient_id)

        print(f'Number of errors occurred: {count} --> patients affected by error {set(lost_patients)}')
        
        # Save the feature_data DataFrame as a CSV file in the base_dir
        if Store_Original == True:
            feature_data.to_csv(os.path.join(base_dir, f"{dataset}_firstorder_original_{img_type}.csv"))    

        if Filter == "SquareRoot":
            feature_data_sqrroot.to_csv(os.path.join(base_dir, f"{dataset}_firstorder_sqrroot_{img_type}.csv"))

        if Filter == "Wavelet":
            feature_data_wavelet.to_csv(os.path.join(base_dir, f"{dataset}_firstorder_wavelet_{img_type}.csv"))
        
        if Filter == "Square":
            feature_data_square.to_csv(os.path.join(base_dir, f"{dataset}_firstorder_square_{img_type}.csv"))
        
        if Filter == "Exponential":
            feature_data_exponential.to_csv(os.path.join(base_dir, f"{dataset}_firstorder_exponential_{img_type}.csv"))
        
        if Filter == "Logarithm":
            feature_data_logarithm.to_csv(os.path.join(base_dir, f"{dataset}_firstorder_logarithm_{img_type}.csv"))
            
        
if __name__ == "__main__":
    main()