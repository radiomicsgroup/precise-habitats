### Extract 3D radiomics features from perturbed images

# Import modules
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor
import os
import sys, getopt
from os.path import join
import radiomics
from radiomics import setVerbosity
import logging
from collections import OrderedDict
import six
from datetime import datetime


## INPUT
# argv[0]: /nfs/rnas/PRECISION/COHORT/data/images/PATIENT1_TIMEPOINT1 #folding storing original image and subfolder "labels" which stores exported segmentations (label maps)
# argv[1]: /nfs/rnas/PRECISION/COHORT/data/images_perturbed #folder storing all perturbed images
# argv[2]: /nfs/rnas/PRECISION/COHORT/analysis/radiomics/perturbed_R1B25 #output folder to store extracted features
# argv[3]: /nfs/rnas/REPRO/COHORT/pipelines/feature_extraction/ROI_R1B25.yaml #parameter file

## EXAMPLE:
#python compute_features_parallel_perturbed.py /nfs/rnas/PRECISION/COHORT/data/images/PATIENT1_TIMEPOINT1 /nfs/rnas/PRECISION/COHORT/data/images_perturbed  /nfs/rnas/PRECISION/COHORT/analysis/radiomics/perturbed_R1B25 /nfs/rnas/REPRO/COHORT/pipelines/feature_extraction/ROI_R1B25.yaml

def readimages(ipath): #function reads original image
    import pandas as pd
    File1 = pd.DataFrame(columns= ['Patient', 'mask'])  #dataframe with only patient and labels
    for root, dirs, files in os.walk(ipath): 
     if not 'Dir' in root and not 'Store' in root:
         if not os.path.basename(root)=='labels':
            root2=root+'/labels'
            for file in os.listdir(root2):
               if file.endswith('label.nrrd'):
                    image_str = file.split('-')[0]
                    image_file = root + '/' + image_str + '.nrrd'
                    patient = image_str.split('_')[0]
                    timepoint = image_str.split('_')[1]
                    patient_timepoint=patient+'_'+timepoint
                    mask_file = root2 + '//' + file
                    File1 = File1.append({'Patient':patient_timepoint,'mask': mask_file},ignore_index=True)
            return File1
            #File1.to_csv('test_File1.csv')


def readimages_pert(ipath_pert): #function reads perturbed image
    import pandas as pd
    file2=[]
    for root, dirs, files in os.walk(ipath_pert): 
        if not root.endswith('Store') and not root.endswith('@eaDir'):
            for file in files:
                if not file.endswith('.DS_Store') and not file.endswith('@eaDir') and not file.endswith('source'): #had to add this because I'm working with Macbook (@Olivia)
                    image_file= root + '/' + file
                    file2.append(image_file)
                    File2=pd.DataFrame(file2)
                    File2.columns=["image"]
                    Image=File2["image"]
                    File2["Patient"]= [os.path.basename(str(file)).split('_CT')[0] for file in File2['image']] #aquí me lo junta por paciente así que en mi txt file no cal poner por paciente de perturbed
            return File2
            #File2.to_csv('test_File2.csv')

def setupLogger(LOG):
    rLogger = radiomics.logger
    logHandler = logging.FileHandler(filename=LOG, mode='w')
    logHandler.setLevel(logging.INFO)
    logHandler.setFormatter(logging.Formatter('%(levelname)-.1s: (%(threadName)s) %(name)s: %(message)s'))
    rLogger.addHandler(logHandler)
    return rLogger

def main(argv):
    ROOT = argv[0]
    ROOT_pert=argv[1]

    PARAMS = os.path.join(ROOT, argv[3])  # Parameter file
    LOG = os.path.join(ROOT, 'log.txt')  # Location of output log file
    logger = setupLogger(LOG)
    logger.info('pyradiomics version: %s', radiomics.__version__)
    logger.info('Reading images...')
    odir = os.path.join(argv[2])
    if not os.path.exists(odir): os.mkdir(odir)
    # Merging to obtain final df with correct prturbed image and mask
    df1=readimages(ROOT)
    df2=readimages_pert(ROOT_pert)
    df=df1.merge(df2,on='Patient')
    df=df[["Patient", "image", "mask"]]


    cases = [OrderedDict(row) for i, row in df.iterrows()]
    for case in cases:
        feature_vector = OrderedDict(case)
        logger.info('Case is %s', case)
        try:
            t = datetime.now()
            imageFilepath = case['image']  # Required
            maskFilepath = case['mask']  # Required
            mask = sitk.ReadImage(maskFilepath)
            lsif = sitk.LabelShapeStatisticsImageFilter()
            lsif.Execute(mask)
            labels = lsif.GetLabels()
            label = labels[0]
            pat_folder = maskFilepath.split('//')[1].split('.nrrd')[0]
            if not os.path.exists(odir +'/' + pat_folder): os.mkdir(odir +'/' + pat_folder)
            extractor = featureextractor.RadiomicsFeatureExtractor(PARAMS)
            logger.info('Reading image %s and mask %s with label %s', imageFilepath, maskFilepath, label)
            logger.info('Saving results in %s', odir)
            feature_vector.update(extractor.execute(imageFilepath, maskFilepath, voxelBased=True, label = label))
            
            for key, val in six.iteritems(feature_vector):
                if isinstance(val, sitk.Image):
                    tfm_file = odir +'/' + pat_folder +'/'+key+'.nrrd'  
                    sitk.WriteImage(val, tfm_file)
                    logger.info('Features %s from Image %s is saved', key,pat_folder)

            delta_t = datetime.now() - t
            logger.info('Patient %s processed in %s', case['Patient'],  delta_t)

        except:
            logger.error('Feature extraction failed!', exc_info=True)
            logger.error(sys.exc_info())

if __name__ == "__main__":
   main(sys.argv[1:])