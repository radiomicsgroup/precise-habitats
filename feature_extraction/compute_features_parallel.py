### Extract 3D radiomics features from original images

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
# argv[0]: /nfs/rnas/PRECISION/COHORT/data/images/PATIENT1_TIMEPOINT1 #folder storing original image and subfolder "labels" which stores exported segmentations (label maps)
# argv[1]: /nfs/rnas/PRECISION/COHORT/analysis/radiomics/original_R1B25 #output folder to store extracted features
# argv[2]: /nfs/rnas/REPRO/COHORT/pipelines/feature_extraction/ROI_R1B25.yaml #parameter file

## EXAMPLE:
#python compute_features_parallel.py /nfs/rnas/PRECISION/COHORT/data/images/PATIENT1_TIMEPOINT1 /nfs/rnas/PRECISION/COHORT/analysis/radiomics/original_R1B25 /nfs/rnas/REPRO/COHORT/pipelines/feature_extraction/ROI_R1B25.yaml

def readimages(ipath): #function reads original image
    import pandas as pd
    df = pd.DataFrame(columns= ['Patient', 'image', 'mask']) 
    for root, dirs, files in os.walk(ipath):
        if not 'Dir' in root and not 'Store' in root:
            if not os.path.basename(root)=='labels':
                root2=root+'/labels'
                for file in os.listdir(root2):
                    if file.endswith('label.nrrd'):
                            image_str = file.split('-')[0]
                            image_file = root + '/' + image_str + '.nii'
                            patient = image_str.split('_')[0]
                            timepoint = image_str.split('_')[1]
                            patient_timepoint=patient+'_'+timepoint
                            mask_file = root2 + '//' + file
                            df = df.append({'Patient':patient_timepoint,'image': image_file,'mask': mask_file},ignore_index=True)
                return df
                #df.to_csv('Cases_parallel.csv')
#readimages(ipath) #in case you want to try it out first

def setupLogger(LOG):
    rLogger = radiomics.logger
    logHandler = logging.FileHandler(filename=LOG, mode='w')
    logHandler.setLevel(logging.INFO)
    logHandler.setFormatter(logging.Formatter('%(levelname)-.1s: (%(threadName)s) %(name)s: %(message)s'))
    rLogger.addHandler(logHandler)
    return rLogger

def main(argv):
    ROOT = argv[0]
    PARAMS = os.path.join(ROOT, argv[2])  # Parameter file
    LOG = os.path.join(ROOT, 'log.txt')  # Location of output log file
    logger = setupLogger(LOG)
    logger.info('pyradiomics version: %s', radiomics.__version__)
    logger.info('Reading images...')
    odir = os.path.join(argv[1])
    if not os.path.exists(odir): os.mkdir(odir)
    
    df = readimages(os.path.join(ROOT))    
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
