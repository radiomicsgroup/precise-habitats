### Calculate test-retest repeatibility metrics for all 3D radiomics features extracted from one lesion

# Import modules
import os,nrrd,argparse
import numpy as np
import SimpleITK as sitk
import scipy
import pandas as pd
import sys
import scipy.stats as st
from scipy.stats import f as fstat
import sklearn
from sklearn import preprocessing


### INPUT:
# -testROI /nfs/rnas/PRECISION/COHORT/analysis/radiomics/original_R1B12/PATIENT1_TIMEPOINT1-LIVER1   #Folder containing 3D features extracted from original image
# -retestROI /nfs/rnas/PRECISION/COHORT/analysis/radiomics/perturbed_R1B12/PATIENT1_TIMEPOINT1-LIVER1    #Folder containing 3D features extracted from perturbed image
# -exclude glcm_MCC # Feature to be excluded
# -odir /nfs/rnas/PRECISION/COHORT/pipelines/analysis/repeat_csv #Output directory to store csv with icc data

### EXAMPLE:
# python metrics_repeat.py --testROI /nfs/rnas/PRECISION/COHORT/analysis/radiomics/original_R1B12/PATIENT1_TIMEPOINT1-LIVER1 --retestROI /nfs/rnas/PRECISION/COHORT/analysis/radiomics/perturbed_R1B12/PATIENT1_TIMEPOINT1-LIVER1 --exclude glcm_MCC --odir /nfs/rnas/PRECISION/COHORT/pipelines/analysis/repeat_csv

# Intraclass Correlation Coefficient
def ICC3A1(data): #data must be n rows (voxels) and k columns (conditions). Example: (1484,2)
    #Degrees of freedom
    [n, k] = data.shape
    dfc = k - 1 #degrees of freedom for columns ("bertween judges")
    dfr = n - 1 #deegres of freedom for rows ("between targets")
    dfe = dfr * dfc #degrees of freedom for error

    #Grand mean of all voxels (for both original and perturbed):
    grand_mean=np.mean(data) #when no aixs is given, np.mean computes mean of flattened array (all values)

    #Sum Square Row (SSR):
    sq_diff_SSR=[]
    for i in range(0,n):
        row=data[i]
        mean_row=np.mean(row)
        sq_diff=(mean_row-grand_mean)**2
        sq_diff_SSR.append(sq_diff)
    SSR=np.sum(k*sq_diff_SSR)

    #Sum Square Column (SSC):
    sq_diff_SSC=[]
    for i in range(0,k):
        column=data[:,i]
        mean_column=np.mean(column)
        sq_diff=(mean_column-grand_mean)**2
        sq_diff_SSC.append(sq_diff)
    SSC=np.sum(n*sq_diff_SSC)

    #Sum Square Total (SST)
    demeaned_data = data - grand_mean
    SST = np.sum(demeaned_data**2)

    #Sum Square Error (SSE):
    SSE=SST-SSC-SSR

    #Mean Squared 
    MSR=SSR/dfr
    MSE=SSE/dfe
    MSC=SSC/dfc/n
    
    #ICC(3A,1), F statistics and CI
    ICC=(MSR-MSE)/(MSR +  dfc*MSE + (k/n)*(MSC-MSE))
    FC = MSC / MSE #columns effect #do not use it. 
    FR =  MSR / MSE #rows effect
    F=fstat.ppf(q=1-.05/2,dfn=n-1,dfd=(n-1)*(k-1),loc=0,scale=1)
    FU=FR*F #Upper limit
    FL=FR/F #Lower limit
    LCL=(FL-1)/(FL+(k-1))
    UCL=(FU-1)/(FU+(k-1))
    
    return np.round(ICC,3), np.round(LCL,3), np.round(UCL,3)

# Repeatability Coefficient
def RC(data):
    var_rows=[] #stores within-subject sample variances
    for i in range(0,data.shape[0]): #for every row
        var_row=np.var(data[i],ddof=1) #variance of every row
        var_rows.append(var_row)
    var_tot=np.mean(var_rows) #Mean within-subject sample variance
    std_tot=np.sqrt(var_tot) #Within-subject standard deviation 
    rc=(1.96*np.sqrt(2)*std_tot)*100 #Repeatability Coefficient

    # Note: we could also compute it like this, see suppl.5:
    # diff_row=[] #stores squared measurement differences
    # for i in range(0,data.shape[0]):
    #     row=data[i]
    #     diff=row[1]-row[0] #computes measurement differences per subject
    #     diff_row.append(diff**2) #squared difference is stored
    # diff_sum=np.sum(diff_row) #total squared differences
    # mean_diff_sum=diff_sum/data.shape[0] #mean squared differences
    # rc=(1.96*np.sqrt(mean_diff_sum))*100 #Repeatability Coefficient
    return np.round(rc,3)

# Coefficient of Variation
def CV(data):
    grand_mean=np.mean(data) #when no aixs is given, np.mean computes mean of flattened array (all values)
    var_rows=[] #list to store original-perturbed values 
    for i in range(0,data.shape[0]): #for every row
        var_row=np.var(data[i]) #variance of every row
        var_rows.append(var_row)
    var_tot=np.mean(var_rows)
    std_tot=np.sqrt(var_tot)
    cv=(std_tot/grand_mean)*100
    return np.round(cv,3)


if __name__ == '__main__':
    # Print help and parse arguments
    parser = argparse.ArgumentParser(description='Get repeatability metrics for all 3D radiomic features stored in NRRD files for one ROI (lesion)')
    parser.add_argument('--testROI', help='Path to folder containing all features from original (test) lesion', required=True)
    parser.add_argument('--retestROI', help='Path to folder containing all features from perturbed (retest) lesion')
    parser.add_argument('--exclude', help='Name of feature to exclude', required=False)
    parser.add_argument('--odir', help='Path to store csv file of every ROI', required=True)
    args = parser.parse_args()

    # Get file names of input and output
    idir_original=args.testROI
    idir_perturbed=args.retestROI
    exclude=args.exclude
    odir=args.odir

    #Get lesion name and setting
    lesion_name=os.path.basename(idir_original).split('-label')[0] #Get lesion name from directory
    setting_name=os.path.dirname(idir_original).split('original_')[1]

    # Loop through subjects and measurements: load data
    print("")
    print("****************************************************************************")
    print("              Repeatability Metrics Computation                ")
    print("****************************************************************************")
    print("")
    print("  Data folder for test ROI: {}".format(idir_original))
    print("  Data folder for retest ROI: {}".format(idir_perturbed))
    print("  Excluded features: {}".format(exclude))
    print("  Lesion: {}".format(lesion_name))
    print("  Setting: {}".format(setting_name))
    print("  Output csv will be stored in: {}".format(odir))
    print("")
    print("")
    print("      Loading data for ICC calculation. Please wait...")


    # We create the list of included feature names 
    feat_names=[]
    feats=os.listdir(idir_original)
    for f in feats:
        if 'original' in f and not exclude in f: #excluding features specified in arguments
            feat_name=f.split('original_')[1].split('.nrrd')[0]
            feat_names.append(feat_name)


    # Load first feature to get ROI size, define other variables
    Nfeat=len(feat_names) #total number of features per ROI
    k=2 #two conditions (original-perturbed). In the repro/repeat literature also called "sessions" or "raters" or "measurements"

    ref_feat = idir_original+'/original_firstorder_10Percentile.nrrd' #template feature
    ref_nrrd = sitk.ReadImage(ref_feat) #read 3D feature image
    feat_arr = sitk.GetArrayFromImage(ref_nrrd) #take 3D feature as array
    f = feat_arr[~np.isnan(feat_arr)] #remove NaN values
    imgsize=f.shape #obtain ROI size (number of voxels)
    fullmatsize= tuple([Nfeat]) + tuple([k]) + imgsize #array of 3: Nfeat (layers), k conditions (rows), n voxels (columns)
    fulldata= np.zeros(fullmatsize) #Example: (92, 2, 1484) #92 layers (features). For every layer (feature) I have two rows (two measurements) with 1484 columns (voxels).

    n=fulldata.shape[2] #total number of voxels
    min_max_scaler = preprocessing.MinMaxScaler() #to scale features

    # Loop to obtain array with all info: voxel values for the two measurements for every feature
    i=1
    for mm in range(1, k+1): #for every condition
        for feat_name in feat_names: #for every feature
                if mm==1:
                    feat_path=idir_original+'/original_'+feat_name+'.nrrd' #if mm=1; then first measurement --> original folder
                else:
                    feat_path=idir_perturbed+'/original_'+feat_name+'.nrrd'#if mm=2; then second measurement --> perturbed folder
                feat_nrrd = sitk.ReadImage(feat_path) #reading 3D feature (image)
                feat_arr = sitk.GetArrayFromImage(feat_nrrd) #converting image feature to array
                f = feat_arr[~np.isnan(feat_arr)] #removing NaNs. Also converted to 1D.
                f_scaled=min_max_scaler.fit_transform(f.reshape(-1,1))
                feat_arr_scaled=f_scaled.reshape(np.shape(f))
                if feat_arr_scaled.shape==imgsize:
                    if i<(Nfeat+1): #from i=1 to i=92, features from 1st measurement.
                        fulldata[i-1,mm-1,:] = feat_arr_scaled   # Store values in the global data set, index of features ranges from 0 to Nfeat-1 (91 in our case)
                    else: #from i=93 to i=184, features from 2nd measurement
                        fulldata[i-(Nfeat+1),mm-1,:] = feat_arr_scaled   # Store values in the global data set,
                else: #sometimes one feature misses 1 or 2 voxels, in that case, append NaN to maintain ROI size
                    missing_voxels=abs(imgsize[0]-feat_arr_scaled.shape[0]) #number of missing voxels
                    feat_arr_scaled=np.append(feat_arr_scaled,np.repeat(0,missing_voxels)) #append 0s
                    print('ALERT: ROI size mismatch for feature {} by {} voxels.'.format(feat_name, missing_voxels))	  			 
                    if i<(Nfeat+1): #from i=1 to i=92, features from 1st measurement.
                        fulldata[i-1,mm-1,:] = feat_arr_scaled   # Store values in the global data set, index of features ranges from 0 to Nfeat-1 (91 in our case)
                    else: #from i=93 to i=184, features from 2nd measurement
                        fulldata[i-(Nfeat+1),mm-1,:] = feat_arr_scaled   # Store values in the global data set,
                i+=1 #advance one step
    print("      Computing Repatability Metrics. Please wait...")

    # Empty lists to store info from variables for all features
    icc_tot=[] #icc
    icc_LCL_tot=[] #lower CI for icc
    icc_UCL_tot=[] #upper CI for icc
    cv_tot=[] #CV
    rc_tot=[] #RC
    print("      Creating dataframe. Please wait...")

    # Lopp to obtain metrics for all features
    for i in range(0,fulldata.shape[0]): #for every feature:
        data_for_anova=np.transpose(fulldata[i,:,:]) #Now 1484 voxels (rows) and 2  conditions (columns)
    
        #Compute metrics for each feature
        ICC, ICC_LCL, ICC_UCL=ICC3A1(data_for_anova)
        cv=CV(data_for_anova)
        rc=RC(data_for_anova)
        
        #Append metrics values to global list
        icc_tot.append(ICC)
        icc_LCL_tot.append(ICC_LCL)
        icc_UCL_tot.append(ICC_UCL)
        cv_tot.append(cv)
        rc_tot.append(rc)

    #Create list of lesion name, which is the same for all features for the same ROI
    lesion=[lesion_name]*Nfeat #repeat as many times as lesion

    #Create list of parameters, which is the same for all features for the same ROI
    setting=[setting_name]*Nfeat

    #Create list of feature class
    featclass=[]
    for feature in feat_names:
        class_name=feature.split('_')[0]
        featclass.append(class_name)

    #Create dataframe for my ROI
    pre_df = {'lesion': lesion, 'setting': setting, 'feature':feat_names,'feature_class':featclass,
            'ICC':icc_tot, 'ICC_LCL': icc_LCL_tot, 'ICC_UCL': icc_UCL_tot,
            'CV': cv_tot, 'RC': rc_tot, 'primary_tumor': 'colorectal', 'lesion_locaion': 'liver', 'cohort':'COHORT'}

    df = pd.DataFrame(pre_df)
    df.to_csv(odir+'/'+lesion_name+'_'+setting_name+'.csv')

#Enjoy your csv!
print('****************************** PHinisheD!! Enjoy your dataframe!! ***************************** ')
# sys.exit(0)
