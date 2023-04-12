### Compute Habitats wtih CT data (3D radiomics features). Specifically, for the same lesion, compute habitats:
# (i) with precise original 3D radiomics features
# (ii) with all original 3D radiomics features
# (iii) with precise perturbed 3D radiomics features
# (iv) with all perturbed 3D radiomics features

## INPUT
# argv[0]: PATIENT1_TIMEPOINT1_LESION #name of lesion 
# argv[1]: /nfs/rnas/PRECISION/COHORT/analysis/radiomics #folder containing 3D features to compute habitat
# argv[2]: /nfs/rnas/PRECISION/pretty_figures #output folder to store boxplot

## EXAMPLE:
#python habitat_computation.py PATIENT1_TIMEPOINT1_LESION /nfs/rnas/PRECISION/COHORT/analysis/radiomics /nfs/rnas/PRECISION/pretty_figures

# install packages
import time, os, argparse, csv, sys, nrrd
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import matplotlib.cm as cm
import SimpleITK as sitk
import nibabel as nib
from sklearn.mixture import GaussianMixture as GMM
from sklearn.cluster import KMeans
import seaborn as sns
from gmm_mml import GmmMml
from sklearn.mixture import BayesianGaussianMixture as BGM
from sklearn.metrics.cluster import contingency_matrix
import munkres
from munkres import Munkres
from sklearn.decomposition import PCA
from scipy import stats
from scipy.stats import spearmanr
from matplotlib.patches import Rectangle

start_t=time.time()
def dsc(original, perturbed, h):
    """
    Compute the Dice similarity coefficient (DSC) between two input arrays for a specific label (habitat).
    
    -----------
    Input
    -----------
    - original: numpy array representing the original data
    - perturbed: numpy array representing the perturbed data
    - h: specific label (habitat) to compute the DSC for

    -----------
    Output
    -----------
    - dice: the Dice similarity coefficient rounded to 3 decimal places
    """
    dice = np.sum(perturbed[original==h]==h)*2.0 / (np.sum(perturbed==h) + np.sum(original==h)) #compute dice score between original and perturbed habitats
    return np.round_(dice,3) #round to three decimals

def optimal_k(data):
    """
    Determine the optimal number of clusters (k) for a dataset using Gaussian Mixture Model (GMM) and the Bayesian Information Criterion (BIC).
    -----------
    Input
    -----------
    - data: a dataset to cluster

    -----------
    Output
    -----------
    - k: the optimal number of clusters (k) based on the maximum change in the gradient of BIC scores
    """

    bic_scores = []
    k_list = range(1,6) #setting max habitats to 5. 1 is not included.
    for knum in k_list:
        gmm = GMM(n_components=knum, random_state=123)
        gmm = gmm.fit(data)
        bic_scores.append(gmm.bic(data))
    x = k_list
    y = np.gradient(bic_scores)
    m = np.diff(y)/np.diff(x) #slope (change between every 2 points)
    m_changes=abs(np.diff(m)) #slope changes in absolute value
    i_max=np.where(m_changes==m_changes.max())[0][0]+1 #gives the the position of maxmimum change in gradient
    k=k_list[i_max]
    return k

def get_featim(idir,les):
    """This function takes ad irectory and lesion name and returns a feature image"""
    ori_dir=idir+'/original_R3B12' #3D radiomics ORIGINAL
    pert_dir=idir+'/perturbed_R3B12' #3D radiomics PERTURBED
    im_path=ori_dir+'/'+les+'/original_firstorder_90Percentile.nrrd'
    im_path_pert=pert_dir+'/'+les+'/original_firstorder_90Percentile.nrrd'
    feat_im = sitk.ReadImage(im_path)
    feat_im_pert = sitk.ReadImage(im_path_pert)
    return feat_im, feat_im_pert

def readCT(idir,les):
    """
    This function takes a directory as input (which contains 3D CT-radiomics features) as well as the name of the lesion (les) 
    and returns an array "Allfeatures" which contains the values of the voxels for all features extracted from both the original 
    lesion and the perturbed one, a list of tuples "shape_len" which contains the shape of each lesion, a list of strings 
    "features_names" (list of feature names) and a list of strings "lesions" which contains the name of the lesion (original or perturbed).

    -----------
    Input
    -----------
    - idir: directory containing 3D CT-radiomics features
    - les: lesion name

    -----------
    Output
    -----------
    - Allfeatures: numpy array containing values of the voxels for all features extracted from both the original and perturbed lesions
    - shape_len: list of tuples containing the shape of each lesion
    - features_names: list of strings containing feature names
    - lesions: list of strings containing the name of the lesion (original or perturbed)

    This function takes a directory as input (which contains 3D CT-radiomics features) as well as the name of the lesion (les) and returns an array
    "Allfeatures" which contains the values of the voxels for all features extracted from both the original lesion
    and the perturbed one, a list of tuples "shape_len" which contains the shape of each lesion, a list of strings "features_names" (list of feature names)
    and a list of strings "lesions" which contains the name of the lesion (original or perturbed)"""

    Allfeatures=np.empty((0,91)) #91 columns because we have 91 3D features extracted
    shape_len=[]
    lesions=[]
    features=[] #list that will have all voxels
    features_names=[] #lsit that will have feature names
    if "ori" in idir: #we want to differentiate between original and perturbed lesion because we will need to separate them later when we compute habitats
        lesions.append('original_'+les)
    else:
        lesions.append('perturbed_'+les)
    for file in os.listdir(os.path.join(idir,les)): #for every feature inside the subfolder radiomics/original_R3B12/lesion
        if not "glcm_MCC" in file and not "firstorder_TotalEnergy" in file: #excluded features
            #reading the feature image and obtaining 1D array
            feature=nrrd.read(os.path.join(idir,les,file)) #extraction of 3D feature map
            feat1D=feature[0].flatten() #flattening the feature (we will have a list of the feature values for every voxel). Shape: (52920,), size: 52920 voxels
            feat2D=feat1D.reshape(-1,1) #we reshape to an array of 2D, needed for scaling, Shape: (52920,1), size: same as before
            scaler = MinMaxScaler().fit(feat2D)  #Compute the minimum and maximum to be used for later scaling
            feat2D_scaled=scaler.transform(feat2D) #Scale features (each 2D array) according to feature_range (scaler)
            feat1D_scaled=feat2D_scaled.flatten()
            features.append(feat1D_scaled) #appending the flattened feature to my major list of features 
            feature_name = file.split('original_')[1].split('.nrrd')[0] #feature name
            features_names.append(feature_name)
    shape_len.append((feature[0].shape, len(feature[0].flatten()))) #shape and length of 3D feature map, that is, the size of the lesion. First feature is enough because they all have the same size.
    Allfp=np.transpose(np.array(features),axes=(1,0)) #all of my features for one lesion
    Allfeatures=np.concatenate((Allfeatures, Allfp)) #all of my features for original lesion + perturbed lesion
    return Allfeatures, shape_len, features_names, lesions

def filtering(df_ori, df_pert, kind):
    """
    Filtering of radiomic features (aka feature selection) with Spearman.
    We can filter by  precise features + Spearman OR Spearman only.
    
    -----------
    Input
    -----------
    - df_ori: dataframe with values from original features
    - df_pert: dataframe with values from perturbed features
    - kind: type of filtering ("precise" or "all")
    
    -----------
    Output
    -----------
    - df1_ori: dataframe with values of selected original features
    - df1_pert: dataframe with values of selected original features
    - selected_feat: list of selected features
    in addition, 3 figures are saved in relation to correlation matrix
    """

    if kind=="precise":
        #Filtering PRECISE features
        robust_names=['firstorder_10Percentile',
            'firstorder_90Percentile',
            'firstorder_Energy',
            'firstorder_Mean',
            'firstorder_Median',
            'firstorder_Minimum',
            'firstorder_RootMeanSquared',
            'glcm_Autocorrelation',
            'glcm_JointAverage',
            'glcm_SumAverage',
            'gldm_DependenceEntropy',
            'gldm_GrayLevelNonUniformity',
            'gldm_HighGrayLevelEmphasis',
            'gldm_LargeDependenceLowGrayLevelEmphasis',
            'gldm_LowGrayLevelEmphasis',
            'gldm_SmallDependenceHighGrayLevelEmphasis',
            'glrlm_GrayLevelNonUniformity',
            'glrlm_HighGrayLevelRunEmphasis',
            'glrlm_LongRunHighGrayLevelEmphasis',
            'glrlm_LongRunLowGrayLevelEmphasis',
            'glrlm_LowGrayLevelRunEmphasis',
            'glrlm_RunLengthNonUniformity',
            'glrlm_RunPercentage',
            'glrlm_RunVariance',
            'glrlm_ShortRunHighGrayLevelEmphasis',
            'ngtdm_Coarseness']

        df_ori=df_ori[robust_names]
        df_pert=df_pert[robust_names]
    
    else:
        robust_names = list(df_ori.columns)

    # We need to use the same features to compare original and perturbed habitats.
    corr_matrix, p_matrix = stats.spearmanr(df_ori) # compute correlation and p-value matrix
    mask = np.triu(np.ones_like(corr_matrix), k=1).astype(bool) # mask with only lower triangle of corr matrix
    p_mask = np.triu(np.ones_like(p_matrix), k=1).astype(bool) # mask with only lower triangle of pvalue matrix
    combined_mask = mask & (p_matrix < 0.05) # select only elements of lower traingle corrmask if correlaitons are significant
    # Use the combined mask to identify the highly correlated features that are statistically significant
    to_drop = [column for column in df_ori.columns if any(combined_mask[df_ori.columns.get_loc(column)] & (corr_matrix[:, df_ori.columns.get_loc(column)] > 0.7))]
    
    # Drop the highly correlated features from the dataframes
    df1_ori = df_ori.drop(to_drop, axis=1)
    df1_pert = df_pert.drop(to_drop, axis=1)
    selected_feat = df1_ori.columns

    # # Plot and save the correlation matrices
    # fig, ax = plt.subplots(figsize=(19, 15))
    # sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
    # ax.set_title('Spearman Correlation Matrix', fontsize=16)
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    # ax.set_yticklabels(ax.get_yticklabels(), rotation=0, va='center', fontsize=8)
    # plt.tight_layout()
    # fig_name = '/nfs/rnas/REPRO/paper/code/dice/test/gmm_sep_reshaped2/experiments16feb/corr_matrix_spicystats2.jpg'
    # plt.savefig(fig_name, format='jpg', dpi=1200, bbox_inches='tight')
    # plt.clf()

    # fig, ax = plt.subplots(figsize=(19, 15))
    # sns.heatmap(p_matrix, annot=True, cmap='coolwarm', ax=ax)
    # ax.set_title('P-value Matrix', fontsize=16)
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    # ax.set_yticklabels(ax.get_yticklabels(), rotation=0, va='center', fontsize=8)
    # plt.tight_layout()
    # fig_name = '/nfs/rnas/REPRO/paper/code/dice/test/gmm_sep_reshaped2/experiments16feb/pvalue_matrix_spicystats2.jpg'
    # plt.savefig(fig_name, format='jpg', dpi=1200, bbox_inches='tight')
    # plt.clf()
    
    # fig, ax = plt.subplots(figsize=(19, 15))
    # sns.heatmap(df1_ori.corr(method='spearman'), annot=True, cmap='coolwarm', ax=ax)
    # ax.set_title('Filtered Correlation Matrix', fontsize=16)
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    # ax.set_yticklabels(ax.get_yticklabels(), rotation=0, va='center', fontsize=8)
    # plt.tight_layout()
    # fig_name = '/nfs/rnas/REPRO/paper/code/dice/test/gmm_sep_reshaped2/experiments16feb/filtered_corr_matrix_spicystats2.jpg'
    # plt.savefig(fig_name, format='jpg', dpi=1200, bbox_inches='tight')
    # plt.clf()
    return df1_ori, df1_pert, selected_feat

def save_txt(odir_les_txt_csv,name,selected_feat):
       """
    This function saves the selected features to a text file.

    -----------
    Input
    -----------
    - odir_les_txt_csv: output directory where the text file will be saved
    - name: name of the lesion
    - selected_feat: list of selected features

    -----------
    Output
    -----------
    - A text file containing the selected features is saved to the output directory.
    """
    with open(odir_les_txt_csv+'/'+name+'_features.txt', 'w') as f:
        f.write('\n'.join(selected_feat))

def compute_habitats(df1, df1_pert, Allfeatures, k, shape_len):
     """
    This function inputs two dataframes "df1" and "df1_pert" and an array "Allfeatures", 
    and an optimal k, and outputs two arrays with predicted labels (habitats) for each voxel in df1 and df1_pert, 
    matched to the cluster means of df1 using Euclidean distance.

    -----------
    Input
    -----------
    - df1: dataframe containing original data
    - df1_pert: dataframe containing perturbed data
    - Allfeatures: numpy array containing values of the voxels for all features extracted from both the original and perturbed lesions
    - k: optimal number of clusters
    - shape_len: list of tuples containing the shape of each lesion

    -----------
    Output
    -----------
    - ori_cluster_3D: 3D numpy array containing predicted habitat labels for each voxel in the original data
    - pert_cluster_3D: 3D numpy array containing predicted habitat labels for each voxel in the perturbed data, matched to the cluster means of the original data


    This function inputs two dataframes "df1" and "df1_pert" and an array "Allfeatures", 
    and an optimal k, and outputs two arrays with predicted labels (habitats) for each voxel in df1 and df1_pert, 
    matched to the cluster means of df1 using Euclidean distance"""

    # Fit and predict for original
    dataset=np.array(df1)
    model=GMM(n_components=k, random_state=123) #creating model
    model.fit(dataset)
    ori_labels = model.predict(dataset) # add 1 to start the labels at 1
    ori_cluster=np.zeros(Allfeatures.shape[0])-1 #new 1D numpy array with the same number of rows (voxels) as allfeatures, filled with -1 values
    ori_cluster[np.where(~np.isnan(Allfeatures))[0]]=1 #array will have -1s in NaN values and 1s in non-NaN values
    ori_cluster[ori_cluster==1]=ori_labels+1 #transferring labels to mask so I have an array with -1s in NaN voxels and predicted labels in my non-NaN voxels. Labels+1 means that labels go from 1 to k.
    
    # Fit and predict for perturbed
    pert_dataset = np.array(df1_pert)
    pert_model=GMM(n_components=k, random_state=123) #creating model
    pert_model.fit(pert_dataset)
    pert_labels = pert_model.predict(pert_dataset) # add 1 to start the labels at 1
    pert_cluster=np.zeros(Allfeatures.shape[0])-1 #new 1D numpy array with the same number of rows (voxels) as allfeatures, filled with -1 values
    pert_cluster[np.where(~np.isnan(Allfeatures))[0]]=1 #array will have -1s in NaN values and 1s in non-NaN values
    pert_cluster[pert_cluster==1]=pert_labels+1 #transferring labels to mask so I have an array with -1s in NaN voxels and predicted labels in my non-NaN voxels. Labels+1 means that labels go from 1 to k.
    
    # Match perturbed labels to original via Hungarian algorithm
    cont_mat = contingency_matrix(ori_cluster,pert_cluster) #counts number of voxels that belong to each cluster
    cluster_matcher= munkres.Munkres().compute(cont_mat.max() - cont_mat)  #finds optimal one-to-one matching between clusters via the Munkres algorithm
    ori_values = np.unique(ori_cluster)
    pert_values = np.unique(pert_cluster)
    matching_dict = {}
    for match in cluster_matcher:
        matching_dict[pert_values[match[1]]] = ori_values[match[0]]
    pert_cluster=[matching_dict[cluster] for cluster in pert_cluster]
    pert_cluster=np.array(pert_cluster)

    # Reshape to 3D 
    ori_cluster_3D=ori_cluster.reshape(shape_len[0][0])
    pert_cluster_3D=pert_cluster.reshape(shape_len[0][0])
    return ori_cluster_3D, pert_cluster_3D

def save_nifti_fig(les,lesion, name,feat_im,k,odir_les_hab,odir_les_figures):
     """
    This function saves the habitats as a NIfTI file and a middle slice of the habitat as a JPG image.

    -----------
    Input
    -----------
    - les: numpy array containing predicted habitat labels for each voxel
    - lesion: lesion name
    - name: name of the habitat
    - feat_im: SimpleITK image object containing the original image
    - k: number of clusters (habitats)
    - odir_les_hab: output directory where the NIfTI file will be saved
    - odir_les_figures: output directory where the JPG image will be saved

    -----------
    Output
    -----------
    - NIfTI file containing habitats saved to the specified output directory
    - JPG image of the middle slice of the habitat saved to the specified output directory
    """
    # Saving habitats as niftii
    cmap=cm.get_cmap("Paired", k)
    cmap.set_under(color="white")
    cluster_arr=np.transpose(les)
    cluster_img = sitk.GetImageFromArray(cluster_arr)
    cluster_img.CopyInformation(feat_im)
    nifti_path = os.path.join(odir_les_hab, f"{lesion}_{name}_{k}.nii")
    sitk.WriteImage(cluster_img, nifti_path)

    # Saving middle slice of habitat as jpg
    sns.set_theme(style="ticks")
    slice_path = os.path.join(odir_les_figures, f"{lesion}_{name}_{k}.jpg")
    fig = plt.figure(1,figsize=(4, 3))
    dslice = cluster_arr[int(cluster_arr.shape[2]/2),:,:]
    print(int(cluster_arr.shape[2]/2))
    im=plt.imshow(dslice, cmap=cmap,vmin=0.0000000001, origin='lower', extent=[0, cluster_arr.shape[1], 0, cluster_arr.shape[0]])
    # plt.colorbar(im, ticks=np.arange(1,k+1)) #colors as a colorbar
    handles = [Rectangle((0, 0), 1, 1, color=cmap(i)) for i in range(k)]
    labels = ['{}'.format(i+1) for i in range(k)]
    plt.title(name, fontsize=16, weight='bold', y=1.09)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    legend=plt.legend(handles, labels, loc='center', bbox_to_anchor=(0.5, -0.5), ncol=3, fontsize=12, title="Habitats", title_fontsize='large')
    legend.get_title().set_fontweight('bold')
    fig.savefig(slice_path, format='jpg', dpi=300, bbox_inches='tight')
    # plt.show()
    plt.clf()

def main(argv):
    lesion=argv[0] #lesion name
    idir=argv[1] #main directory where original and perturbed 3D features are
    odir = argv[2] #main output directory

    print("Reading data for lesion: ", lesion)
    # Reading paths to original and perturbed 3D radiomic features
    ori_dir=idir+'/original_R3B12' #3D radiomics ORIGINAL
    pert_dir=idir+'/perturbed_R3B12' #3D radiomics PERTURBED
   
    # Creating output folders for each of the outputs of interest
    odir_les=odir+'/'+lesion #main output folder, one per lesion
    odir_les_hab=odir_les+'/habitats'    #subfolder: output habitats
    odir_les_dice_csv=odir_les+'/dice_csv' #subfolder: output both txt files (with lists fo rprecise + all features elected) + csv DICE score
    odir_les_txt_csv=odir_les+'/txt_csv' #subfolder: output both txt files (with lists fo rprecise + all features elected) + csv DICE score
    odir_les_figures=odir_les+'/figures' #subfolder: output figures
    if not os.path.exists(odir_les): os.mkdir(odir_les)
    if not os.path.exists(odir_les_hab): os.mkdir(odir_les_hab)
    if not os.path.exists(odir_les_txt_csv): os.mkdir(odir_les_txt_csv)
    if not os.path.exists(odir_les_dice_csv): os.mkdir(odir_les_dice_csv)
    if not os.path.exists(odir_les_figures): os.mkdir(odir_les_figures)
   
    # Obtaining my dataframe (which contains all voxel values of all features for both original and perturbed lesion) to clusterize
    Allfeatures, shape_len, f_names, lesions = readCT(ori_dir,lesion)
    Allfeatures_pert, shape_len_pert, f_names_pert, lesions_pert = readCT(pert_dir,lesion)
    
    # Create 2D numpy array containing only non-NaN values of original AllFeatures array, then convert to df.
    Allfeatures_mask=np.reshape(Allfeatures[np.where(~np.isnan(Allfeatures))],[-1,Allfeatures.shape[1]]) #select only non-NaN values and then reshape to 2D np array
    df = pd.DataFrame(Allfeatures_mask) #mask only contains non_NaN values, this is the input to the clustering model, I want to predict labels (habitats) for each voxel
    Allfeatures_mask_pert=np.reshape(Allfeatures_pert[np.where(~np.isnan(Allfeatures_pert))],[-1,Allfeatures_pert.shape[1]]) #select only non-NaN values and then reshape to 2D np array
    df_pert = pd.DataFrame(Allfeatures_mask_pert) #mask only contains non_NaN values, this is the input to the clustering model, I want to predict labels (habitats) for each voxel

    # We rename the columns of the dataframes to match the names of the features
    f_names=[f_names[i] for i in list(df.columns)]
    df.columns=list(f_names)
    df_pert.columns=list(f_names)
    print("Data reading completed!")

    print("Starting feature selection...")
    # Filter features & save selected features
    filter_kind='precise'
    filter_kind2='all'
    df1, df1_pert, selected_feat=filtering(df, df_pert, filter_kind)
    df2, df2_pert, selected_feat2=filtering(df,df_pert, filter_kind2)

    save_txt(odir_les_txt_csv,'precise',selected_feat)
    save_txt(odir_les_txt_csv,'all',selected_feat2)
    print("Feature selection completed!")

    print("Starting habitat computation...")
    # Compute Habitats
    k=optimal_k(df1) #choosing optimal k
    ori_cluster, pert_cluster = compute_habitats(df1, df1_pert, Allfeatures, k, shape_len)
    ori_cluster2, pert_cluster2 = compute_habitats(df2, df2_pert, Allfeatures, k, shape_len)
    print("Habitats computed! ")

    print("Starting dice computation...")
    # Compute DICE
    d = pd.DataFrame()
    d2 = pd.DataFrame()
    for h in range(1,k+1):
        dice = dsc(ori_cluster,pert_cluster, h)
        dice2 = dsc(ori_cluster2, pert_cluster2, h)
        
        d=d.append({'lesionID':lesion,
                'feat_selection': filter_kind,
                'n_radiohabitats': k,
                'habitat': h,
                'lesion_location': 'liver',
                'DSC': dice }, ignore_index=True)

        d2=d2.append({'lesionID':lesion,
                'feat_selection': filter_kind2,
                'n_radiohabitats': k,
                'habitat': h,
                'lesion_location': 'liver',
                'DSC': dice2 }, ignore_index=True)
    dfinal = pd.concat([d,d2])
    # csv_file = odir_les_dice_csv+ '/'+ lesion +'.csv'
    # dfinal.to_csv(csv_file, index = False)
    print("DICE computed! ")

    print("Saving habitats...")
    # Save Habitats as niftii and middle slice as figures
    feat_im,feat_im_pert=get_featim(idir,lesion)
    save_nifti_fig(ori_cluster,lesion, "precise_original", feat_im,k,odir_les_hab,odir_les_figures)
    save_nifti_fig(pert_cluster,lesion, "precise_perturbed", feat_im_pert,k,odir_les_hab,odir_les_figures)
    save_nifti_fig(ori_cluster2,lesion, "all_original", feat_im,k,odir_les_hab,odir_les_figures)
    save_nifti_fig(pert_cluster2,lesion, "all_perturbed", feat_im_pert,k,odir_les_hab,odir_les_figures)

    end_t = time.time()
    total_t = end_t - start_t
    total_t_min=total_t/60
    print("Total time was: ", np.round(total_t,3), 'seconds or ', np.round(total_t_min,3), 'minutes!')

if __name__ == "__main__":
   main(sys.argv[1:])
   print('All done! Enjoy your habitats!')



