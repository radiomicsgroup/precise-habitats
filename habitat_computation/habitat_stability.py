### Analyze Habitat Stability (DICE Score) via Wilcoxon +  plot results

#Import libraries
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import seaborn as sns
import scipy.stats as stats
import statannot
from statannot import add_stat_annotation
import statannotations
from statannotations.Annotator import Annotator
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import pingouin as pg

# function that defines colors
def rgb(red, green, blue, maxColorValue):
    return (red / maxColorValue, green / maxColorValue, blue / maxColorValue)
    
col_1 = rgb(red=1, green=70, blue=100, maxColorValue=255)  # dark blue-green
col_2 = rgb(red=242, green=150, blue=2, maxColorValue=255)  # orange
col_3 = rgb(red=210, green=19, blue=44, maxColorValue=255)  # deep red
col_4 = rgb(red=7, green=157, blue=128, maxColorValue=255)  # teal
col_5 = rgb(red=255, green=218, blue=185, maxColorValue=255) # peach
col_6 = rgb(red=176, green=224, blue=230, maxColorValue=255) # powder blue
col_7 = rgb(red=221, green=160, blue=221, maxColorValue=255) # lavender blush

#output directory to save figures
odir= '/nfs/rnas/PRECISION/pretty_figures'

############ STATISTICS
# Using Wilcoxon Test (non parametric disitrbutin)
# Reset the indices of the dataframes
cohort = 'COHORT'
data_df = pd.read_csv('/nfs/rnas/PRECISION/code/dice/dice_' + cohort + '.csv')
data_df_all = data_df[data_df['feat_selection'] == 'all']
data_df_precise = data_df[data_df['feat_selection'] == 'precise']
data_df_all.reset_index(drop=True, inplace=True)
data_df_precise.reset_index(drop=True, inplace=True)

#liver and lung
data_df_all_liver=data_df_all[data_df_all['lesion_location']=='liver']
data_df_precise_liver=data_df_precise[data_df_precise['lesion_location']=='liver']
data_df_all_lung=data_df_all[data_df_all['lesion_location']=='lung']
data_df_precise_lung=data_df_precise[data_df_precise['lesion_location']=='lung']

# Medians
print("MEDIANS")
all_liver=data_df_all_liver['DSC']
precise_liver=data_df_precise_liver['DSC']
all_lung=data_df_all_lung['DSC']
precise_lung=data_df_precise_lung['DSC']
print('Median (IQR) liver all:', np.percentile(all_liver,50), '(', np.percentile(all_liver,25), '-',np.percentile(all_liver,75), ')')
print('Median (IQR) liver precise:', np.percentile(precise_liver,50), '(', np.percentile(precise_liver,25), '-',np.percentile(precise_liver,75), ')')
print('Median (IQR) lung all:', np.percentile(all_lung,50), '(', np.percentile(all_lung,25), '-',np.percentile(all_lung,75), ')')
print('Median (IQR) lung precise:', np.percentile(precise_lung,50), '(', np.percentile(precise_lung,25), '-',np.percentile(precise_lung,75), ')')

# Perform Wilcoxon signed-rank test
print("WILCOXON")
stat_liver, p_liver = stats.wilcoxon(data_df_precise_liver['DSC'], data_df_all_liver['DSC'])
stat_lung, p_lung = stats.wilcoxon(data_df_precise_lung['DSC'], data_df_all_lung['DSC'])

alpha = 0.05
if p_liver > alpha:
    print('Wilcoxon signed-rank test results: z = {:.3f}, p = {:.9f}'.format(stat_liver, p_liver))
else:
    print('SIGNIFICANT FOR LIVER!')
    print('Wilcoxon signed-rank test results: z = {:.3f}, p = {:.9f}'.format(stat_liver, p_liver))

if p_lung > alpha:
    print('Wilcoxon signed-rank test results: z = {:.3f}, p = {:.9f}'.format(stat_lung, p_lung))
else:
    print('SIGNIFICANT FOR LUNG!')
    print('Wilcoxon signed-rank test results: z = {:.3f}, p = {:.9f}'.format(stat_lung, p_lung))

print("EFFECT SIZE")

# Compute Cohen's delta
cohen_d_liver = pg.compute_effsize(data_df_precise_liver['DSC'], data_df_all_liver['DSC'], paired=True, eftype='cohen')
cohen_d_lung = pg.compute_effsize(data_df_precise_lung['DSC'], data_df_all_lung['DSC'], paired=True, eftype='cohen')

print('Cohen\'s delta for liver:', cohen_d_liver)
print('Cohen\'s delta for lung:', cohen_d_lung)



####### BOXPLOT
x="lesion_location"
y="DSC"
hue="feat_selection"
hue_order=['precise', 'all']

pairs = [(("liver", "precise"), ("liver", "all")),
    (("lung", "precise"), ("lung", "all"))]
sns.set_style('white')

ax = sns.boxplot(data = data_df,
                hue = hue,
                hue_order=hue_order,
                x = x,
                y = y,
                order = ['liver', 'lung'], palette=[col_3,col_4])

annot = Annotator(None, pairs)
annot.new_plot(ax, pairs, plot='boxplot',
           data=data_df, x=x, y=y, hue=hue, hue_order=hue_order)
annot.configure(test='Wilcoxon', loc='inside', verbose=1, )
annot.apply_test().annotate()
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.ylabel("DSC", fontsize=14, weight='bold')
labels=['Liver', 'Lung']
ax.set_xticklabels(labels, weight='bold')
ax.set(xlabel=None)
legend_elements = [Patch(facecolor=col_3, edgecolor='none',
                         label='Precise'), Patch(facecolor=col_4, edgecolor='none',
                         label='Non-Precise')]
legend=plt.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, -0.4), ncol=2, fontsize=12, title="Features", title_fontsize='large')
legend.get_title().set_fontweight('bold')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.savefig(odir+'/Figure5_dice.jpg',bbox_inches='tight', dpi=300)