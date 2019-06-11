from nilearn import plotting
from nilearn.image import resample_img, mean_img
from nilearn import datasets


# Data directory, include dicom files for multiple patients for all testing
test_dir = '/home/harryzhang/Desktop/test_registration/'


# plot MNI 152 atlas
plotting.plot_anat(test_dir+'baseanat.nii.gz')

# plot anatomical of dicom
plotting.plot_anat(test_dir+'t2_from_dicom.nii.gz')

# plot skull stripping result
plotting.plot_anat(test_dir+'t2_stripped.nii.gz')

# registered T2
plotting.plot_anat(test_dir+'r_t2.nii.gz')

# DWI regenerated from dicom
plotting.plot_anat(test_dir+'dwia.nii.gz')

# DWI registered
plotting.plot_anat(test_dir+'r_dwi.nii.gz')

# ROI plot of T2 registered
plotting.plot_roi(test_dir+'r_t2.nii.gz')
# ROI plot of DWI registered
plotting.plot_roi(test_dir+'r_dwi.nii.gz')

# calculate mean of time series for registered perfusion
mean_pwi = mean_img(test_dir+'r_pwi.nii.gz')
# plot registered mean image of perfusion
plotting.plot_anat(mean_pwi)

# calculate mean of time series for raw perfusion
mean_pwi = mean_img(test_dir+'pwi_from_dicom.nii.gz')
# plot raw mean image of perfusion
plotting.plot_anat(mean_pwi)

# load ICBM T2 atlas
atlas = nib.load(test_dir+'ICBM_T2Atlas.hdr')
# down sample the atlas to our dwi shape
atlas_ds = resample_img(atlas, target_affine=atlas.affine, target_shape=dwi.shape, interpolation='nearest')

plotting.plot_anat(test_dir+'ICBM_T2Atlas.hdr')

# another MNI152 ICBM atlas for use, there are several versions
# select different ones and repeat the image registration and compare the best result
plotting.plot_roi(test_dir+'/mni_icbm152_nlin_sym_09a/mni_icbm152_t2_tal_nlin_sym_09a.nii')

# use harvard oxford atlas to map to DWI, T2 and mean Perfusion
dataset = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
ho_maps = dataset.maps
plotting.plot_roi(ho_maps, test_dir+'r_dwi.nii.gz', title="H-O atlas on DWI")

plotting.plot_roi(ho_maps, test_dir+'r_t2.nii.gz', title="H-O atlas on T2")

plotting.plot_roi(ho_maps, mean_pwi, title="H-O atlas on mean PWI")