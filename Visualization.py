from nilearn import plotting
from nilearn.image import smooth_img, resample_img, mean_img
from nilearn import datasets
import matplotlib.pyplot as plt

plotting.plot_anat(test_dir+'baseanat.nii.gz')

plotting.plot_anat(test_dir+'t2_from_dicom.nii.gz')

plotting.plot_anat(test_dir+'t2_stripped.nii.gz')

plotting.plot_anat(test_dir+'r_t2.nii.gz')

plotting.plot_anat(test_dir+'dwia.nii.gz')

plotting.plot_anat(test_dir+'r_dwi.nii.gz')

plotting.plot_roi(test_dir+'r_t2.nii.gz')

plotting.plot_roi(test_dir+'r_dwi.nii.gz')

mean_pwi = mean_img(test_dir+'r_pwi.nii.gz')

plotting.plot_anat(mean_pwi)

mean_pwi = mean_img(test_dir+'pwi_from_dicom.nii.gz')

plotting.plot_anat(mean_pwi)

atlas = nib.load(test_dir+'ICBM_T2Atlas.hdr')
atlas_ds = resample_img(atlas,target_affine=atlas.affine,target_shape=dwi.shape,interpolation='nearest')

plotting.plot_anat(test_dir+'ICBM_T2Atlas.hdr')

plotting.plot_roi(test_dir+'/mni_icbm152_nlin_sym_09a/mni_icbm152_t2_tal_nlin_sym_09a.nii')

dataset = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
ho_maps = dataset.maps
plotting.plot_roi(ho_maps, test_dir+'r_dwi.nii.gz',title="H-O atlas on DWI")

plotting.plot_roi(ho_maps, test_dir+'r_t2.nii.gz',title="H-O atlas on T2")

plotting.plot_roi(ho_maps, mean_pwi, title="H-O atlas on mean PWI")