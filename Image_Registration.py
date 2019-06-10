import os
from os.path import abspath
import numpy as np
import nibabel as nib
from nipype import Workflow, Node, MapNode, Function
from nipype.interfaces import fsl
from nipype.interfaces.dcm2nii import Dcm2niix
from nipype.testing import example_data
import Utils.applyxfm4d


#Data directory, include dicom files for multiple patients for all testing
test_dir = '/home/harryzhang/Desktop/test_registration/'


"""
Step 1
Dicom to Nifti format conversion

Tool used: Dcm2niix (https://github.com/rordenlab/dcm2niix)
install dcm2niix first, then use nipype wrapper instead of cmd

sample code here shows only T2, DWI and Perfusion conversion
adjust source_dir for other modalities
@param:
compression: Gz compression level, 1=fastest, 9=smallest
generated cmd line:
'dcm2niix -b y -z y -5 -x n -t n -m n -o output_dir -s n -v n source_dir' 
"""
# T2 nifti generation
converter = Dcm2niix()
converter.inputs.source_dir = test_dir+'dicom/ep2d_diff_3scan_trace*ep_b0'
converter.inputs.compression = 5
converter.inputs.out_filename = 't2_from_dicom'
converter.inputs.output_dir = test_dir
converter.run()

# dwi nifty generation
converter = Dcm2niix()
converter.inputs.source_dir = test_dir+'dicom/ep2d_diff_3scan_trace*ep_b1000t'
converter.inputs.compression = 5
converter.inputs.out_filename = 'dwi'
converter.inputs.output_dir = test_dir
converter.run()

# perfusion nifti generation
converter = Dcm2niix()
converter.inputs.source_dir = test_dir+'dicom/ep2d_perf'
converter.inputs.compression = 5
converter.inputs.out_filename = 'pwi_from_dicom'
converter.inputs.output_dir = test_dir
converter.run()


"""
Step 2 (Optional)
Skull Stripping

Tool used: FSL (https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FSL)
install FSL for Linux first, then use nipype wrapper instead of cmd

algorithm used: BET (https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/BET/UserGuide)
bet calls the main brain extraction program  bet2

Main bet2 options:
-o generate brain surface outline overlaid onto original image

-m generate binary brain mask

-s generate rough skull image (not as clean as what betsurf generates)

-n don't generate the default brain image output

-f <f> fractional intensity threshold (0->1); default=0.5; smaller values give larger brain outline estimates

-g <g> vertical gradient in fractional intensity threshold (-1->1); default=0; positive values give larger brain outline at bottom, smaller at top

-r <r> head radius (mm not voxels); initial surface sphere is set to half of this

-c < x y z> centre-of-gravity (voxels not mm) of initial mesh surface.

-t apply thresholding to segmented brain image and mask

-e generates brain surface as mesh in .vtk format.

Note that -f is frac, -g is vertical gradient
these two parameters dramatically influence the final result of skull stripping

generated cmd line:
'bet structural.nii brain_anat.nii -f 0.70'
"""
# BET Skull Stripping
# tweaks frac
btr = fsl.BET()
btr.inputs.in_file = test_dir+'t2_from_dicom.nii.gz'
btr.inputs.frac = 0.2
btr.inputs.vertical_gradient = 0.7
btr.inputs.out_file = test_dir+'t2_stripped.nii.gz'
res = btr.run()

"""
Step 3
Register T2 to MNI Atlas

Tool used: FSL-FLIRT (https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FLIRT)
FLIRT is in FSL, use Nipype to call the wrapper

[Mandatory]
in_file: (an existing file name)
        input file
        argument: ``-in %s``, position: 0
reference: (an existing file name)
        reference file
        argument: ``-ref %s``, position: 1

[Optional]
out_file: (a file name)
        registered output file
        argument: ``-out %s``, position: 2
out_matrix_file: (a file name)
        output affine matrix in 4x4 asciii format
        argument: ``-omat %s``, position: 3
out_log: (a file name)
        output log
        requires: save_log
in_matrix_file: (a file name)
        input 4x4 affine matrix
        argument: ``-init %s``
apply_xfm: (a boolean)
        apply transformation supplied by in_matrix_file or uses_qform to use
        the affine matrix stored in the reference header
        argument: ``-applyxfm``
apply_isoxfm: (a float)
        as applyxfm but forces isotropic resampling
        argument: ``-applyisoxfm %f``
        mutually_exclusive: apply_xfm
datatype: ('char' or 'short' or 'int' or 'float' or 'double')
        force output data type
        argument: ``-datatype %s``
cost: ('mutualinfo' or 'corratio' or 'normcorr' or 'normmi' or
          'leastsq' or 'labeldiff' or 'bbr')
        cost function
        argument: ``-cost %s``
cost_func: ('mutualinfo' or 'corratio' or 'normcorr' or 'normmi' or
          'leastsq' or 'labeldiff' or 'bbr')
        cost function
        argument: ``-searchcost %s``
uses_qform: (a boolean)
        initialize using sform or qform
        argument: ``-usesqform``
display_init: (a boolean)
        display initial matrix
        argument: ``-displayinit``
angle_rep: ('quaternion' or 'euler')
        representation of rotation angles
        argument: ``-anglerep %s``
interp: ('trilinear' or 'nearestneighbour' or 'sinc' or 'spline')
        final interpolation method used in reslicing
        argument: ``-interp %s``
sinc_width: (an integer (int or long))
        full-width in voxels
        argument: ``-sincwidth %d``
sinc_window: ('rectangular' or 'hanning' or 'blackman')
        sinc window
        argument: ``-sincwindow %s``
bins: (an integer (int or long))
        number of histogram bins
        argument: ``-bins %d``
dof: (an integer (int or long))
        number of transform degrees of freedom
        argument: ``-dof %d``
no_resample: (a boolean)
        do not change input sampling
        argument: ``-noresample``
force_scaling: (a boolean)
        force rescaling even for low-res images
        argument: ``-forcescaling``
min_sampling: (a float)
        set minimum voxel dimension for sampling
        argument: ``-minsampling %f``
padding_size: (an integer (int or long))
        for applyxfm: interpolates outside image by size
        argument: ``-paddingsize %d``
searchr_x: (a list of from 2 to 2 items which are an integer (int or
          long))
        search angles along x-axis, in degrees
        argument: ``-searchrx %s``
searchr_y: (a list of from 2 to 2 items which are an integer (int or
          long))
        search angles along y-axis, in degrees
        argument: ``-searchry %s``
searchr_z: (a list of from 2 to 2 items which are an integer (int or
          long))
        search angles along z-axis, in degrees
        argument: ``-searchrz %s``
no_search: (a boolean)
        set all angular searches to ranges 0 to 0
        argument: ``-nosearch``
coarse_search: (an integer (int or long))
        coarse search delta angle
        argument: ``-coarsesearch %d``
fine_search: (an integer (int or long))
        fine search delta angle
        argument: ``-finesearch %d``
schedule: (an existing file name)
        replaces default schedule
        argument: ``-schedule %s``
ref_weight: (an existing file name)
        File for reference weighting volume
        argument: ``-refweight %s``
in_weight: (an existing file name)
        File for input weighting volume
        argument: ``-inweight %s``
no_clamp: (a boolean)
        do not use intensity clamping
        argument: ``-noclamp``
no_resample_blur: (a boolean)
        do not use blurring on downsampling
        argument: ``-noresampblur``
rigid2D: (a boolean)
        use 2D rigid body mode - ignores dof
        argument: ``-2D``
save_log: (a boolean)
        save to log file
verbose: (an integer (int or long))
        verbose mode, 0 is least
        argument: ``-verbose %d``
bgvalue: (a float)
        use specified background value for points outside FOV
        argument: ``-setbackground %f``
wm_seg: (a file name)
        white matter segmentation volume needed by BBR cost function
        argument: ``-wmseg %s``
wmcoords: (a file name)
        white matter boundary coordinates for BBR cost function
        argument: ``-wmcoords %s``
wmnorms: (a file name)
        white matter boundary normals for BBR cost function
        argument: ``-wmnorms %s``
fieldmap: (a file name)
        fieldmap image in rads/s - must be already registered to the
        reference image
        argument: ``-fieldmap %s``
fieldmapmask: (a file name)
        mask for fieldmap image
        argument: ``-fieldmapmask %s``
pedir: (an integer (int or long))
        phase encode direction of EPI - 1/2/3=x/y/z & -1/-2/-3=-x/-y/-z
        argument: ``-pedir %d``
echospacing: (a float)
        value of EPI echo spacing - units of seconds
        argument: ``-echospacing %f``
bbrtype: ('signed' or 'global_abs' or 'local_abs')
        type of bbr cost function: signed [default], global_abs, local_abs
        argument: ``-bbrtype %s``
bbrslope: (a float)
        value of bbr slope
        argument: ``-bbrslope %f``
output_type: ('NIFTI' or 'NIFTI_PAIR' or 'NIFTI_GZ' or
          'NIFTI_PAIR_GZ')
        FSL output type
args: (a unicode string)
        Additional parameters to the command
        argument: ``%s``
environ: (a dictionary with keys which are a bytes or None or a value
          of class 'str' and with values which are a bytes or None or a
          value of class 'str', nipype default value: {})
        Environment variables
        
Parameters tweaking here:
bins:100 - 800
cost function: mutual information and correlation ratio. Others not applicable for our problem
interpolation: spline provides the best result
dof: 6 or 12 depends on the atlas

generated cmd line:
'flirt -in t2_from_dicom.nii -ref baseanat.nii -out r_t2.nii.gz -omat r_transform.mat -bins 640 -searchcost mutualinfo'
"""

# register t2 to atlas using mutual information

flt = fsl.FLIRT(bins=640, cost_func='mutualinfo', interp='spline',
                searchr_x=[-180, 180], searchr_y=[-180, 180], searchr_z=[-180,180],dof=6)
flt.inputs.in_file = test_dir+'t2_from_dicom.nii.gz'
flt.inputs.reference = test_dir+'baseanat.nii.gz'
flt.inputs.out_file = test_dir+'r_t2.nii.gz'
flt.inputs.out_matrix_file = test_dir+'r_transform.mat'
res = flt.run()

"""
Step 4
Co-register using registered T2 to DWI, perfusion and all other modalities

Tool used: FSL-ApplyXFM (https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FLIRT/UserGuide)
for applying saved transformations and changing FOVs

use the matrix file that generated from T2 to atlas registration
"""
# register with dwi
applyxfm = fsl.preprocess.ApplyXFM()
applyxfm.inputs.in_file = test_dir+'dwia.nii.gz'
applyxfm.inputs.in_matrix_file = test_dir+'r_transform.mat'
applyxfm.inputs.out_file =  test_dir+'r_dwi.nii.gz'
applyxfm.inputs.reference = test_dir+'r_t2.nii.gz'
applyxfm.inputs.apply_xfm = True
result = applyxfm.run()

# register with perfusion
import os
os.chdir('/home/harryzhang/Desktop/test_registration')
# either use my customized applyxf4d wrapper for perfusion data or directly run this CMD
!applyxfm4D pwi_from_dicom.nii.gz r_dwi.nii.gz r_pwi.nii.gz r_transform.mat -singlematrix

# applyxfm4d = applyxfm4d.ApplyXfm4D()
# applyxfm4d.inputs.in_file = test_dir+'pwi.nii.gz'
# applyxfm4d.inputs.single_matrix = test_dir+'r_transform.mat'
# applyxfm4d.inputs.ref_vol = test_dir+'r_dwi.nii.gz'
# applyxfm4d.inputs.out_file = test_dir+'r_pwi.nii.gz'
# res = applyxfm4d.run()