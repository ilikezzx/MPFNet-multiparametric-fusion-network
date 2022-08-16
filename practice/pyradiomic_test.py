#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：bone-segmentation-1 
@Author  ：zzx
@Date    ：2022/8/16 11:14
Description:
    有关 Pyradiomics的练习和测试
"""

import nrrd
import logging
import SimpleITK as sitk
import radiomics

from radiomics import featureextractor

if __name__ == '__main__':
    # Get some test data Download the test case to temporary files and return it's location. If already downloaded,
    # it is not downloaded again, but it's location is still returned.
    imageName = r'C:\Users\12828\Desktop\osteosarcoma\bone tumor data\1038501 qizhongyi\20140124MRI\S5 T2 tra STIR 1\1038501 qizhongyi_20140124MRI_S5 T2 tra STIR 1.nrrd'
    maskName = r'C:\Users\12828\Desktop\osteosarcoma\bone tumor data\1038501 qizhongyi\20140124MRI\S5 T2 tra STIR 1\Untitled.nii.gz'

    if imageName is None or maskName is None:  # Something went wrong, in this case PyRadiomics will also log an error
        print('Error getting testcase!')
        exit()

    dicom = sitk.ReadImage(imageName)
    # 像素比例长宽高信息
    resampledPixelSpacing = dicom.GetSpacing()


    # Regulate verbosity with radiomics.verbosity (default verbosity level = WARNING)
    # radiomics.setVerbosity(logging.INFO)

    # Get the PyRadiomics logger (default log-level = INFO)
    logger = radiomics.logger
    logger.setLevel(logging.DEBUG)  # set level to DEBUG to include debug log messages in log file

    logger = logging.getLogger("radiomics.glcm")
    logger.setLevel(logging.ERROR)

    # Set up the handler to write out all log entries to a file
    handler = logging.FileHandler(filename='testLog.txt', mode='w')
    formatter = logging.Formatter("%(levelname)s:%(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Define settings for signature calculation
    # These are currently set equal to the respective default values
    settings = {}
    settings['binWidth'] = 25
    settings['resampledPixelSpacing'] = resampledPixelSpacing
    # [3,3,3] is an example for defining resampling (voxels with size 3x3x3mm)
    settings['interpolator'] = sitk.sitkBSpline

    # Initialize feature extractor
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)

    # By default, only original is enabled. Optionally enable some image types:
    extractor.enableImageTypes(Original={}, LoG={}, Wavelet={})

    # Disable all classes except firstorder
    # extractor.disableAllFeatures()

    # Enable all features in firstorder
    # extractor.enableFeatureClassByName('firstorder')

    # Only enable mean and skewness in firstorder
    # extractor.enableFeaturesByName(firstorder=['Mean', 'Skewness'])

    print("Calculating features")
    featureVector = extractor.execute(imageName, maskName)

    for featureName, featureValue in featureVector.items():
        print("%s: %s" % (featureName, featureValue))
