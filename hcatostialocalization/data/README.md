Datasets depend on two external libraries: HearticDatasetsUtilities and HCATNetwork

[HCATnetwork](https://github.com/AAMIASoftwares-research/HCATNetwork)

[HearticDatasetsUtilities](https://github.com/AAMIASoftwares-research/DatasetUtilities)

Both are public and should be quite easy to explore.

They're both added to install_requirements.sh

To get affine matrices to transform coordinate frames, check the HearticDatasetUtilities
and find the image class for each dataset. There are some useful functions defined that can be used to go from RAS to IJK and vice-versa. 