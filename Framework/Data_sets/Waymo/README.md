The raw data can be obtained at [Google Cloud](https://console.cloud.google.com/storage/browser/waymo_open_dataset_motion_v_1_1_0/uncompressed/scenario;tab=objects?prefix=&forceOnObjectsSortingFiltering=false), which will require [regersting with the Waymo Open dataset](https://waymo.com/intl/en_us/open/download/).

There, one has to download the three folder named *training/*, *validation/*, and *testing/*.

Those downloads then need then need do be extracted into the *Waymo/data/* folder, where they need to be rearranged to build the following structure:
```
└── ../Waymo/data
    ├── training/
    │   ├── training.tfrecord-00000-of-01000
    │   ├── training.tfrecord-00001-of-01000
    │   └── ...
    ├── validation/
    │   ├── validation.tfrecord-00000-of-00150
    │   ├── validation.tfrecord-00001-of-00150
    │   └── ...
    └── testing/
        ├── testing.tfrecord-00000-of-00150
        ├── testing.tfrecord-00001-of-00150
        └── ...
```
