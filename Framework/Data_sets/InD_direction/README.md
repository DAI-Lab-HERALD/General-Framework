In order to obtain the raw data, refer to the instructions at https://www.ind-dataset.com/.

The data should include the following files in this folder:
```
└── ../InD_direction
    ├── data
    │   ├── xx_background.png
    │   ├── xx_recordingMeta.csv
    │   ├── xx_trackMeta.csv
    │   └── xx_tracks.csv
    └── maps
        └── lanelets
            ├── 1_bendplatz
            │   └── location1.osm
            ├── 2_frankenburg
            │   └── location2.osm
            ├── 3_heckstrasse
            │   └── location3.osm
            └── 4_aseag
                └── location4.osm
```
Here, *xx* ranges from 00 to 32.

After placing the raw data in the /data/ folder, one has to execute the [InD_processing.py](https://github.com/julianschumann/General-Framework/blob/main/Framework/Data_sets/InD_direction/InD_processing.py) script in the folder above for preprocessing.
