In order to obtain the raw data, refer to the instructions at https://www.round-dataset.com/.

The data should include the following files in this folder:
```
└── ../RounD_round_about
    ├── data
    │   ├── xx_background.png
    │   ├── xx_recordingMeta.csv
    │   ├── xx_trackMeta.csv
    │   └── xx_tracks.csv
    └── maps
        └── lanelets
            ├── 0_neuweiler
            │   └── location0.osm
            ├── 1_kackertstrasse
            │   └── location1.osm
            └── 2_thiergarten
                └── location2.osm
```
Here, *xx* ranges from 00 to 23.

After placing the raw data in the /data/ folder, one has to execute the [RoundD_processing.py](https://github.com/julianschumann/General-Framework/blob/main/Framework/Data_sets/RounD_round_about/RounD_processing.py) script in the folder above for preprocessing.
