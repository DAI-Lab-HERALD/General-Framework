In order to obtain the raw data it is necessary to first create an account at https://www.nuscenes.org.

Upon doing this, go to https://www.nuscenes.org/nuscenes.

Go to the section titled **Downloads**.

Under **Map expansion** download *Map expansion pack (v1.3) \[US\]*.

Under **Full dataset (v1.0)** download *Metadata \[US\]*.

The data should be extracted and placed into the following structure:

```
├── NuScenes
│   ├── data
│   │   ├── maps
│   │   │   ├── basemap
│   │   │   │   ├── boston-seaport.png
│   │   │   │   ├── singapore-hollandvillage.png
│   │   │   │   ├── singapore-onenorth.png
│   │   │   │   ├── singapore-queenstown.png
│   │   │   ├── expansion
│   │   │   │   ├── boston-seaport.json
│   │   │   │   ├── singapore-hollandvillage.json
│   │   │   │   ├── singapore-onenorth.json
│   │   │   │   ├── singapore-queenstown.json
│   │   │   ├── prediction
│   │   │   │   ├── prediction_scenes.json
│   │   │   ├── 36092f0b03a857c6a3403e25b4b7aab3.png
│   │   │   ├── 37819e65e09e5547b8a3ceaefba56bb2.png
│   │   │   ├── 53992ee3023e5494b90c316c183be829.png
│   │   │   ├── 93406b464a165eaba6d9de76ca09f5da.png
│   │   ├── v1.0-trainval
│   │   │   ├── attribute.json
│   │   │   ├── calibrated_sensor.json
│   │   │   ├── category.json
│   │   │   ├── ego_pose.json
│   │   │   ├── instance.json
│   │   │   ├── log.json
│   │   │   ├── map.json
│   │   │   ├── sample.json
│   │   │   ├── sample_annotation.json
│   │   │   ├── sample_data.json
│   │   │   ├── scene.json
│   │   │   ├── sensor.json
│   │   │   ├── visibility.json
```


