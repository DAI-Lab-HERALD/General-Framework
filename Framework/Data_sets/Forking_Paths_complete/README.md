In order to obtain the raw data, use the Google Drive link at [Forking Paths GitHub repository](https://github.com/JunweiLiang/Multiverse#the-forking-paths-dataset).

The raw data includes as folder called *next_x_v1_dataset*. This folder includes the following four folders:
    - bbox
    - rgb_videos
    - seg_videos
    - temp

For more information on the contents of these folders we direct users to the [dataset GitHub repository](https://github.com/JunweiLiang/Multiverse/blob/master/forking_paths_dataset/README.md).

As we are only extracting trajectories from a top-down view, the only raw data required in the /data/ folder are the *cam4.json files from the bbox folder.

After placing the raw data in the /data/ folder, one has to execute the FP_processing.py script in the folder above for preprocessing.
