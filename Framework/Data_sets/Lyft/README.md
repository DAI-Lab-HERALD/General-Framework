The raw data can be obtained at the [Toyota website](https://woven.toyota/en/prediction-dataset).

There, one has to download the [sample dataset](https://d20lyvjneielsk.cloudfront.net/prediction-sample.tar), the [first](https://d20lyvjneielsk.cloudfront.net/prediction-train.tar) and 
[second](https://d20lyvjneielsk.cloudfront.net/prediction-train_full.tar) part of the training set, as well as the [validation data](https://d20lyvjneielsk.cloudfront.net/prediction-validate.tar).
Furthermore, the [semantic map](https://d20lyvjneielsk.cloudfront.net/prediction-semantic_map.tar) and [aerial map](https://d20lyvjneielsk.cloudfront.net/prediction-aerial_map.tar) are needed as well.

Those downloads then need then need do be extracted into the *Lyft/data/* folder, where they need to be rearranged to bild the following structure:
```
├── ../Lyft/data
│   ├── LICENSE
│   ├── aerial_map
│   │   ├── aerial_map.png
│   │   ├── nearmap_images
│   |   │   ├── ...
│   ├── feedback.txt
│   ├── meta.json
│   ├── scenes
│   │   ├── sample.zarr
│   |   │   ├── ...
│   |   ├── train.zarr
│   |   │   ├── ...
│   |   ├── train.zarr
│   |   │   ├── ...
│   |   ├── train_full.zarr
│   |   │   ├── ...
│   |   ├── validation.zarr
│   |   │   ├── ...
│   ├── semantic_map
│   │   ├── semantic_map.pb
```
