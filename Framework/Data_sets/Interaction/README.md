In order to obtain the raw data it is necessary to fill out a **Request Form** provided on [https://interaction-dataset.com/]{https://interaction-dataset.com/}. In the Request Form, select 'Behavior prediction / trajectory forecasting' as the **Research Area**.

Upon doing this, you should receive an E-mail after some days.

In this E-mail, under the point **3) Download links for prediction** click *Data for Multi-Agent Tracks*. This will initiate a download of the files.

The data should be extracted and placed into the following structure:

```
└── ../Interaction/data
    ├── maps
    │   ├── *.osm
    │   ├── *.osm_xy 
    ├── test_conditional-multi-agent
    │   ├── *.csv
    ├── test_multi-agent
    │   ├── *.csv
    ├── train
    │   ├── *.csv
    └── val
        ├── *.csv
```
There should be 18 .osm and 18 .osm\_xy files in the *maps* folder, 15 .csv files in *test\_conditional-multi-agent*, 17 .csv files in *test_multi-agent*, 12 .csv files in *train*, and 12 .csv files in *val*.

