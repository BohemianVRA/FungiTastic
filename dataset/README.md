# About FungiTastic data

## Download instructions

There are two options to download the dataset:
1. Download the whole dataset from kaggle [~50GB]
2. Use the download script (preferred)

### 1. Download the dataset from kaggle [~50GB]:
While the structure of the data is kept the same, the dataset available on Kaggle only contains the 500p images.

[info] Using the Kaggle API, you have to always download all the data and subsets.

To download the data, you have to:
1. Register and login to Kaggle.
2. Install Kaggle API `pip install kaggle`
4. Store Kaggle login settings and locally.
   ```
   !mkdir ~/.kaggle
   !touch ~/.kaggle/kaggle.json
   api_token = {"username":"FILL YOUR USERNAME","key":"FILL YOUR APIKEY"}
   ```
5. Use CLI `kaggle datasets download -d picekl/fungitastic`

### 2. Use the download script
The preferred approach is to use our script to get the data that allows downloading 
different subsets separately and in the desired image resolution.

The following example downloads the metadata (common for all subsets) and the 'FungiTastic-Mini'
dataset subset with the image resolution of 300px
and saves it in the current directory. The argument 'save_path' is required.
   
```
cd datasets
python download.py --metadata --images --subset "m" --size "300" --save_path "./"  
```

By default, the data is extracted in the save_path folder and copies the file structure from Kaggle.
After extraction, the zip files are deleted.

**Related options:**
- **size**: [_300_, _500_, _720_, _fullsize_]
- **subset**: [full, m, fs]
- **keep_zip**: Do not delete the downloaded zip files (default: False)
- **no_extraction**: Do not extract the downloaded zip files (default: False)
- **rewrite**: Rewrite existing files (default: False)
- **satellite**: Download satellite data.
- **climatic**: Download climatic data.

Downloading segmentation masks, climate data and satellite images:

```
cd datasets
python download.py --masks --climatic --satellite --save_path "./"  
```