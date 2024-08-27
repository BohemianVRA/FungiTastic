# About FungiTastic

## Download instructions

There are two options to download the dataset:

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
The preferred approach is to use our script to get the data that allows downloading different subsets separately and in desired resolution.
   
```
cd datasets
python download.py --subset "m" --size "300" --save_path "./"  
```

**Dataset related options:**
- **size**: _300_, _500_, _720_, _fullsize_
- **subset**: full, m, fs, dna