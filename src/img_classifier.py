import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats
import kagglehub
import os
import matplotlib.colors as mcolors
import matplotlib.collections as mcollections
import datetime as dt
from dateutil.relativedelta import relativedelta
import requests as req
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy.typing as npt

# Import from kaggle and set as the database.
path_vgdf = os.path.abspath(kagglehub.dataset_download("asaniczka/video-game-sales-2024"))
csv_vgdf = os.path.join(path_vgdf, "vgchartz-2024.csv")
datetime_parse = lambda x : dt.strptime(x, '%Y-%m-%d')
vgdf = pd.read_csv(csv_vgdf, parse_dates=['release_date', 'last_update'], date_format=datetime_parse)

# Data Cleaning.
has_missing_img = vgdf["img"].str.contains("default.jpg")
has_missing_last_update = vgdf["last_update"].isna()
has_missing_critic_score = vgdf["critic_score"].isna()

## Replace no dates in last update to release date.
vgdf.fillna({"last_update": vgdf["release_date"]}, inplace=True)
vgdf.dropna(inplace=True)
vgdf['img_link'] = vgdf['img']
vgdf.drop('img', axis=1)

"""
Genre - Data Cleaning
"""
# Data Optimization
vgdf["genre"] = vgdf["genre"].astype("category")
vgdf["release_date"] = vgdf["release_date"].astype('datetime64[ns]')
vgdf["last_update"] = vgdf["last_update"].astype('datetime64[ns]')

"""
"""

IMAGE_DATABASE = 'box_art.csv'
TEMP_FOLDER = 'temp'

IMAGE_COLUMNS = [
  'img_link'
, 'hist_hue'
, 'hist_val'
, 'hist_sat' 
, 'clr_warm'
, 'clr_green'
, 'clr_cool'
, 'clr_purple'
, 'clr_black'
, 'clr_white'
, 'clr_grays'
, 'clr_very_sat'
]

URL_ROOT = "https://www.vgchartz.com"
URL_GAME_BOXART = '/games/boxart/'
PREF_IMG_SIZE = (256, 256)
GEN_BSIZE = 64
HUE_BSIZE = 12

SAT_MASK = 25
VAL_MASK = 25
HUE_CLASSIFICATION_THRESHOLD = (0.2) # 1/4 = 0.25, lowered to 0.20 to account the possibility of balanced all hues
VAL_CLASSIFICATION_THRESHOLD = (0.3) # 1/3 = 0.33, lowered to 0.30 to account the possibility of balanced black, gray, and white

BIN_WIDTH = 256 / GEN_BSIZE  # Bin width for 12-bin histogram
BINS_LOWSAT = int(SAT_MASK / BIN_WIDTH)  # Bins to consider for low saturation
BINS_WHITES = int(VAL_MASK / BIN_WIDTH)  # Bins to consider for low value (black potential)
BINS_BLACKS = int(VAL_MASK / BIN_WIDTH)  # Bins to consider for high value (white potential)

hue_categories = {
  IMAGE_COLUMNS[4] : [0,  1,  2],
  IMAGE_COLUMNS[5] : [3,  4,  5],
  IMAGE_COLUMNS[6] : [6,  7,  8],
  IMAGE_COLUMNS[7] : [9, 10, 11]
}


# [1] Check for file existance of IMAGE_DATABASE
    # Err[IMG_DB] -> create it.
    # Err[Temp]   -> create it.

def image_db_check() -> pd.DataFrame:
  """Checks if the required `temp` directory and image database is located.

  Returns:
      pd.DataFrame: The located or newly created DataFrame file.
  """
  if os.path.isfile(IMAGE_DATABASE):
    print("[ O K ] - Image Database located")
    boxart_df = pd.read_csv(IMAGE_DATABASE)
  else:
    print("[ERROR] - No Image database found, creating....")
    boxart_df = pd.DataFrame(columns=IMAGE_COLUMNS)
    boxart_df.to_csv(IMAGE_DATABASE)
      
  if os.path.isdir(TEMP_FOLDER):
    print("[ O K ] - Temporary Folder located")
    return boxart_df
  else:
    print("[ERROR] Please create a `temp` folder in the same directory of execution (aka root of your repo).")
    exit(-1)

def calcHist(image: cv2.typing.MatLike, channel: int, mask: npt.NDArray[np.uint8], binsize: int, ranges: list[int]) -> cv2.typing.MatLike:
  return cv2.calcHist([image], [channel], mask, [binsize], ranges) 

def normImage(image: cv2.typing.MatLike) -> npt.NDArray:
  return cv2.normalize(image, image, norm_type=cv2.NORM_L1).flatten()

percentage = lambda ratio : True if ratio > VAL_CLASSIFICATION_THRESHOLD else False
percentage_all = lambda ratio1, ratio2, ratio3: True if (ratio1 <= VAL_CLASSIFICATION_THRESHOLD and ratio2 <= VAL_CLASSIFICATION_THRESHOLD and ratio3 <= VAL_CLASSIFICATION_THRESHOLD) else False

boxart_df = image_db_check()

def image_classifier(partial_img_url: str):
  """Classifies images based on the give URL Link

  Args:
      partial_img_url (str): Partial link with the structure of '/games/boxart/`filename`',  and it must be from https://www.vgchartz.com.

  Returns:
      None
  """
  global boxart_df
  # [2] Check NULLs for each row information
    # All Ok  -> Merge.
    # Err     -> Preprocessing.  
  
  if not boxart_df.empty:
    if boxart_df[boxart_df['img_link'] == partial_img_url].isnull().sum().sum() == 0:
      print(f'[ O K ] - {partial_img_url} complete...')
      return boxart_df[boxart_df['img_link'] == partial_img_url];
  
  print(f'[ERROR] - Missing {partial_img_url}, analyzing...')
  
  #### WEBSCRAPING ####
  res = req.get(URL_ROOT + partial_img_url)
  
  if res.status_code != 200:
    print(f"[ERROR]|WEB - Status Code: {res.status_code}")
    return;
  
  file_name = partial_img_url.replace(URL_GAME_BOXART, '')
  img_jpg = os.path.join(TEMP_FOLDER, file_name)
  img_png = img_jpg.replace('.jpg', '.png')
  
  with open(img_jpg, 'wb') as img_file:
    img_file.write(res.content)
    print(f"[PROCS] - Writing: {img_jpg}")

  with Image.open(img_jpg) as im: 
    im.save(img_png, "PNG")
    print(f"[PROCS] - Writing: {img_png}")
    
  print(f"[PROCS] - Removing: {img_jpg}")  
  os.remove(img_jpg)
  
  #### NORMALIZATION ####
  print(f"[PROCS] - Normalizing")
  image = cv2.imread(img_png, cv2.IMREAD_COLOR)
  image = cv2.resize(image, PREF_IMG_SIZE)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
  
  grays_mask = (image[:, :, 1] > SAT_MASK)
  black_mask = (image[:, :, 2] > VAL_MASK)
  white_mask = (image[:, :, 2] < (255 - VAL_MASK))
  mask = grays_mask & black_mask & white_mask 
  
  #### HISTOGRAM EXTRACTION ####
  print(f"[PROCS] - Histogram Extraction")
  hist_hue = normImage(calcHist(image, 0, mask.astype(np.uint8), HUE_BSIZE, [0, 180]))
  hist_sat = normImage(calcHist(image, 1, None                 , GEN_BSIZE, [0, 256]))
  hist_val = normImage(calcHist(image, 2, None                 , GEN_BSIZE, [0, 256]))
  
  #### PALETTE CLASSIFICATION ####
  print(f"[PROCS] - Classify: Colors & Grays")
  hue_classes = {hue_key: False for hue_key in hue_categories.keys()}
  
  for (hue_group, bin_number) in hue_categories.items():
    if np.sum(hist_hue[bin_number]) > HUE_CLASSIFICATION_THRESHOLD:
      hue_classes[hue_group] = True
  
  ratio_lowsat  = np.sum(hist_sat[:BINS_LOWSAT])    # Low saturation (grayscale potential)
  ratio_black   = np.sum(hist_val[:BINS_WHITES])    # Low value (black potential)
  ratio_white   = np.sum(hist_val[-BINS_BLACKS:])   # High value (white potential)
  
  grayscale_classification = {
    "clr_black"     : percentage(ratio_black),
    "clr_white"     : percentage(ratio_white),
    "clr_grays"     : percentage(ratio_lowsat),
    "clr_very_sat"  : percentage_all(ratio_black, ratio_white, ratio_lowsat)
  }
  
  print(f"[PROCS] - Removing: {img_png}")
  os.remove(img_png)
  
  image_features = {
    "img_link": partial_img_url,
    "hist_hue": list(hist_hue),
    "hist_sat": list(hist_sat),
    "hist_val": list(hist_val), 
    **hue_classes, 
    **grayscale_classification
  }
  return image_features

#### MERGING ####

def process_all_images():
  """
  Processes all images in `vgdf` and updates `boxart_df` accordingly.
  - Skips already processed images.
  - Analyzes missing entries and appends them to `boxart_df`.
  """
  global boxart_df

  new_entries = []  # Store new observations before batch update

  for idx, row in boxart_df.iterrows():
    partial_img_url = row['img_link']

    # Skip if image is already fully processed
    if not boxart_df.empty and partial_img_url in boxart_df['img_link'].values:
      existing_row = boxart_df[boxart_df['img_link'] == partial_img_url]
      if existing_row.isnull().sum().sum() == 0:
        print(f"[ O K ] - Skipping {partial_img_url}")
        continue

    print(f"[PROCS] - Analyzing {partial_img_url}...")

    # Call the classifier to extract features
    new_entry = image_classifier(partial_img_url)
    
    if new_entry:
        new_entries.append(new_entry)


  if new_entries:
    new_df = pd.DataFrame(new_entries)
    boxart_df = pd.concat([boxart_df, new_df], ignore_index=True)

    boxart_df.to_csv(IMAGE_DATABASE, index=False)
    print(f"[ O K ] - Processed {len(new_entries)} new images. Database updated.")

  else:
    print("[NOTES] - No new images needed processing.")
        
process_all_images()