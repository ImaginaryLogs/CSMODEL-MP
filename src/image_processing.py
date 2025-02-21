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

print(vgdf.info())
print(vgdf.head(5))

title_dict = {
  'title'         : 'Title',
  'genre'         : 'Videogame Genre',
  'total_sales'   : 'Total Global Sales',
  'na_sales'      : 'North American Sales',
  'jp_sales'      : 'Japanese Sales',
  'pal_sales'     : 'African and European Sales',
  'other_sales'   : 'Other Regional Sales',
  'release_date'  : 'Release Date',
  'last_update'   : 'Last Update'
}

"""
Genre - Exploratory Data Analysis
"""
vgdf_unique_titles =  vgdf.groupby(by=['title', 'genre'], observed=True).agg({
    'total_sales':'sum',
    'na_sales':'sum',
    'jp_sales':'sum',
    'pal_sales':'sum',
    'other_sales':'sum',
    'release_date':'min',
    'last_update':'min'
  }).sort_values('total_sales', ascending=False)
print(vgdf_unique_titles)

# The top 3 most profitable unique titles
genre_count = vgdf_unique_titles[["total_sales"]]\
  .agg({'total_sales':'sum'})\
  .sort_values()
print(genre_count)


# The most profitable genre:
genre_sales = vgdf.groupby("genre", observed=True)["total_sales"].sum().reset_index().sort_values(by='total_sales', ascending=False)
print("Most profitable genre:")
print(genre_sales)

DEFAULT_SIZE =(10, 5)

def plot_groupby(df: pd.DataFrame, category_x: str, count_y: str, hue_z: str):
  df[category_x] = df[category_x].astype(str)
  fig, ax = plt.subplots()
  sns.barplot(data=df, x=category_x, y=count_y, hue=hue_z)
  
  plt.title(f'Total Sales per Genre')
  plt.xlabel(f'{title_dict[category_x]}')
  ax.set_xticklabels(df[category_x], rotation=45, ha='right', rotation_mode='anchor')
  
  plt.ylabel(f'{title_dict[count_y]}')
  
  plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
  plt.grid(linestyle=':', alpha=0.5)
  plt.show()

# plot_groupby(genre_sales, 'genre', 'total_sales', 'genre')

# The Top 3 Best Unique Videogames per Genre Category
genre_popular = vgdf_unique_titles.sort_values("total_sales")
genre_popular = genre_popular.groupby(['genre'], observed=True)\
  ['total_sales'].nlargest(3)
  
  
# Release Date timeline -> Genre
# Sonic                 ->
# Shooters              ->




"""
Image processing - Web Scrapping
"""
media_path = 'media'
URL = "https://www.vgchartz.com"
URL_GAME_ART_LOCATION = '/games/boxart/'
PREF_IMG_SIZE = (128, 128)
GEN_BSIZE = 12
HUE_BSIZE = 12 # Fixed, do not touch

DOWNLOAD_ALL = False
DOWNLOAD_LIMIT = 50

def select_images():
  return vgdf if DOWNLOAD_ALL else vgdf.head(DOWNLOAD_LIMIT)

def image_location(img_path_partial: str) -> tuple[str, str, str]:
  file_name = img_path_partial.replace(URL_GAME_ART_LOCATION, '')
  img_jpg = os.path.join(media_path, file_name)
  img_png = img_jpg.replace('.jpg', '.png')
  
  return (file_name, img_jpg, img_png)

def image_webscrape(img_path_partial: str) -> str|None:
  """Retrieves the corresponding image from the video game database https://www.vgchartz.com.

  Args:
      img_path_partial (str): the partial link to retrieve the image.
  """
  (file_name, img_jpg, img_png) = image_location(img_path_partial)

  if os.path.isdir(media_path) and os.path.isfile(img_png):
    print(f"[EXISTS]: {img_png}")
    return img_png
  
  if not (os.path.isdir(media_path)):
    os.mkdir(media_path)
  
  res = req.get(URL + img_path_partial)
  
  if res.status_code != 200:
    print(f"Error. Status Code: {res.status_code}")
    return None

  with open(img_jpg, 'wb') as img_file:
    img_file.write(res.content)
    print(f"[WRITE]: {file_name}")
    im = Image.open(img_jpg)
    im.save(img_png, "PNG")
    return img_png
    
  os.remove(img_jpg)

vgdf['img_png_path'] = select_images()['img_link'].apply(image_webscrape)

    
"""
Image processing - Data Cleaning
"""

calc_hist = lambda image, channel, ranges : cv2.calcHist([image], [channel], None, [GEN_BSIZE], ranges) 
normalize = lambda x: cv2.normalize(x, x, norm_type=cv2.NORM_L1).flatten()

def hsv_preprocess_extract(img_path):
  image = cv2.imread(img_path, cv2.IMREAD_COLOR)
  
  if image is None:
      return None, None
  
  # Resize the image to small yet reasonable size
  image = cv2.resize(image, PREF_IMG_SIZE)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
  # Data Cleaning with a Filter
  # 0 - hue, 1 - sat, 2 - val
  mask = image[:, :, 1] & image[:, :, 2] > 30 
  return image, mask

def get_hist_hue(img_path: str):
  img, mask = hsv_preprocess_extract(img_path)
  if img is None:
    print(f"[ERROR]: No Image Located {img_path}")
    return None
  
  hue_hist = normalize(cv2.calcHist([img], [0], mask.astype(np.uint8), [HUE_BSIZE], [0, 180]))
  return hue_hist

def get_hist_sat(img_path: str):
  img, mask = hsv_preprocess_extract(img_path)
  
  if img is None:
    return None
  
  sat_hist = (normalize(cv2.calcHist([img], [1], mask.astype(np.uint8), [GEN_BSIZE], [0, 256])))
  return sat_hist

def get_hist_val(img_path: str):
  img, mask = hsv_preprocess_extract(img_path)
  
  if img is None:
    return None
  
  val_hist = (normalize(cv2.calcHist([img], [2], mask.astype(np.uint8), [GEN_BSIZE], [0, 256])))
  return val_hist

apply_all           = lambda apply, h, s, v : (apply(h), apply(s), apply(v))
channel_extraction  = lambda channelfunc : select_images()["img_png_path"].apply(lambda x: channelfunc(x))
channel_mean        = lambda name : select_images()[name].mean()

vgdf["hist_hue"], vgdf["hist_sat"], vgdf["hist_val"] = apply_all(channel_extraction, get_hist_hue, get_hist_sat, get_hist_val)

def plot_hsv_hist():
  h, s, v = apply_all(channel_mean, 'hist_hue', 'hist_sat', 'hist_val')
  fig, axes = plt.subplots(3, 1, figsize=(16, 8))

  adjusted_pos = lambda cap, bin_step_size : (
    np.linspace(0, cap, bin_step_size, endpoint=False) + (cap / bin_step_size) / 2
  )

  hue_x_pos = adjusted_pos(180, HUE_BSIZE)
  sat_x_pos = adjusted_pos(256, GEN_BSIZE)
  val_x_pos = adjusted_pos(256, GEN_BSIZE)

  hue_colors = np.array(
    [[[h, 255, 255]] for h in hue_x_pos.astype(np.uint8)],  # approximate linspace float to int
    dtype=np.uint8                                          # As an integer type
  )

  hue_colors = (
    cv2.cvtColor(hue_colors, cv2.COLOR_HSV2RGB)             # To RGB
    .reshape(-1, 3) / 255                                   # Flatten down to three (n, 3)
  )

  def hist_custom(ax, name, top_limit=256):
    ax.set_title(name)
    ax.set_xlim([0, top_limit])
    ax.set_xticks(np.linspace(0, top_limit, 10, dtype=int))
  
  axes[0].bar (hue_x_pos, h, color=hue_colors, width=180/HUE_BSIZE)
  hist_custom(axes[0], "Hue", top_limit=180)
  axes[1].plot(sat_x_pos, s, color='green')
  hist_custom(axes[1], "Saturation")
  axes[2].plot(val_x_pos, v, color='black')
  hist_custom(axes[2], "Value")

  plt.show()

plot_hsv_hist()

def get_hist_hue(img_path: str):
  img, mask = hsv_preprocess_extract(img_path)
  
  if img is None:
    return None
  
  hue_hist = (normalize(cv2.calcHist([img], [0], mask.astype(np.uint8), [HUE_BSIZE], [0, 180])))
  return hue_hist

hue_categories = {
  "has_pal_warm"            : [0, 1, 2],
  "has_pal_forest_neutral"  : [3, 4, 5],
  "has_pal_cool"            : [6, 7, 8],
  "has_pal_pastel_neutral"  : [9, 10, 11]
}

HUE_CLASSIFICATION_THRESHOLD = 0.20

def categorize_hue(hue_history):
  one_hot = {hue_key: np.uint8(0) for hue_key in hue_categories.keys()}
  print(f"hue_hist: {hue_history}")
  for (hue_group, bin_number) in hue_categories.items():
    if np.sum(hue_history[bin_number]) > HUE_CLASSIFICATION_THRESHOLD:
      one_hot[hue_group] = np.uint8(1)
  
  return one_hot

vgdf["hist_hue"].dropna()

vgdf["color_categories"] = select_images()["hist_hue"].apply(categorize_hue)


# Expand into one-hot encoded columns
color_df = select_images()["color_categories"].apply(pd.Series, dtype=np.uint8)
vgdf = pd.concat([vgdf, color_df], axis=1).drop(columns=["color_categories"])

print(vgdf.info())

print(vgdf["has_pal_warm"].value_counts())
print(vgdf["has_pal_forest_neutral"].value_counts())
print(vgdf["has_pal_cool"].value_counts())
print(vgdf["has_pal_pastel_neutral"].value_counts())