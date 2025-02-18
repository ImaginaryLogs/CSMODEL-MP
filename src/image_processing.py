import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats
import kagglehub
import os
from datetime import datetime
import requests as req
from PIL import Image

# Import from kaggle and set as the database.
path_vgdf = os.path.abspath(kagglehub.dataset_download("asaniczka/video-game-sales-2024"))
csv_vgdf = os.path.join(path_vgdf, "vgchartz-2024.csv")
datetime_parse = lambda x : datetime.strptime(x, '%Y-%m-%d')
vgdf = pd.read_csv(csv_vgdf, parse_dates=['release_date', 'last_update'], date_format=datetime_parse)

# Data Cleaning.
has_missing_img = vgdf["img"].str.contains("default.jpg")
has_missing_last_update = vgdf["last_update"].isna()
has_missing_critic_score = vgdf["critic_score"].isna()

## Replace no dates in last update to release date.
vgdf.fillna({"last_update": vgdf["release_date"]}, inplace=True)
vgdf.dropna(inplace=True)


"""
Genre - Data Cleaning
"""
# Data Optimization
vgdf["genre"] = vgdf["genre"].astype("category")
vgdf["release_date"] = vgdf["release_date"].astype('datetime64[ns]')
vgdf["last_update"] = vgdf["last_update"].astype('datetime64[ns]')

print(vgdf.info())
print(vgdf.head(5))

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
genre_sales = vgdf.groupby("genre", observed=True)["total_sales"].sum().sort_values(ascending=False)

print("Most profitable genre:")
print(genre_sales)


# The Top 3 Best Unique Videogames per Genre Category
genre_popular = vgdf_unique_titles.sort_values("total_sales").groupby(['genre'], observed=True)\
  ['total_sales'].nlargest(3)
print(genre_popular)

# Evolution of Genres through the decades.

"""
Image processing - Web Scrapping
"""
media_path = 'media'
URL = "https://www.vgchartz.com"

def retreive_image(img_path_partial: str) -> None:
  """Retrieves the corresponding image from the video game database https://www.vgchartz.com.

  Args:
      img_path_partial (str): the partial link to retrieve the image.
  """
  file_name = img_path_partial.replace('/games/boxart/', '')
  img_jpg = os.path.join(media_path, file_name)
  img_png = img_jpg.replace('.jpg', '.png')

  if os.path.isdir(media_path) and os.path.isfile(img_png):
    print(f"File {img_png} already exists.")
    return
  
  if not (os.path.isdir(media_path)):
    os.mkdir(media_path)
  
  res = req.get(URL + img_path_partial)
  
  if res.status_code != 200:
    print(f"Error. Status Code: {res.status_code}")
    return
    
  with open(img_jpg, 'wb') as img_file:
    img_file.write(res.content)
    print(f"Downloaded: {file_name}")
    im = Image.open(img_jpg)
    im.save(img_png, "PNG")
    
  os.remove(img_jpg)
    
vgdf['img'].head(10).apply(retreive_image)
    

"""
Image processing - Data Cleaning
"""

"""
Image processing - Visualization
"""