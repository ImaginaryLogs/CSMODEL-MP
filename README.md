<table>
  <tr>
   <td colspan="2" >
    <h1><strong>The Marketability of
      
Video Game Sales in 2024</strong></h1>
    A Statistical Modelling and Simulation (CSMODEL) Project.
   </td>
  </tr>
  <tr>
   <td>
     
 Group 4  
 <strong>Ancheta, Liam Micheal Alain Barredo</strong>  
 De La Salle University>  
 <a href="mailto:liam_michael_ancheta@dlsu.edu.ph">liam_michael_ancheta@dlsu.edu.ph</a>  
       
 <strong>Campo, Roan Cedric Vinarao</strong>  
 De La Salle University  
 <a href="mailto:roan_campo@dlsu.edu.ph">roan_campo@dlsu.edu.ph</a>  
       
 <strong>Domingo, Angela Sophia De Leon</strong>  
 De La Salle University>  
 <a href="mailto:angela_domingo@dlsu.edu.ph">angela_domingo@dlsu.edu.ph</a> 
 
 <strong>Sanchez, Chloe Jeanine Esguerra</strong>  
 De La Salle University>  
 <a href="mailto:chloe_sanchez@dlsu.edu.ph">chloe_sanchez@dlsu.edu.ph</a>   

   </td>
   <td>
       <strong>Abstract</strong>
      <p>
      <strong>Keywords</strong>
      <p>
      Video Game, Business, Data Analytics 
   </td>
  </tr>
</table>



# 


# Introduction

Video Games are any interactive activity that uses a computational device, typically an electronic, to allow users to gain pleasure, and many people are engaged in this pastime. The market is high in demand for new titles of this kind of interactive entertainment, and many companies have produced titles that have varying amounts of success. Moreover, the Interactive Entertainment Industry is one of the fastest growing industries in the late 20th and early 21st Century. With the rise of such industry, this research explores what factors influence the success of these products and its implications.


## About the Dataset

This dataset was sourced from Kaggle Datasets and was authored by Asanickza and Brannen [2]. It served as a continuation of two previous datasets of a similar theme entitled *2019 Video Game Sales* and *2020 Video Game Sales* [3, 4]. The dataset contains general attributes of a video game, such as its publisher, developer, console, and sales across different regions as well as globally. The version of the dataset being used for analysis would be the February 2025 version of the dataset.  The data set was web scraped from Video Game Charts – a business intelligence and research firm that analyzes the video games market [2, 3].


## Collection Process and its Implications

The date of the dataset implies the inclusion of most, if not all, of the games before 2024. This also implies that several consoles either outdated or not in use for modern games have also been included in the dataset. Any popular or high-selling games made in the first month of the year or beyond will not be included. The data acquired not only by this dataset but also by the datasets referenced by the current one was 


## Structure of Dataset of the File

The dataset contains around 64K observations detailing the information, perceived critic score, and the sale metrics of each game and their released platform. Some games have repeat entries due to the platforms it was released on and differing publishers and developers for the same franchise. All relevant data was already consolidated in one file, and some variables included in previous versions of the dataset were removed due to an excess amount of null data.

The dataset has 14 variables, and it has the following variables in no particular order: (1-5) regional and total sales of the game, (6) the platforms where the game was released, (7) the box art, (8) the title, (9) the main genre it belongs to, (10) the release date of the game, (11 - 12) the developer and the publisher responsible for it, (13) critic scores, (14) and the date since it was last updated. 

The variables related to sales were preprocessed relative to the millions, so the values are expected not to be completely raw. In addition to this, the sales included in the dataset have regional breakdowns; this consists of North American, European and African, Japanese, and other unspecified region sales. The platforms where the games are released are explicitly separated from each other; metrics of the same game released on different platforms will have different results per platform.


## Exploratory Data Analysis



1. Game Franchise / Series
    * Feature engineering is required to consolidate games under a similar series/name
2. Location and Platform/Console* (by platform/console)
    * Platform is viable but requires more cleaning with the various amount of consoles/devices available 
        * Some consoles mentioned are operating systems
    * Location is difficult as it uses a different metric to compare influence/performance
        * Regions are heavily consolidated
3. Developer and Publisher*
4. Genre
5. Release Date
    * Feature engineering to sort it by the four seasons (winter, summer, autumn, spring) (maybe games perform better if released in certain seasons)


## Research Questions

“What factors have the greatest influence on Video Game Sales?”


# Data Cleaning


# Exploratory Data Analysis



1. Game Franchise vs Global Sales
2. Console vs Global Sales
3. Developer vs Global
4. Genre vs Global Sales
5. *Color Analysis vs Global Sales
6. Critic Score vs Global Sales
7. Critic Score/Sales vs Release Date


# Research Questions

What factors significantly affect the marketability of a video game title?


# References



1. Henry E Lowood. 1998. Electronic game | Definition, History, Systems, & Facts. *Encyclopedia Britannica*. Retrieved February 10, 2025 from https://www.britannica.com/topic/electronic-game
2. asaniczka. 2024. Video Game Sales 2024. *Kaggle.com*. Retrieved February 10, 2025 from [https://www.kaggle.com/datasets/asaniczka/video-game-sales-2024](https://www.kaggle.com/datasets/asaniczka/video-game-sales-2024)
3. Abdulshaheed Alqunber. 2019. Video Games Sales 2019. *Kaggle.com*. Retrieved February 10, 2025 from [https://www.kaggle.com/datasets/ashaheedq/video-games-sales-2019](https://www.kaggle.com/datasets/ashaheedq/video-games-sales-2019) 
4. Bayne Brannen. 2020. Video Game Sales 2020. *Kaggle.com*. Retrieved February 10, 2025 from [https://www.kaggle.com/datasets/baynebrannen/video-game-sales-2020](https://www.kaggle.com/datasets/baynebrannen/video-game-sales-2020) 
5. Abdulshaheed Alqunber. 2025. vgchartzScrape: a web scraping project for data capture of vgchartz. GitHub. Retrieved February 10, 2025 from [https://github.com/ashaheedq/vgchartzScrape](https://github.com/ashaheedq/vgchartzScrape) 
6. ‌Video Game Charts. 2025. About VGChartz. *VGChartz*. Retrieved February 10, 2025 from [https://www.vgchartz.com/about.php](https://www.vgchartz.com/about.php) 

‌
