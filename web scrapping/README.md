# Flipkart 5G Mobile Analytics Pipeline

An automated data harvesting, safe balancing, and competitor analysis pipeline targeting 5G mobile devices priced under ₹50,000 on Flipkart. This project maps pricing structures, isolates hardware pattern specifications, and extracts consumer ratings.

---

## Project Overview

This repository contains the complete implementation of a web-scraping pipeline built with **Python, Requests, BeautifulSoup, and Pandas**. It addresses a critical data alignment challenge commonly encountered in web scraping: handling missing feature fields (e.g., unrated products or items without price listings) without shifting columns or corrupting data rows.

---

## File Structure

* **`webscrap.ipynb`**: The primary Jupyter Notebook containing the ingestion setup, automated multi-page scraping, spec engineering, and visual analysis.
* **`Flipkart_5G_Mobile_Under_50000.csv`**: The raw scraped dataset containing exactly 216 complete records.
* **`Flipkart_5G_Mobile_Details.csv`**: The structured dataset containing parsed and cleaned specs:
  * **Brand**: Phone manufacturer (OnePlus, Samsung, OPPO, etc.)
  * **Clean_Price_INR**: Normalized pricing integers
  * **RAM_GB**: Isolated RAM memory capacity
  * **ROM_GB**: Isolated storage capacity
  * **Battery_mAh**: Battery capacity
  * **Display_Inches**: Screen sizes mapped to inches
  * **Processor**: Silicon unit (Snapdragon, Dimensity, Exynos, Tensor)
* **`competitor_price_distribution.png`**: Boxplot showing competitor pricing margins.
* **`brand_ratings.png`**: Bar chart depicting customer ratings by brand.
* **`ram_rom_price_matrix.png`**: Heatmap illustrating pricing against memory configurations.

---

## Scraping & Alignment Solution

### The Challenge
A naive global scraping of tags like names, prices, and reviews page-wide creates lists of unequal sizes because some products are unrated or lack prices. When joined, the rows shift vertically, pairing product names with incorrect prices and review scores from subsequent listings.

### The Solution
The pipeline queries target node metrics **per product card container** (`div.lvJbLV.col-12-12` containing `div.RG5Slk`). If a sub-element like a rating score or price is missing, it is appended as `None` (`NaN` in Pandas) for that specific product, guaranteeing perfect matrix alignment and preserving data integrity.

---

## Setup & Execution

### 1. Installation
Install the required dependencies:
```bash
pip install beautifulsoup4 lxml pandas requests matplotlib seaborn
```

### 2. Running the Pipeline
Open the notebook in Jupyter:
```bash
jupyter notebook webscrap.ipynb
```
Run all cells to execute the scraping loop, output the aligned CSV datasets, and generate the analysis figures.

---

## Key Analytics Summary

### Brand Pricing & Market Share
* **OnePlus** and **Samsung** hold the largest volume under the ₹50,000 threshold.
* **Google** sits strictly in the premium tier, averaging ~₹46.2k, while **Samsung** covers the entire spectrum (from ₹12.7k to ₹50k).

### Memory Pricing deltas
* **8 GB RAM** and **12 GB RAM** models dominate the market.
* Upgrading from 8 GB to 12 GB RAM only incurs an average price increase of **~₹2,150**, indicating high competitor specification competition.

### Consumer Ratings
* **vivo** leads customer satisfaction with an average rating of **4.48 / 5.0**, followed closely by **OPPO** (**4.43 / 5.0**).