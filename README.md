# 🏎️ F1 Championship Analytics

An end-to-end data science project that explores Formula 1 race data – from web scraping and data visualization to machine learning and game theory. This project uncovers performance patterns, driver comparisons, and winning probabilities using advanced analytics techniques.

---

## 📌 Project Flow

1. **🔍 Web Scraping**
   - F1 race data is scraped from an official source using BeautifulSoup.
   - Tables (driver standings, race results, etc.) are extracted and stored as structured CSV files.

2. **📊 Data Analysis**
   - **Top 3 driver comparison** across multiple races based on key performance indicators.
   - **In-depth comparison** between a selected driver and the season champion.

3. **📈 Visualization with Power BI**
   - Dashboard includes race-by-race breakdown, podium finishes, constructor stats, and driver profiles.
   - Filterable by driver, season, and location.

4. **🌲 Machine Learning – Feature Importance**
   - A **Random Forest model** is trained to determine which factors (e.g., qualifying position, lap time, tire strategy) most influence race victories.
   - Feature importance plots reveal key contributors to performance.

5. **♟️ Game Theory Simulation**
   - Winning probabilities for the championship are simulated using **Game Theory concepts**.
   - Models strategic decision-making across races and updates winning chances dynamically based on race outcomes.

---

## 🛠️ Tech Stack

- **Python** – Data scraping, processing, and ML modeling  
- **BeautifulSoup** – HTML parsing and table extraction  
- **Pandas & NumPy** – Data manipulation and analysis  
- **Scikit-learn** – Random Forest modeling  
- **Power BI** – Interactive data visualization  
- **Game Theory** – Strategic modeling of race outcomes

---

## 📁 Project Structure

```bash
├── data.py               # Flask app for scraping and Excel export
├── notebooks/            # Jupyter Notebooks for EDA and modeling
├── PowerBI/              # Power BI dashboard files
└── README.md             # Project overview and usage guide
## 📷 Sample Dashboard (Power BI)
![image](https://github.com/user-attachments/assets/7c3567b9-f6ed-4af6-a3cc-aaef8bb44ec8)
![image](https://github.com/user-attachments/assets/fe9029cc-c4dc-4b3c-94ae-282186d1cf19)
![image](https://github.com/user-attachments/assets/b095de21-2347-4bb5-ba1b-8851db828fdf)



