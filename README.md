# **News and Load: A Quantitative Exploration of NLP for Forecasting Electricity Demand**

This repository contains the implementation of the research paper **"News and Load: A Quantitative Exploration of Natural Language Processing Applications for Forecasting Day-ahead Electricity System Demand."** The study explores the use of textual data (news articles) in improving electricity demand forecasting through **Natural Language Processing (NLP)** techniques.

---

## **Overview**
Accurate electricity demand forecasting is crucial for grid management and energy planning. Traditional methods rely heavily on meteorological and temporal data, but **social events in news** can also impact energy consumption. This project investigates the potential of using **news articles** as an additional source of information for day-ahead load forecasting.

We use **Natural Language Processing (NLP)** to extract meaningful features from news data and integrate them into machine learning models, improving forecast accuracy.

---

## **Features**
✅ Extracts structured features from unstructured text (news articles).  
✅ Implements **sentiment analysis, topic modeling, word embeddings, and frequency-based methods**.  
✅ Applies **Granger causality tests** to filter relevant textual features.  
✅ Uses **ExtraTrees Regression** for forecasting.  
✅ Provides **local and global explainability** via **LIME (Local Interpretable Model-agnostic Explanations)** and **causality tests**.  
✅ Compares different **word embedding techniques**: TF-IDF, Word2Vec, and GloVe.  
✅ Evaluates **regional forecasting** (Northern Ireland) and **short-term forecasting** performance.

---

## **Dataset Sources**
This project uses **electricity demand** and **news datasets**:

1. **Electricity Demand Data** (UK & Northern Ireland)  
   📌 [ENTSO-E Transparency Platform](https://transparency.entsoe.eu)  

2. **News Data (BBC News Articles)**  
   - 📌 [BBC News Crawler](https://github.com/LuChang-CS/news-crawler)  
   - 📌 [BBC News Archive](https://dracos.co.uk/made/bbc-news-archive/archive.php)  
   - 📌 [BBC Downloaded News](https://1drv.ms/f/s!AuGdRIQWT-F7hT3FQ57nEEk4hrLt?e=rnOmBB)  


## **Experiments & Results**
📌 **Text-based forecasting improves accuracy!**  
📌 **Best features**:  
   - Word frequencies in **news titles**  
   - Sentiment scores  
   - GloVe word embeddings  
📌 **Causal relationships** between specific news topics and electricity demand.

📊 **Key Findings:**
- **Politics (e.g., Northern Ireland elections)** affect electricity consumption.  
- **Pandemic-related news** correlates with lower demand.  
- **Transportation and military events** show predictive power.  
- **Short-term forecasts (few hours ahead) improve with textual features.**  

📌 **Detailed results** can be found in the paper.

---

## **Citation**
If you use this project in your research, please cite:

```bibtex
@article{bai2024newsload,
  author    = {Yun Bai, Simon Camal, Andrea Michiorri},
  title     = {News and Load: A Quantitative Exploration of Natural Language Processing Applications for Forecasting Day-ahead Electricity System Demand},
  journal   = {IEEE Transactions on Smart Grid},
  year      = {2024}
}
```

---

## **Acknowledgements**
This research was conducted at **MINES Paris - PSL University, Centre for Processes, Renewable Energies, and Energy Systems (PERSEE)** and supported by the **China Scholarship Council (CSC Nos. 202106020064).**

---

## **License**
📜 MIT License – Feel free to use and modify!

---
