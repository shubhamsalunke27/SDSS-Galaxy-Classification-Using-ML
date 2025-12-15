
# SDSS Galaxy Classification Using Machine Learning

## Project Overview
This project focuses on the **automated classification of galaxies** from the Sloan Digital Sky Survey (SDSS) dataset using **Machine Learning techniques**. The main goal is to classify galaxies into categories such as **Elliptical, Spiral, and Irregular** based on photometric and spectral features.  

By automating galaxy classification, the system reduces manual effort for astronomers and provides a scalable and accurate solution for large datasets.

---

## Project Features
- Automated galaxy classification using Machine Learning models.  
- Comparative analysis of **Logistic Regression, Random Forest, and SVM** algorithms.  
- Preprocessing and exploratory data analysis (EDA) to enhance model performance.  
- Model deployment with a simple user interface for predictions on new data.  
- Generation of evaluation metrics including **accuracy, precision, recall, F1-score, and confusion matrix**.

---

## Project Structure

```

SDSS-Galaxy-Classification-Using-ML/
│
├─ Code/
│   ├─ data_preprocessing.ipynb
│   ├─ model_training.ipynb
│   ├─ model_deployment.ipynb
│   └─ utils.py (optional)
│
├─ Data/
│   └─ SDSS_dataset.csv (or provide link if dataset is too large)
│
├─ Output/
│   ├─ trained_model.pkl
│   ├─ performance_metrics.csv
│   └─ screenshots/ (EDA plots, confusion matrix, graphs)
│
├─ Demo_Video/
│   └─ Galaxy_Classification_Demo.mp4
│
├─ Report/
│   └─ Internship_Project_Documentation.pdf
│
├─ requirements.txt
└─ README.md

````

---

## How to Run

1. **Clone the repository**:
```bash
git clone https://github.com/YourUsername/SDSS-Galaxy-Classification-Using-ML.git
````

2. **Install required libraries**:

```bash
pip install -r requirements.txt
```

3. **Run Jupyter Notebooks in order**:

   * `data_preprocessing.ipynb` – Data cleaning, normalization, and preparation.
   * `model_training.ipynb` – Train models, tune hyperparameters, evaluate performance.
   * `model_deployment.ipynb` – Load trained model and make predictions.

4. **Use trained model** (`trained_model.pkl`) to classify new galaxy data.

---

## Key Libraries & Tools

* **Python** – Main programming language
* **NumPy & Pandas** – Data manipulation and analysis
* **Matplotlib & Seaborn** – Data visualization
* **Scikit-learn** – Machine learning algorithms and evaluation
* **Jupyter Notebook** – Development and demonstration
* **GitHub** – Version control

---

## Results & Performance

* **Random Forest Classifier** achieved **92% accuracy** on the test dataset.
* High precision, recall, and F1-score across all galaxy classes.
* Confusion matrices and plots validate the model’s reliability.

---

## Demo Video

🎥 [Galaxy Classification Demo Video]([Demo_Video/Galaxy_Classification_Demo.mp4](https://docs.google.com/videos/d/1EWwz7X0Z0VLjrVEmYQk-eqAslxr1zMl5k1bNZFS-r-0/play))

---

## Author

**Shubham Salunke**
Computer Engineering Student | Machine Learning Enthusiast

---

## GitHub Repository Link

[SDSS-Galaxy-Classification-Using-ML]([https://github.com/YourUsername/SDSS-Galaxy-Classification-Using-ML](https://github.com/shubhamsalunke27/SDSS-Galaxy-Classification-Using-ML))

---

## References

1. Sloan Digital Sky Survey (SDSS) – [https://www.sdss.org](https://www.sdss.org)
2. Scikit-learn Documentation – [https://scikit-learn.org](https://scikit-learn.org)
3. NumPy Documentation – [https://numpy.org](https://numpy.org)
4. Pandas Documentation – [https://pandas.pydata.org](https://pandas.pydata.org)
5. Matplotlib Documentation – [https://matplotlib.org](https://matplotlib.org)
6. Seaborn Documentation – [https://seaborn.pydata.org](https://seaborn.pydata.org)
7. Research papers on galaxy classification using machine learning

---




