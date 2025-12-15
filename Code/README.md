# SDSS-Galaxy-Classification
Sloan Digital Sky Survey (SDSS) galaxy classification using machine learning  

A machine learning project to classify galaxies from the Sloan Digital
Sky Survey (SDSS DR18) into STARFORMING and STARBURST using magnitudes,
fluxes, radii, and redshift.

## Dataset Used: SDSS Galaxy Classification DR18

Features: - Magnitudes: u, g, r, i, z - Fluxes: modelFlux_u …
modelFlux_z - Radii: petroRad_u … petroRad_z - redshift - Target:
subclass

1.  Data Loading df = pd.read_csv(“sdss_100k_galaxy_form_burst.csv”)
    df[“subclass”].replace([“STARFORMING”,“STARBURST”], [0,1],
    inplace=True)

2.  Data Cleaning

-   Checked missing values
-   No major missing data
-   Statistical summary performed

3.  Exploratory Data Analysis (EDA)

-   Univariate: pie chart, boxplots
-   Bivariate: barplots
-   Multivariate: correlation heatmap Insights:
-   Flux–magnitude highly correlated
-   Radii weakly correlated

4.  Outlier Handling Applied IQR method to cap extreme values.

5.  Feature Selection Selected 9 best features: u, modelFlux_i,
    modelFlux_z, petroRad_u, petroRad_g, petroRad_i, petroRad_r,
    petroRad_z, redshift

6.  Class Balancing (SMOTE) Balanced STARFORMING and STARBURST classes.

7.  Train–Test Split train_test_split(…, stratify=y)

8.  Feature Scaling StandardScaler used to normalize values.

9.  Model Training Models tested: Decision Tree, Logistic Regression,
    Random Forest Random Forest chosen as final model.

10. Saving the Model Saved RF.pkl and scaler.pkl using pickle.

11. Testing Manual Inputs Predicted STARFORMING / STARBURST for test
    samples.

## Conclusion
A complete ML pipeline built for SDSS galaxy classification
with Random Forest as final model.
