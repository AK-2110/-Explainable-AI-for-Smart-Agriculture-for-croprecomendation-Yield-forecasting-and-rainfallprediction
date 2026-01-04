Abstract 
Agriculture Climate change, resource scarcity, and population growth are posing previously unheard-of challenges to agriculture, making sustainable and precision farming methods essential to guaranteeing world food security. Applications like rainfall forecasting, crop yield forecasting, and crop recommendation have been made possible by recent developments in machine learning (ML) and artificial intelligence (AI). But for farmers, agronomists, and policymakers, the opaque nature of the majority of AI models frequently restricts openness, interpretability, and confidence. We suggest an Explainable Smart Agriculture Framework to solve this problem by combining explainable AI (XAI) methods with cutting-edge predictive models.In order to capture semantic and temporal dependencies in agricultural datasets, the suggested framework uses XLNet-based feature extraction and the Enhanced Barnacle Mating Optimization (EBMO) algorithm for robust feature selection. Models for prediction, such as Support Vector Machines (SVM), Long Short-Term Memory (LSTM), Gated Recurrent Unit (GRU), and Transformer-based architectures are trained on various datasets from the National Climate Atlas (NCA), the National Atlas & Thematic Mapping Organisation (NATMO), and the International Crops Research Institute for the Semi-Arid Tropics (ICRISAT). To improve transparency, SHAP (SHapley Additive explanations) and LIME (Local Interpretable Model-agnostic Explanations) are used for model interpretation. They emphasize the importance of features such as rainfall, soil moisture, and photosynthetically active radiation (PAR). Experimental evaluations show that the proposed framework not only increases predictive accuracy by up to 15% compared to baseline models but also offers actionable insights into feature contributions. This helps connect high-performance AI systems with practical agricultural decision-making. By combining predictability with interpretability, this framework promotes the adoption of trustworthy AI in agriculture. It allows policymakers, researchers, and farmers to make more informed, clear, and sustainable decisions.   

Keywords:  Smart Agriculture, Explainable AI (XAI), Crop Recommendation, Yield Forecasting, Rainfall Prediction, Transformer Models, XLNet, Enhanced Barnacle Mating Optimization (EBMO), SHAP, LIME, Precision Farming, Sustainable Agriculture 

I. Introduction 
The foundation of the world's food supply is agriculture, but it faces previously unheard-of difficulties like resource shortages, soil erosion, and climate variability. Conventional farming practices frequently make judgments based on experience, which makes them inflexible in the face of quickly shifting conditions. Data-driven decision-making is made possible by smart agriculture, which is sustainability requires the use of applications like rainfall prediction, yield forecasting, and crop recommendation[3], [4]. 
The majority of ML/DL models function as opaque black boxes, notwithstanding AI's efficacy. Policymakers, agronomists, and farmers want interpretable insights that clarify the reasoning behind a model's predictions. Explainable AI (XAI), which offers both accuracy and interpretability, becomes essential in this situation[1],[2]. Our research combines cutting-edge ML methods with XAI to address: 
 
Figure:1 model diagram 
which crops are best suited to a particular soil type and climate. 
The amount of yield that a farmer can anticipate in the given circumstances. 
The amount and timing of rainfall will help with resource planning and irrigation. 
 
This paper's contributions include: 
 
creation of a crop recommendation system with XLNet + SVM that is enabled by XAI. 
A yield forecasting model that uses time-series analysis and XLNet + LSTM/GRU. 
Transformer architectures are used in a rainfall forecast model. 
Combining SHAP and LIME to provide clear explanations. 
extensive tests using comparative evaluation on datasets from multiple sources. 
 
                                      Figure:2 Block Diagram 

II. Related Work: 
The "Explainable AI for Crop Recommendation, Yield Forecasting, and Rainfall Prediction in Smart Agriculture" is critical for setting up the research through the investigation of studies examining Explainable AI (XAI), crop recommendations, forecasting environmental factors such as rainfall and yield, and the limitations of those prior studies and the novelty of the proposed framework. The section is divided into subsections, examining old versus modern approaches to crop recommendations (e.g. Random Forests, SVM), yield forecasts (CNN, LSTMs), and rainfall prediction (ARIMA to transformer models), while acknowledging the predictive abilities but lack of transparency of traditional models, which contributes to a lack of trust for farmers and policy makers [3], [4], [6], [7], [9]. Many studies by Dwivedi et al. (2024) and El-Kenawy et al. (2024), refer to black-box model opacity, and since there are limited integrated framework models based on XAI that evaluate multiple agricultural tasks and yield forecasts [8], [11],[20]. This section delineates the research gap--a comprehensive, interpretable AI framework--and situates the paper's contribution of using Enhanced Barnacle Mating Optimization (EBMO), XLNet, and XAI methods such as SHAP and LIME as the contextualization of the research gap solves this issue while improving accuracy by 15% while being reliable and transparent [16], [17], [19]. This section is focused on the conference audience where technical nuances are a blend of being too technical and too mundane, with consideration for justifying the research of peer review and practicality of the methods demonstrated to the farmers. 
 
II. Associated Research 
 
a. Crop  Recommendation 
 
Traditional crop recommendation models use advanced machine learning methods like Random Forests, Support Vector Machines (SVM) and Gradient Boosting to analyze soil and weather data and recommend appropriate crops [3], [4]. FARMIT, that combines IoT and Random Forest models, is an operational example of a crop recommendation system that performs well using machine learning and data [5], [14]. Nevertheless, these "black-box" methods often do not elucidate why certain crops are recommended, which understandably leads to a lack of confidence in adoption by farmers and agricultural stakeholders. Explainable AI (XAI) addresses this need for farmers to have clarity of model output and prediction. There are approaches such as LIME that provide "local" explanations pertaining to specific predictions, as well as SHAP, that describes individual and overall influence of predictors [1], [16]. Yneos and Buchkowski (2023) are early leaders in simpler and inherently interpretable models such as decision trees to make prediction, and this should complement other methodologies such as counterfactual explanations. The move is to establishing a hybrid system of XAI, by incorporating non-black-box machine learning prediction while mapping a clear and interpretable post hoc explanation. The ultimate goal for these systems is simple, to provide an explanation that is both trust inducing for the farmer/audience who understand wellness (as disinterested bystanders), while being beneficial for explaining and prompting good or better farming practices optimized for the local domain. 
 
Figure: 3- model diagram of crop recommendation 

b. Forecasting Yield 
 
Accurately forecasting crop yields is an essential part of precision agriculture that facilitates better crop planning, resource optimization and, ultimately, food security on a global scale. New advances in Artificial Intelligence (AI) have led to the increased use of deep learning models including Convolutional Neural Networks (CNNs), Long Short-Term Memory networks (LSTMs) and Gated Recurrent Units for crop yield prediction purposes[6],[7],[11]. These models effectively capture complex spatial and temporal patterns through various agricultural datasets (e.g. weather datasets, satellite imagery, and soil characteristics). Hybrid models which incorporate Machine Learning (ML) and Deep Learning (DL) models, may provide key advantages in forecast accuracy, as noted by the research conducted by El-Kenawy et al.[22]. Researchers report that potato yield forecasts were based on hybrid ML and DL models showed a significant increase in forecast accuracy from the use of these hybrid models. These hybrid models can leverage the transparency, feature selection, and interpretability of ML models combined with the process oriented approaches of high-dimensional, unstructured datasets characteristic of DL models. Many DL models have a major limitation ie; lack of transparency; DL models operate as "black boxes," and may often leave farmers, agronomists and decision makers confused or unclear about the derivation of their resultant forecasts which can undermine trust and adoption in actual agricultural settings[8]. 
 
Figure: 4- Yield Fore Casting 
C:\Users\DELL\Downloads\line_yield_forecasting.png 
        Figure: 5-Yield forecasting across years (2001–2015). 
 
c. Forecast of Rainfall 
 
The deep learning (DL) techniques have significantly enhanced rainfall prediction. The predictive weather in the past was largely by the use of traditional statistical methods such as the ARIMA and linear regression. The methods however have been superseded by DL methods which have upheld the ability of the methods to discover the complex, non-linear trends of data over time[9]. 
 
 RNNs, along with LSTMs, were the pioneers in the deep learning (DL) field that had pre-chosen the methods of prediction of their feature due to the ability to work with the sequential data. However, they might not easily trace back the long historical relations and they might be costly in calculating in case the data is large.  
 
The current and emerging trend in meteorology is the application of transformer based architecture in the area of precipitation forecasting as the most appropriate prediction technique. Transformers which use self-attention to identify delayed relationships in time series data in a faster way are different than step wise working LSTMs. So it is highly beneficial in the case of datasets of climate having yearly seasonal cycles and multi-year variability[10],[19].  
 
Other than that, the parallel working model also acquires training speed and demonstrates greater scalability. There is however a feel of something being left out with such technological advancement; they are open as most of the DL models are built in such a manner that cannot be understood and therefore they are very much like black boxes providing little or no clue of what logic is being used to drive their predictions [17]. specific to rainfall prediction, so as to shed light-on-decisions.C:\Users\DELL\Downloads\donut_feature_importance.png 
Figure: 6-Donut Chart – Feature Importance (Rainfall 40%, Soil pH 20%, Nitrogen 15%, PAR 15%, Moisture 10%). 
 
 
 
 
Figure: 7- Rainfall Prediction 
 
d) Explainable AI in Agriculture: 
 
SHAP and LIME markets XAI methods have proven quite effective in bridging the user trust - model performance gap in understanding[1],[2][16]. These methods allow the users to learn the most powerful environmental and soil parameters showing the dependence of the model on features. Recent survey of key issues in the last decade suggests the growing interest and the possible advantage of using ensemble-XAI techniques to address the transparency issue in diverse and complicated agricultural tasks[20]. In any case, the existing literature tends to consider rainfall prediction, yield forecasting and crop recommendation as individual problems. Few studies have studies that have formulated the structure of XAI in order to address these tasks concurrently with each other resulting in the gap in research which is addressed in this work[17],[20]. 
 
Figure: 8 – Expandable AI Schematic diagram 
C:\Users\DELL\Downloads\flowchart_agriculture_framework.png 
Figure: 8 – Block Diagram of process. 
 
III. Methodology 
a) Data Sources: 
The proposed structure, the application of different sets of data is critical in making agricultural forecasts that are not only strong but also varied. Firstly, as an initiator of such statistics, the yield of crops and production, the International Crops Research Institute for the Semi-Arid Tropics (ICRISAT) [8] has been selected as a source. The data about soil which is composed of pH, nitrogen, phosphorus, potassium and texture are provided by NATMO geo-portal [3]. The utilized weather information comprises of temperature, humidity, solar radiation, and rainfall and all of them are obtained on NASA POWER-site[9],[10],[19].C:\Users\DELL\Downloads\bar_model_accuracy.png 
Figure:8-Model comparison (SVM, LSTM, GRU, Transformer vs. Accuracy). 
 
b) Data Preprocessing: 
 
To achieve clean and usable data, some procedure work of data preprocessing has been performed as follows:  
 
Outlier Removal and Missing Values: The anomalies have been identified using The Inter-Quartile Range (IQR) tool and the blank areas have been filled in using the data imputation tool [2], [6].  
 
Normalization: Z-score and Min-Max are utilized to scale the features depending on the distribution of the attribute [7].  
 
Categorical Encoding: The one-hot encoding method [3], [4] has been used to encode different types of soils and types of crops into number. 
  
Feature Engineering: New measures such as Normalized Difference Vegetable Index (NDVI) and crop development phases as well as drought indicators were implemented on the initial data [8], [11]. 
 
 
 
 
 
c) Feature Extraction: 
 
Features are represented by using XLNet, a transformer-based language model, which is designed for the tabular and sequential agricultural data. Traditional feature extraction methods cannot perform this function. XLNet can recognize the interactions among soil conditions, weather variables, and crop responses, which are bidirectional temporal dependencies, thus, it can provide more comprehensive features [16], [20]. 
 
d) Feature Selection: 
 
In order to optimize the model inputs, We decided to use the EBMO evolutionary algorithm to optimize the inputs of our model. The main goal of this algorithm was to select most significant features. EBMO enhances the rate of convergence. and avoids local optima thereby improving the forecasting ability across diverse tasks[16], [17].  
 
C:\Users\DELL\Downloads\pie_crop_distribution.png 
 
Figure:9-Distribution of crops (Rice, Sorghum, Cotton, Sugarcane, Rabi). 
 
e) Predictive Models: 
This framework is a combination of various predictive models to solve various agricultural activities:  
Classification with XLNet characteristics and Support Vector Machines [3], [4], [16].  
Yield Forecasting Time series prediction using XLNet + LSTM/GRU networks [6], [7], [11], [18].  
Whereas, Rainfall Prediction: Predicting long-term climate dependencies modeled using transformer-based architectures[9], [10], [19]. 

f) Explainability Layer: 
 
To be interpretable, two XAI techniques are combined:  
 
• LIME gives local explanations, in the sense of demonstrating why a given crop or forecast should be suggested to a given region[5], [18]. 
 
 • SHAP gives the global and local priorities of features, which assists in determining the important variables (e.g., rainfall, soil pH, PAR) [1], [2], [16], [19].  
 
The features represented are through the use of XLNet, a transformer-based language model, which is one that is designed to work with the tabular and sequential agricultural data.  
 
This is a feature extraction task that traditional feature extraction techniques are unable to do. XLNet is able to identify the associations between soil conditions, weather variables and crop reactions, which are two way temporal relations and, therefore, it can offer more holistic features [16], [20]. 
 
IV. Experimental Setup 
a) Dataset Description: 
The 6 datasets are of the years 2001 to 2015, and cover 34 districts of South India. The data are the crop data of rice, sorghum, sugarcane, cotton, and Rabi crops[8], [11].  
The soil characteristics are N, pH, P, K, and organic matter, while the weather variables include rainfall, humidity, temperature, and solar irradiance [9], [10], [19]. 
b) Computational Environment: 
The computational environment is commonly categorized into three types: interactive, batch, and batch-interactive. <|human|> Computational Environment.  
There are three common types of computational environment: interactive, batch, and batch-interactive.  
Hardware: Intel i7-10700, 16 GB RAM, 512GB SSD.  
Software Python 3.10, TensorFlow, Scikit-learn, SHAP, LIME [2], [12]. 
c) Evaluation Metrics: 
The performance is measured according to standard regression and classification measures:  
• Regression problems: Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Square Error (RMSE) [6], [7], [11]. Classification tasks: Accuracy, Precision, Recall and F1-score[3], [4]. 
 • Goodness of fit: R 2 score in the prediction of tasks [9], [10]. On major crops, the XLNet + SVM model achieved accuracies of more than 98%[3], [4], [16].  
Nitrogen values and soil pH became known by the SHAP values as significant determinants [1], [2], [16]. It was demonstrated that rainfall was the key determinant of rice requirement by LIME [5], [18]. 
 
C:\Users\DELL\Downloads\scatter_rainfall_vs_yield.png 
          Figure:9- Rainfall vs Yield (positive correlation). 

V. Findings and Conversation 
