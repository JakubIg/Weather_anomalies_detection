One of the fields using algorithms to find anomalies in the data set is meteorology, and the search for weather anomalies using machine learning methods is the subject of numerous scientific papers. The aim of this work is to create models that will be able to recognize weather anomalies in the best possible way. The study is based on data from Dublin Airport and includes hourly weather information from 1989 (00:00) to 31 October 2019 (23:00). The database is updated with some delay, so the data we downloaded ends in October this year. The classification was carried out in an unsupervised manner, so it does not define in advance the observation data as weather anomalies, but this task is left to the classification methods. The potential outlier observations distinguished by the algorithms will then be verified on the basis of the weather description, already included in the data set. The study was conducted in Python, while the tables were created in MS Excel.  
  
K-means, Isolation Forest and One-Class SVM methods were used to create models to detect anomalies. The Rand Index method and the comparison of the results with the weather variables in the current and previous hour were used to choose the method which results turned out to be the best.  

## Contributors

- **Jakub Ignatik**
- **Mateusz Jałocha** (https://github.com/MateuszJalocha, https://www.linkedin.com/in/mateusz-jałocha/)
- **Jakub Augustynek** (https://www.linkedin.com/in/jakub-augustynek-881a07188/)
