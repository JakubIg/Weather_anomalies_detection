###Detecting anomalies###

# =============================================================================
#1.Libraries
# =============================================================================
#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
from sklearn import preprocessing
from sklearn.decomposition import PCA
from dateutil.parser import parse
import itertools as it
import seaborn as sns
from sklearn.cluster import KMeans
#3D plot
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D

# =============================================================================
# 2.Functions
# =============================================================================
#Distance from points to centroids
def getDistanceByPoint(data, model):
    distance = pd.Series()
    for i in range(0,len(data)):
        Xa = np.array(data.loc[i])
        Xb = model.cluster_centers_[model.labels_[i]-1]
        distance.set_value(i, np.linalg.norm(Xa-Xb))
    return distance

#Histogram for outliers
def outliers_hist(weather, outlier_index, add_to_index, present_or_past):
    if(present_or_past == "w"):
        x_axis_name = "Past hour weather (w)"
    else:
        x_axis_name = "Present hour weather (ww)"
    
    #Preparing data
    weather_outliers = pd.DataFrame(outlier_index) + add_to_index
    weather = weather[weather_outliers.values[:,0]]
    weather = pd.DataFrame({x_axis_name: weather, 'Count':np.repeat(1,len(weather))})
    groupedvalues=weather.groupby(x_axis_name).sum().reset_index()
    pal = sns.color_palette("Blues_d", len(groupedvalues))
    rank = groupedvalues["Count"].argsort().argsort() 
    
    #Creating plot
    g=sns.barplot(x=x_axis_name,y='Count',data=groupedvalues, palette=np.array(pal[::-1])[rank])
    for index, row in groupedvalues.iterrows():
        g.text(row.name,row.Count + 2, round(row.Count,2), color='black', ha="center")
    sns.set(font_scale = 1.5)
    g.set_xticklabels(g.get_xticklabels(),rotation=90, ha='right')
    plt.show()

#Create mean values barplot
def creating_plot(combined_val_date):
    # PLOTTING
    combined_val_date = combined_val_date.drop(index = [len(combined_val_date) - 1, len(combined_val_date) - 2])
    fig, ax = plt.subplots(figsize = (12,6))    
    fig = sns.barplot(x = "Month", y = "Value", hue = "Year", data = combined_val_date, 
                      estimator = sum, ci = None, ax=ax)
    x_dates = combined_val_date['Month'].unique()
    ax.set_xticklabels(labels=x_dates, rotation=45, ha='right')
    sns.set(font_scale = 1.1)
    # Put the legend out of the figure
    plt.legend(title = "Year",bbox_to_anchor=(1.03, 0.9), loc=2, borderaxespad=0.)
    plt.show()
    
#Create 3d plots with signed anomalies
def threeDim_plot(dataset,to_model_columns):
    pca = PCA(n_components=3)  # Reduce to k=3 dimensions
    scaler = StandardScaler()
    #normalize the metrics
    X = scaler.fit_transform(dataset[to_model_columns])
    X_reduce = pca.fit_transform(X)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_zlabel("x_composite_3")# Plot the compressed data points
    ax.scatter(X_reduce[:, 0], X_reduce[:, 1], zs=X_reduce[:, 2], s=4, lw=1, label="inliers",c="green")# Plot x's for the ground truth outliers
    ax.scatter(X_reduce[outlier_index,0],X_reduce[outlier_index,1], X_reduce[outlier_index,2],
               lw=2, s=60, marker="x", c="red", label="outliers")
    ax.legend()
    plt.show()

#Create pivot histograms to see density of normal observatons and anomalies
def pivot_anomaly_histograms(dataset,column):
    anom = dataset[dataset['anomaly'] == 'anomaly']
    norm = dataset[dataset['anomaly'] == 'normal']
    sns.kdeplot(anom[column], shade=True, color="r", label = 'anomaly')
    sns.kdeplot(norm[column], shade=True, color="g", label = 'normal')
    plt.ylabel('Distribution [%]')
    plt.xlabel('Value')
    sns.set(font_scale = 1.5)

#How the weather changed over the next few hours
def weather_changes(weather, weather_list,present_or_past,val_added,outlier_index):
    if(present_or_past == "w"):
        x_axis_name = "Past hour weather (w)"
    else:
        x_axis_name = "Present hour weather (ww)"
    
    weather_throw_hours = pd.DataFrame()
    weather_throw_hours[x_axis_name] = weather_list
    for add_to_index in val_added:
        print(add_to_index)
        #Preparing data
        weather_outliers = pd.DataFrame(outlier_index) + add_to_index
        weather_outliers = weather[weather_outliers.values[:,0]]
        weather_outliers = pd.DataFrame({x_axis_name: weather_outliers,
                                         'Count':np.repeat(1,len(weather_outliers))})
        groupedvalues=weather_outliers.groupby(x_axis_name).sum().reset_index()
        df = groupedvalues
        if(present_or_past == "w"):
            df.set_index("Past hour weather (w)", inplace = True)
        else:
            df.set_index("Present hour weather (ww)", inplace = True)
        #Creating frame
        weather_throw_hours[add_to_index] = np.array(df.loc[weather_list])

    
    
    first_columns = weather_throw_hours.values[:,0]
    second_columns = pd.DataFrame(weather_throw_hours.values[:,1:len(weather_throw_hours.columns)])
    to_plot = pd.DataFrame({x_axis_name: [0], 'Count': [0]})
    for j in range(0, len(weather_throw_hours)):
        for i in range(0, (len(weather_throw_hours.columns) - 1)):
            to_plot = to_plot.append(pd.DataFrame({x_axis_name: [first_columns[j]], 'Count': [second_columns.values[j,i]]}))
    to_plot.set_index(np.arange(0,len(to_plot)), inplace = True)
    to_plot.drop(0, inplace = True)
    to_plot.set_index(np.arange(0,len(to_plot)), inplace = True)
    to_plot = pd.concat([to_plot,pd.DataFrame({'Hour':np.tile(val_added, len(weather_list))})], axis = 1)
    #Creating plot
    fig, ax = plt.subplots(figsize = (12,6))    
    fig=sns.barplot(x=x_axis_name,y='Count', hue = "Hour",data=to_plot,estimator = sum, ci = None, ax=ax)
    x_dates = to_plot[x_axis_name].unique()
    ax.set_xticklabels(labels=x_dates, ha='right')
    sns.set(font_scale = 1.5)
    plt.show()
# =============================================================================
# 3. Dataset
# =============================================================================
# Importing the dataset
dataset = pd.read_csv("dublin.csv",skiprows = 23)

#Removing unnecessary columns
columns_to_remove = ['ind', 'ind.1', 'ind.2', 'ind.3', 'ind.4']
dataset.drop(columns_to_remove, inplace=True, axis=1)
    
#Detecting NaN, NA and empty strings
dataset.isnull().sum()

dataset.isna().sum()

for colnames in dataset.columns:
    print(colnames)
    print(len(dataset[dataset[colnames] == " "]))

#Numbers of rows with empty values
vappr_empty = dataset.index[dataset['vappr'] == " "].tolist()[0]
rhum_empty = dataset.index[dataset['rhum'] == " "].tolist()[0]
wddir_empty = dataset.index[dataset['wddir'] == " "].tolist()[0]

#Replacing missing value with value from 1 hour before and changing
#data type to int/float.
dataset.at[vappr_empty, 'vappr'] = dataset.at[vappr_empty-1, 'vappr']
dataset.at[rhum_empty, 'rhum'] = dataset.at[rhum_empty-1, 'rhum']
dataset.at[wddir_empty, 'wddir'] = dataset.at[wddir_empty-1, 'wddir']

dataset['vappr'] = dataset['vappr'].astype(float)
dataset['rhum'] = dataset['rhum'].astype(int)
dataset['wddir'] = dataset['wddir'].astype(int)

# =============================================================================
# 4. Creating plots to detect if mean variables has changed throw passed years
# =============================================================================
#Creating variable containing all years
years = [dataset.values[0,0], dataset.values[(len(dataset.index) - 1),0]]
years = np.arange(parse(years[0]).year,parse(years[1]).year + 1)

#Separating date column
date = pd.DataFrame(dataset["date"].str.split(" ", n = 1, expand = True)) 
dataset["hour"] = date.values[:,1]
date = date[date.columns[0]].str.split("-", n = 2, expand = True)  
dataset["year"] = date.values[:,2]
dataset["month"] = date.values[:,1]
dataset["day"] = date.values[:,0]

###Creating variable containing all combinations of variables such as months and years###
months = pd.Series(dataset["month"]).unique()
months_years = pd.DataFrame([i for i in it.product(years,months)])

mean_all_cols = []
for j in range(1, len(dataset.columns) - 4):
    print(j)
    mean_values = []
    #Getting mean values in all months in all years
    for i in range(0, len(months_years.index) - 1):
        x = dataset.values[(dataset['year'] == str(months_years.values[i,0])) & (dataset['month'] == months_years.values[i,1])]
        mean_values.append(x[:,j].mean())
    
    #Combinig frames with dates and mean values
    combined_val_date = pd.concat([pd.DataFrame(mean_values),months_years],axis = 1)
    combined_val_date = combined_val_date.drop(index = [len(combined_val_date) - 1, len(combined_val_date) - 2])
    combined_val_date.columns = ['Value','Year','Month']
    mean_all_cols.append(combined_val_date)

#Getting descriptive statistics
perc = [0.2,0.8]
stats = dataset.values[:,1:16]
stats= pd.DataFrame(stats)
stats.columns = dataset.columns[1:16]
colnames_without_ind = stats.columns
for colname in colnames_without_ind:
    stats[colname] = stats[colname].astype(float)
stats.describe(percentiles = perc)

##PLOTS
#Rain
creating_plot(mean_all_cols[0])
#Temp
creating_plot(mean_all_cols[1])
#Wetb
creating_plot(mean_all_cols[2])
#Dewtb
creating_plot(mean_all_cols[3])
#Vappr
creating_plot(mean_all_cols[4])
#Rhum
creating_plot(mean_all_cols[5])
#Msl
creating_plot(mean_all_cols[6])
#Wdsp
creating_plot(mean_all_cols[7])
#Wddir
creating_plot(mean_all_cols[8])
#Sun
creating_plot(mean_all_cols[11])
#Vis
creating_plot(mean_all_cols[12])
#Clht
creating_plot(mean_all_cols[13])
#Clamt
creating_plot(mean_all_cols[14])

# =============================================================================
# 5. Additional code
# =============================================================================
#Looking at values for cloud ceiling heighht. If there are no clouds,
#value is equal to 999. It was decided to change that value to -1, to make
#distance to other observations smaller.
dataset.clht.unique()
fig, axs = plt.subplots()
plt.hist(dataset['clht'])
plt.legend()
plt.show()

dataset['clht'].replace([999], [-1], inplace = True)

#Removing 'w' and 'ww' variables before going to clustering
past_weather = dataset['w']
present_weather = dataset['ww']
columns_to_remove = ['w', 'ww']
dataset.drop(columns_to_remove, inplace=True, axis=1)


#Creating dummy variables for months
dummy_months = pd.DataFrame({'Month': dataset['month']})
dummy_months = pd.get_dummies(dummy_months['Month'])
dummy_months.drop('jan', inplace = True, axis = 1)
# =============================================================================
#6. Creating Models
# =============================================================================
#Standarizing data
to_model_columns=dataset.columns[1:14]
min_max_scaler = preprocessing.StandardScaler()
np_scaled = pd.DataFrame(min_max_scaler.fit_transform(dataset[to_model_columns]))
np_scaled.columns = to_model_columns
#Combining dataset with months dummy columns
np_scaled = pd.concat([np_scaled,dummy_months], axis = 1)

####Isolation Forest####
#specify the 14 metrics column names to be modelled

from sklearn.ensemble import IsolationForest
clf=IsolationForest(contamination=0.01, behaviour='new',
                    n_jobs=-1, random_state=42)
clf.fit(np_scaled)
pred = clf.predict(np_scaled)
dataset['anomaly']=pred
outliers=dataset.loc[dataset['anomaly']==-1]
outlier_index=list(outliers.index)

#Histogram of past weather
outliers_hist(past_weather, outlier_index, 1,"w")
#Histogram of present weather
outliers_hist(present_weather, outlier_index, -1,"ww")
#Histogram of past weather in next observation
outliers_hist(past_weather, outlier_index, 5,"w")
#Histogram of present weather in next observation
outliers_hist(present_weather, outlier_index, 5,"ww")

#Find the number of anomalies and normal points here points classified -1 are anomalous
print(dataset['anomaly'].value_counts())


#Changing labels from numbers to more clear ones
dataset['anomaly'].replace([1], ['normal'], inplace = True)
dataset['anomaly'].replace([-1], ['anomaly'], inplace = True)

#Compare density plot for normal observations and anomalies // Column rain
pivot_anomaly_histograms(dataset, 'rain')
#Compare density plot for normal observations and anomalies // Column temp
pivot_anomaly_histograms(dataset, 'temp')
#Compare density plot for normal observations and anomalies // Column wetb
pivot_anomaly_histograms(dataset, 'wetb')
#Compare density plot for normal observations and anomalies // Column dewpt
pivot_anomaly_histograms(dataset, 'dewpt')
#Compare density plot for normal observations and anomalies // Column vappr
pivot_anomaly_histograms(dataset, 'vappr')
#Compare density plot for normal observations and anomalies // Column rhum
pivot_anomaly_histograms(dataset, 'rhum')
#Compare density plot for normal observations and anomalies // Column msl
pivot_anomaly_histograms(dataset, 'msl')
#Compare density plot for normal observations and anomalies // Column wdsp
pivot_anomaly_histograms(dataset, 'wdsp')
#Compare density plot for normal observations and anomalies // Column wddir
pivot_anomaly_histograms(dataset, 'wddir')
#Compare density plot for normal observations and anomalies // Column sun
pivot_anomaly_histograms(dataset, 'sun')
#Compare density plot for normal observations and anomalies // Column vis
pivot_anomaly_histograms(dataset, 'vis')
#Compare density plot for normal observations and anomalies // Column clht
pivot_anomaly_histograms(dataset, 'clht')
#Compare density plot for normal observations and anomalies // Column clamt
pivot_anomaly_histograms(dataset, 'clamt')

#Creating 3D plot with normal observations and anomalies
threeDim_plot(dataset,to_model_columns)

#How the weather changed over the next few hours // w
weather_changes(past_weather,[0,22,62,66,76], "w", [0,1,2,3,4,5,6])
weather_changes(past_weather,[65,82], "w", [0,1,2,3,4,5,6])
weather_changes(past_weather,[92,94,96,98], "w", [0,1,2,3,4,5,6])


#How the weather changed over the next few hours // ww
weather_changes(past_weather,[2,5,10,21,25,61,63,80], "ww", [0,1,2,3,4,5,6])
weather_changes(past_weather,[77,86,90,92,95,96,97], "ww", [0,1,2,3,4,5,6])

#PCA
isolation_anomaly = dataset['anomaly']
isolation_anomaly.columns = ['isolation_anomaly']
isolation_anomaly = pd.DataFrame(isolation_anomaly.rename('isolation_anomaly'))

#####One class SVM####
from sklearn.svm import OneClassSVM
clf=OneClassSVM(nu=0.95 * 0.01)
clf.fit(np_scaled)
pred = clf.predict(np_scaled)
dataset['anomaly']=pred
outliers=dataset.loc[dataset['anomaly']==-1]
outlier_index=list(outliers.index)
#print(outlier_index)
#Find the number of anomalies and normal points here points classified -1 are anomalous
print(dataset['anomaly'].value_counts())

#Histogram of past weather
outliers_hist(past_weather, outlier_index, 0,"w")
#Histogram of present weather
outliers_hist(present_weather, outlier_index, 0,"ww")
#Histogram of past weather in next observation
outliers_hist(past_weather, outlier_index, 1,"w")
#Histogram of present weather in next observation
outliers_hist(present_weather, outlier_index, 1,"ww")

#Changing labels from numbers to more clear ones
dataset['anomaly'].replace([1], ['normal'], inplace = True)
dataset['anomaly'].replace([-1], ['anomaly'], inplace = True)

#Compare density plot for normal observations and anomalies // Column rain
pivot_anomaly_histograms(dataset, 'rain')
#Compare density plot for normal observations and anomalies // Column temp
pivot_anomaly_histograms(dataset, 'temp')
#Compare density plot for normal observations and anomalies // Column wetb
pivot_anomaly_histograms(dataset, 'wetb')
#Compare density plot for normal observations and anomalies // Column dewpt
pivot_anomaly_histograms(dataset, 'dewpt')
#Compare density plot for normal observations and anomalies // Column vappr
pivot_anomaly_histograms(dataset, 'vappr')
#Compare density plot for normal observations and anomalies // Column rhum
pivot_anomaly_histograms(dataset, 'rhum')
#Compare density plot for normal observations and anomalies // Column msl
pivot_anomaly_histograms(dataset, 'msl')
#Compare density plot for normal observations and anomalies // Column wdsp
pivot_anomaly_histograms(dataset, 'wdsp')
#Compare density plot for normal observations and anomalies // Column wddir
pivot_anomaly_histograms(dataset, 'wddir')
#Compare density plot for normal observations and anomalies // Column sun
pivot_anomaly_histograms(dataset, 'sun')
#Compare density plot for normal observations and anomalies // Column vis
pivot_anomaly_histograms(dataset, 'vis')
#Compare density plot for normal observations and anomalies // Column clht
pivot_anomaly_histograms(dataset, 'clht')
#Compare density plot for normal observations and anomalies // Column clamt
pivot_anomaly_histograms(dataset, 'clamt')

#Creating 3D plot with normal observations and anomalies
threeDim_plot(dataset,to_model_columns)

#How the weather changed over the next few hours // w
weather_changes(past_weather,[0,22,62,66,76], "w", [0,1,2,3,4],outlier_index)
weather_changes(past_weather,[65,82], "w", [0,1,2,3,4],outlier_index)
weather_changes(past_weather,[76,92,94,96,98], "w", [0,1,2,3,4],outlier_index)


#How the weather changed over the next few hours // ww
weather_changes(present_weather,[2,5,10,25,61,63,80], "ww", [0,1,2,3,4],outlier_index)
weather_changes(present_weather,[77,86,92,93,94,95,96,97], "ww", [-5,-4,-3,-2,-1,0,1,2,3,4],outlier_index)

#PCA
svm_anomaly = dataset['anomaly']
svm_anomaly.columns = ['svm_anomaly']
svm_anomaly = pd.DataFrame(svm_anomaly.rename('svm_anomaly'))


#Kmeans
#Elbow plot for optimal number of clusters
X = np_scaled
wcss = []
for i in range(1, 15):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 15), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Fitting K-Means (3 clusters as shown above) to the dataset
kmeans = KMeans(n_clusters = 8, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)
Counter(y_kmeans).keys()
Counter(y_kmeans).values()

#Fitting kmeans to data
kmeans = KMeans(n_clusters=8).fit(X)

#Getting clusters and number of observations in them
X['cluster'] = kmeans.predict(X)
X['cluster'].value_counts()

#Getting the distance between each point and its nearest centroid
#Setting number of anomalies as 1% of data
outliers_fraction = 0.01
distance = getDistanceByPoint(data, kmeans)
number_of_outliers = int(outliers_fraction*len(distance))
threshold = distance.nlargest(number_of_outliers).min()

#isAnomaly contains the anomaly result (binary: 0:normal, 1:anomaly) 
X['anomaly'] = (distance >= threshold).astype(int)

#PCA
pca = PCA(n_components=2)
X_PCA = X.drop(['cluster', 'isAnomaly', 'anomaly'], axis = 1)
X_PCA = pca.fit_transform(X_PCA)
X_PCA = pd.DataFrame(X_PCA)
X_PCA.columns = ['principal_feature1', 'principal_feature2']
X_PCA = pd.concat((X_PCA, X['cluster'], X['anomaly']), axis = 1)

#Visualisation of anomaly with cluster view
fig, ax = plt.subplots()
colors = {'normal':'blue', 'anomaly':'red'}
ax.scatter(X_PCA['principal_feature1'], X_PCA['principal_feature2'],
           c=X_PCA["anomaly"].apply(lambda x: colors[x]))
plt.show()

#Plotting the different clusters with the 2 main features
fig, ax = plt.subplots()
colors = {0:'red', 1:'blue', 2:'green', 3:'pink', 4:'black', 5:'orange', 6:'cyan', 7: 'white'}
ax.scatter(X_PCA['principal_feature1'], X_PCA['principal_feature2'], c=X_PCA["cluster"].apply(lambda x: colors[x]))
plt.show()

#Changing labels from numbers to more clear ones
a = X.loc[X['anomaly'] == 0]
b = X.loc[X['anomaly'] == 1]
X['anomaly'].replace([1], ['anomaly'], inplace = True)
X['anomaly'].replace([0], ['normal'], inplace = True)
outlier_index = X.index[X['anomaly'] == 'anomaly']
dataset['anomaly'] = X['anomaly']
#Histogram of past weather
outliers_hist(past_weather, outlier_index, 0, "w")
#Histogram of present weather
outliers_hist(present_weather, outlier_index, 0, "ww")
#Histogram of past weather in next observation
outliers_hist(past_weather, outlier_index, 1, "w")
#Histogram of present weather in next observation
outliers_hist(present_weather, outlier_index, 1,"ww")

#Compare density plot for normal observations and anomalies // Column rain
pivot_anomaly_histograms(dataset, 'rain')
#Compare density plot for normal observations and anomalies // Column temp
pivot_anomaly_histograms(dataset, 'temp')
#Compare density plot for normal observations and anomalies // Column wetb
pivot_anomaly_histograms(dataset, 'wetb')
#Compare density plot for normal observations and anomalies // Column dewpt
pivot_anomaly_histograms(dataset, 'dewpt')
#Compare density plot for normal observations and anomalies // Column vappr
pivot_anomaly_histograms(dataset, 'vappr')
#Compare density plot for normal observations and anomalies // Column rhum
pivot_anomaly_histograms(dataset, 'rhum')
#Compare density plot for normal observations and anomalies // Column msl
pivot_anomaly_histograms(dataset, 'msl')
#Compare density plot for normal observations and anomalies // Column wdsp
pivot_anomaly_histograms(dataset, 'wdsp')
#Compare density plot for normal observations and anomalies // Column wddir
pivot_anomaly_histograms(dataset, 'wddir')
#Compare density plot for normal observations and anomalies // Column sun
pivot_anomaly_histograms(dataset, 'sun')
#Compare density plot for normal observations and anomalies // Column vis
pivot_anomaly_histograms(dataset, 'vis')
#Compare density plot for normal observations and anomalies // Column clht
pivot_anomaly_histograms(dataset, 'clht')
#Compare density plot for normal observations and anomalies // Column clamt
pivot_anomaly_histograms(dataset, 'clamt')

#Creating 3D plot with normal observations and anomalies
threeDim_plot(dataset,to_model_columns)

#How the weather changed over the next few hours // w
weather_changes(past_weather,[0,22,62,65,66], "w", [0,1,2,3,4,5], outlier_index)
weather_changes(past_weather,[81,82], "w", [0,1,2,3,4,5], outlier_index)
weather_changes(past_weather,[76,92,96,98], "w", [0,1,2,3,4,5],outlier_index)


#How the weather changed over the next few hours // ww
weather_changes(present_weather,[10,21,25,61,63,80], "ww", [0,1,2,3,4,5], outlier_index)
weather_changes(present_weather,[86,90,92,95,96,97], "ww", [0,1,2,3,4,5],outlier_index)

#PCA
kmeans_anomaly = X['anomaly']
kmeans_anomaly = pd.DataFrame(kmeans_anomaly.rename('kmeans_anomaly'))

comparison = pd.concat((comparison, isolation_anomaly,svm_anomaly,kmeans_anomaly), axis = 1)

# =============================================================================
# Coparison of 2D plots
# =============================================================================

comparison = np_scaled[to_model_columns]
pca = PCA(n_components=2)
comparison = pca.fit_transform(comparison)
comparison = pd.DataFrame(comparison)
comparison.columns = ['principal_feature1', 'principal_feature2']

#Visualisation of anomaly with cluster view
fig, ax = plt.subplots()
colors = {'normal':'blue', 'anomaly':'red'}
ax.scatter(np_scaled['principal_feature1'], np_scaled['principal_feature2'],
           c=np_scaled["anomaly"].apply(lambda x: colors[x]))
plt.show()


