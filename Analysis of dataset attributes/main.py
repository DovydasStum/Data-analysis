import pandas as pd_library
import numpy as np
import matplotlib.pyplot as pyplot
import seaborn as seaborn_
from collections import Counter
from sklearn.preprocessing import LabelEncoder

class InOutUtils:
    def PrintResultsForBoth(header, data):
        print("\n\n{0}\n".format(header))
        print("Data count:")
        print(TaskUtils.AtributesCount(data))

        print("\nMissing values %:")
        print(TaskUtils.MissingValues(data))

        print("\nCardinality:")
        print(TaskUtils.Cardinality(data))

    def PrintResultsForContinous(data):      
        print("\nMinimal value:")
        print(TaskUtils.MinValue(data))

        print("\nMaximal value:")
        print(TaskUtils.MaxValue(data))

        print("\nQuartiles (first and third):")
        print(TaskUtils.Quartiles(data))

        print("\nMean:")
        print(TaskUtils.Mean(data))

        print("\nMedian:")
        print(TaskUtils.Median(data))

        print("\nStandart deviation:")
        print(TaskUtils.StandartDeviation(data))

    def PrintResultsForCategorical(data):
        print("\nMode:")
        print(TaskUtils.ModeAndFrequency(data))  

        print("\nMode frequency %:")
        print(TaskUtils.ModePercent(data)) 

        print("\nSecond mode:")
        print(TaskUtils.SecondModeAndFrequency(data))  

        print("\nSecond mode frequency %:")
        print(TaskUtils.SecondModePercent(data))        


class TaskUtils:
    def ReadFile(name):
        data = pd_library.read_csv(name)
        return data

    def AtributesCount(data):
        return data.count()
    
    def MissingValues(data):
        missingvalues = data.isnull().sum() * 100 / len(data)
        return missingvalues
    
    def Cardinality(data):
        return data.nunique()
    
    def MinValue(data):
        return data.min()
    
    def MaxValue(data):
        return data.max()
    
    def Quartiles(data):
        return data.quantile([0.25, 0.75])
    
    def Mean(data):
        return data.mean()
    
    def Median(data):
        return data.median()
    
    def StandartDeviation(data):
        return data.std()
    
    def ModeAndFrequency(data):
        return pd_library.DataFrame({'Column': data.columns,
                         'Value, frequency': [Counter(data[x]).most_common()[0]
                          for x in data]})

    def ModePercent(data):
        return  pd_library.DataFrame({'Column': data.columns,
                         'Value': [data[x].isin([Counter(data[x])
                        .most_common()[0][0]]).sum()
                        * 100 / len(data[x])  for x in data]})   

    def SecondModeAndFrequency(data):
        return  pd_library.DataFrame({'Column': data.columns,
                         'Value, frequency': [Counter(data[x]).most_common()[1]
                          for x in data]})

    def SecondModePercent(data):
        return  pd_library.DataFrame({'Column': data.columns,
                         'Value': [data[x].isin([Counter(data[x])
                        .most_common()[1][0]]).sum()
                        * 100 / len(data[x])  for x in data]})       

    def plotScatterMatrix(data):
        axes = pd_library.plotting.scatter_matrix(data, alpha=0.2)
        for ax in axes.flatten():
            ax.xaxis.label.set_rotation(0)
            ax.yaxis.label.set_rotation(0)
            ax.yaxis.label.set_ha('right')
        pyplot.autoscale()
        pyplot.show()

    def plotHistogram(data):
        fig, ax = pyplot.subplots(figsize=(20, 20)) 
        data['vehicle.year'].hist(ax=ax, bins=29)
    
        # 'vehicle.year', 'reviewCount', 'rating', 'vehicle.make', 'vehicle.model',
        # 'vehicle.type', 'fuelType', 'location.city'
        ax.set_title('vehicle.year')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency') 
        ax.xaxis.set_tick_params(rotation=0, labelsize=12)
        pyplot.show()

    def plotScatterGraph(data):
        #seaborn_.lmplot(x="reviewCount", y="vehicle.year", data=data, line_kws={'color': 'red'})
        #seaborn_.lmplot(x="reviewCount", y="rating", data=data, line_kws={'color': 'red'})
        seaborn_.lmplot(x="vehicle.year", y="rating", data=data, line_kws={'color': 'red'})
        pyplot.show()

    def plotBoxGraph(data):
        df = pd_library.DataFrame(data)
        pyplot.figure(figsize=(12, 8))
        seaborn_.boxplot(data=data, x='vehicle.type', y='rating')
        #pyplot.scatter(outliers['vehicle.type'], outliers['rating'], color='red', label='Outliers')
        pyplot.title('Automobilio tipas pagal įvertinimą')
        pyplot.xlabel('vehicle.type')
        pyplot.ylabel('rating')
        pyplot.xticks(rotation=0)
        pyplot.show()

    def plotBoxGraphOutliers(data):
        df = pd_library.DataFrame(data)
        pyplot.figure(figsize=(12, 8))
        seaborn_.boxplot(data=data, x='vehicle.type', y='rating')
        pyplot.title('Automobilio tipas pagal įvertinimą')
        pyplot.xlabel('vehicle.type')
        pyplot.ylabel('rating')
        pyplot.xticks(rotation=0)

        Q1 = df['rating'].quantile(0.25)
        Q3 = df['rating'].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df['rating'] < lower_bound) | (df['rating'] > upper_bound)]

        percentage = (len(outliers) / len(df)) * 100
        print("\n\nPercentage of outliers:", percentage, "%\n")

        pyplot.show()

    def normalize(data, intervalStart, intervalEnd):
        result = data.copy()
 
        for x in data:
            max = data[x].max()  
            min = data[x].min()
            result[x] = ((data[x] - min) / (max - min)) * (intervalStart - intervalEnd) + intervalStart
        return result 

    def plotBarGraph(data):
        df = pd_library.DataFrame(data)
        #truck_data = df[df['vehicle.type'] == 'truck']
        truck_data = df[df['vehicle.type'] == 'car']
        fuel_type_counts = truck_data['fuelType'].value_counts()
        pyplot.figure(figsize=(8, 6))
        seaborn_.barplot(x=fuel_type_counts.index, y=fuel_type_counts.values)
        pyplot.title('Vehicle type: car')
        pyplot.xlabel('Fuel type')
        pyplot.ylabel('Frequency')
        pyplot.show()
    
    def findAndRemoveOutliers(data, continuous_data, categorical_data):
        Q1 = continuous_data.quantile(0.25)
        Q3 = continuous_data.quantile(0.75)
        IQR = Q3 - Q1

        continuous_data = continuous_data[~((continuous_data < (Q1 - 1.5 * IQR)) | 
                                        (continuous_data > (Q3 + 1.5 * IQR))).any(axis=1)]

        cleaned_data = pd_library.concat([continuous_data, categorical_data], axis=1)
        return cleaned_data
  
    def HistogramCatAndCont(data):
        fig = pyplot.figure(figsize = (20,20))
        ax = fig.gca()
        #data['vehicle.year'].hist(by=data['fuelType'], ax=ax)
        data['rating'].hist(by=data['vehicle.type'], ax=ax)
        pyplot.show()

    def covariation(data):
        cov = data.drop(['vehicle.make', 'vehicle.model', 'vehicle.type',
                         'fuelType', 'location.city'], axis=1).cov()
        print(cov)

    def correlation(data):
        fig = pyplot.figure(figsize = (15,15))
        ax = fig.gca()
        corr = data.drop(['vehicle.make', 'vehicle.model', 'vehicle.type',
                     'fuelType', 'location.city'], axis=1).corr()
        seaborn_.heatmap(corr, annot=True, ax=ax)
        pyplot.show()

    def ConvertCatToCont(data):
        df = pd_library.DataFrame(data)
        label_encoder = LabelEncoder()

        for attribute in df.columns:
            df[attribute] = label_encoder.fit_transform(df[attribute])

        print("Converted atributes:\n",df)

        
def main():
    # Primary data reading
    data = TaskUtils.ReadFile("Final_data.csv")
    continous_atributes =   ['vehicle.year', 'reviewCount', 'rating']
    categorical_atributes = ['vehicle.make', 'vehicle.model', 'vehicle.type',
                             'fuelType', 'location.city']
    continous_data = data[continous_atributes]
    categorical_data = data[categorical_atributes]

 # ---------- 1-3 ---------------
    # Continous data atributes analysis
    #InOutUtils.PrintResultsForBoth("\n\nContinous data analysis:\n", continous_data)
    #InOutUtils.PrintResultsForContinous(continous_data)

    # Categorical data atributes analysis
    #InOutUtils.PrintResultsForBoth("\n\nCategorical data analysis:\n", categorical_data)
    #InOutUtils.PrintResultsForCategorical(categorical_data)

 # ---------- 4-6 ---------------
    # histograms
    #print("\nRecommended columns count: ", np.add(1,3.22*np.log(TaskUtils.AtributesCount(data)))) # 29
    #TaskUtils.plotHistogram(continous_data)
    #data_filtered = TaskUtils.findAndRemoveOutliers(data, continous_data, categorical_data)
    #TaskUtils.plotHistogram(data_filtered)

    #TaskUtils.plotHistogram(categorical_data)
 
 # ---------- 7.1 ---------------
    # scatter plot - continous data
    #TaskUtils.plotScatterGraph(continous_data)

 # ---------- 7.2 ---------------
    # scatter plot matrix (SPLOM) - continous data
    #TaskUtils.plotScatterMatrix(data)

 #  --------- 7.3 ---------------
    # bar plot - categorical data
    #TaskUtils.plotBarGraph(data)
    
 # ---------- 7.4 ---------------
    # box plot - categorical and continous
    filtered = TaskUtils.findAndRemoveOutliers(data, continous_data, categorical_data)
    TaskUtils.plotBoxGraph(filtered)


#  ---------- 7.5 ---------------
    # histograms - categorical and continous
    #TaskUtils.HistogramCatAndCont(data)

#  ---------- 8 ---------------
    # covariation
    #TaskUtils.covariation(data)

    # correlation
    #TaskUtils.correlation(data)

#  ---------- 9 ---------------
    # Normalization
    intervalStart = -3
    intervalEnd = 2
    print("\nNormalization:\n", TaskUtils.normalize(continous_data, intervalStart, intervalEnd).head(),'\n')

#  ---------- 10 ---------------
    # convert categorical to continous
    #TaskUtils.ConvertCatToCont(categorical_data)

if __name__ == "__main__":
    main()