# airline-booking-prediction
A predictive model to predict customer bookings for an airline, and it also highlights the use of data and machine learning techniques to achieve this goal.

## Plotting the important features for the model
![r](https://user-images.githubusercontent.com/115629197/208548097-f8030909-ea90-4885-b704-958972afde37.png)

## Plotting Categorical values

    def plot_categorical_distribution(data: pd.DataFrame = None, column: str = None, height: int = 8, aspect: int = 2):
    """
    Plot the distribution of a categorical variable
    :param data: The dataframe containing the data
    :param column: The column to plot
    :param height: The height of the plot
    :param aspect: The aspect ratio of the plot
    :return: None
    """
    sns.catplot(
        data=data,
        x=column,
        kind='count',
        height=height,
        aspect=aspect,
        order=data[column].value_counts().iloc[:10].index
    ).set(title=f'Distribution of {column}')
    
    
![output 1](https://user-images.githubusercontent.com/115629197/208546698-3f39ff09-6662-46ec-be07-83d1552daa7e.png)

![output 2](https://user-images.githubusercontent.com/115629197/208546705-1a874902-8a64-4abb-95cf-d5be80a45128.png)

![output 3](https://user-images.githubusercontent.com/115629197/208546708-f9872d1a-f053-4fb1-97e2-7a01ed6facb7.png)

![output 4](https://user-images.githubusercontent.com/115629197/208546711-b209bb97-33e3-438b-8f26-9e17df693111.png)

![output 5](https://user-images.githubusercontent.com/115629197/208546715-75d07477-6dc4-4944-b563-0936b875b339.png)

![output 6](https://user-images.githubusercontent.com/115629197/208546717-1b7697a4-ab71-4e6c-84ff-ef74344a74ef.png)

![output 7](https://user-images.githubusercontent.com/115629197/208546719-ece13a29-d3c5-4dfc-87e8-17b80f4e0da2.png)

![output](https://user-images.githubusercontent.com/115629197/208546722-3cea0014-8a1e-4a54-980c-dae30b1b0cd1.png)

## Plotting Continuous variables


    def plot_continuous_distribution(data: pd.DataFrame = None, column: str = None, height: int = 8):

    """
    Plot the distribution of a continuous variable
    :param data: The dataframe containing the data
    :param column: The column to plot
    :param height: The height of the plot
    :return: None
    
    """
    sns.displot(data, x=column, kde=True, height=height, aspect=height/5).set(title=f'Distribution of {column}')
    
    
   ![1](https://user-images.githubusercontent.com/115629197/208547518-710a6901-9af6-45d0-8ebb-50d31b427b5a.png)
   
![2](https://user-images.githubusercontent.com/115629197/208547523-e42940fb-ddfd-4b08-8408-44f497be5474.png)

![output](https://user-images.githubusercontent.com/115629197/208547526-4c764c19-7211-4483-aeeb-61cd469cac82.png)


## Plotting the Correlation of the variables

    def correlation_plot(data: pd.DataFrame = None):
    """
    Plot the correlation matrix of the data
    :param data: The dataframe containing the data
    :return: None
    """
    corr = data.corr()
    corr.style.background_gradient(cmap='coolwarm')
    sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values, annot = True, annot_kws={'size':10})
    # Axis ticks size
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.show()
    
![corr](https://user-images.githubusercontent.com/115629197/208547607-dfa867d1-e812-44b5-858c-327a98e82911.png)


