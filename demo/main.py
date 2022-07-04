from re import S
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#Train Model
df = pd.read_csv('Mall_Customers.csv')
Xtrain = df.copy()
dummies = pd.get_dummies(df['Gender'])
Xtrain[dummies.columns] = dummies
Xtrain = Xtrain.drop(['CustomerID', 'Gender'], axis=1)

model = KMeans(n_clusters = 5)
model.fit(Xtrain)

Clusters = model.predict(Xtrain)
Clusters = pd.DataFrame(data = Clusters, columns = ['Cluster'])
Clusters['CustomerID'] = df.CustomerID

final_data = df.merge(Clusters, on = 'CustomerID', how = 'left')
centers = pd.DataFrame(data = model.cluster_centers_, columns = Xtrain.columns)


#User Interface
st.title('Identify Customer Group')

data = {'CustomerID': 0, 'Gender': 'null', 'Age':0 ,'Annual Income (k$)':0, 'Spending Score (1-100)': 1}

data['CustomerID'] = st.number_input('CustomerID (must be a number)', min_value=0)
data['Gender'] = st.selectbox('Gender', ['Male', 'Female'])
data['Age'] = st.slider('Age', min_value=12, max_value=100, step=1)
data['Annual Income (k$)'] = st.number_input('Annual Income (k$)', min_value=0)
data['Spending Score (1-100)'] = st.number_input('Spending Score (1-100)', min_value=0)

test = pd.DataFrame(data, index = [0])
test['Female'] = 0
test['Male'] = 0


#Predictor
operation_predict = st.button('Predict Group')

def encode():
    if (data['Gender'] == 'Female'):
        test['Female'] = 1

    else :
        test['Male'] = 1

    return test.drop(['Gender', 'CustomerID'], axis = 1)

cluster_priority = {0: 3, 1: 2, 2: 1, 3: 4, 4: 5}
priority_vs_category = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E'}

if (operation_predict):
    X_test = encode()
    test['Cluster'] = model.predict(X_test)
    to_show = f"Customer with CustomerID {data['CustomerID']} belongs to cluster {test['Cluster'][0]}, is {cluster_priority[test['Cluster'][0]]} priority customer (category {priority_vs_category[cluster_priority[test['Cluster'][0]]]} customer)."
    st.markdown(to_show)


#Plot Data
operation_plot = st.button('Plot Clusters')
if (operation_plot):
    #Getting unique labels
    u_labels = final_data['Cluster'].unique()
    fig = plt.figure()

    #plotting the results with any two cols on x and y-axis:
    for i in u_labels:
        x = final_data[final_data.Cluster == i]['Annual Income (k$)']
        y = final_data[final_data.Cluster == i]['Spending Score (1-100)']

        plt.xlabel('Annual Income (k$)')
        plt.ylabel('Spending Score (1-100)')

        plt.scatter( x,  y, label = i)

    #plot centers
    plt.scatter(centers['Annual Income (k$)'], centers['Spending Score (1-100)'], color = 'black')

    #Plot test data point
    plt.plot(test['Annual Income (k$)'], test['Spending Score (1-100)'],  marker = 'o', color = 'yellow', markersize = 20)
    plt.annotate('Data Point', (test['Annual Income (k$)'], test['Spending Score (1-100)']))

    plt.legend()
    st.pyplot(fig)