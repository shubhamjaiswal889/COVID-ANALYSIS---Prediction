#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing


# In[6]:


covid_cases = pd.read_csv(r'C:\Users\shubham.kj\Desktop\covid_19_india.csv')


# In[7]:


covid_vaccine = pd.read_csv(r'C:\Users\shubham.kj\Desktop\covid_vaccine_statewise.csv')


# In[8]:


=pd.read_csv(r'C:\Users\shubham.kj\Desktop\StatewiseTestingDetails.csv')


# In[9]:


covid_cases.columns
covid_cases.head(10)
covid_cases.isna().sum()
covid_cases.dtypes
covid_cases.Date.unique()[:5]


# In[10]:


covid_cases['date_parsed'] = pd.to_datetime(covid_cases.Date , format ="%Y-%m-%d")
covid_cases.dtypes
covid_cases.date_parsed.head()


# In[11]:


covid_cases['days']= covid_cases.date_parsed - min(covid_cases.date_parsed)
covid_cases.head()


# In[12]:


import plotly.graph_objs as go
import seaborn as sns
sns.set()
from plotly.offline import init_notebook_mode, iplot 


# In[13]:


#importing libraries
import scipy
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings("ignore")


# In[21]:


covid_19_India=pd.read_csv(r'C:\Users\shubham.kj\Desktop\covid_19_india.csv')
date_wise_data = covid_19_India[['State/UnionTerritory',"Date","Confirmed","Deaths","Cured"]]
date_wise_data['Date'] = date_wise_data['Date'].apply(pd.to_datetime, dayfirst=True)
date_wise_data = date_wise_data.groupby(["Date"]).sum().reset_index()
def formatted_text(string):
    display(Markdown(string))
 
import scipy
def logistic(x, L, k, x0):
    return L / (1 + np.exp(-k * (x - x0))) + 1
d_df = date_wise_data.copy()
p0 = (0,0,0)
def plot_logistic_fit_data(d_df, title, p0=p0):
    d_df = d_df.sort_values(by=['Date'], ascending=True)
    d_df['x'] = np.arange(len(d_df)) + 1
    d_df['y'] = d_df['Confirmed']

    x = d_df['x']
    y = d_df['y']

    c2 = scipy.optimize.curve_fit(logistic,  x,  y,  p0=p0 )
    #y = logistic(x, L, k, x0)
    popt, pcov = c2

    x = range(1,d_df.shape[0] + int(popt[2]))
    y_fit = logistic(x, *popt)
    
    p_df = pd.DataFrame()
    p_df['x'] = x
    p_df['y'] = y_fit.astype(int)
    
    print("Predicted maximum number of confirmed cases: " + str(int(popt[0])))
    print("Predicted growth rate: " + str(float(popt[1])))
    print("Predicted day of the inflexion: " + str(int(popt[2])) + "")

    x0 = int(popt[2])
    
    traceC = go.Scatter(
        x=d_df['x'], y=d_df['y'],
        name="Confirmed",
        marker=dict(color="brown"),
        mode = "markers+lines",
        text=d_df['Confirmed'],
    )

    traceP = go.Scatter(
        x=p_df['x'], y=p_df['y'],
        name="Predicted",
        marker=dict(color="black"),
        mode = "lines",
        text=p_df['y'],
    )
    
    trace_x0 = go.Scatter(
        x = [x0, x0], y = [0, p_df.loc[p_df['x']==x0,'y'].values[0]],
        name = "X0 - Inflexion point",
        marker=dict(color="blue"),
        mode = "lines",
        text = "X0 - Inflexion point"
    )

    data = [traceC, traceP, trace_x0]

    layout = dict(title = 'Logistic Curve Projection on Confirmed Covid-19 Cases in India',
          xaxis = dict(title = 'Number of days since 30 January 2020', showticklabels=True), 
          yaxis = dict(title = 'Number of confirmed Covid-19 cases'),
          hovermode = 'closest',plot_bgcolor='rgb(275, 270, 273)'
         )
    fig = dict(data=data, layout=layout)
    iplot(fig, filename='covid-logistic-forecast')
    
L = 250000
k = 0.25
x0 = 100
p0 = (L, k, x0)
plot_logistic_fit_data(d_df, 'India') 


# In[20]:


covid_19_India = pd.read_csv(r'C:\Users\shubham.kj\Desktop\covid_19_india.csv')
date_wise_data = covid_19_India[['State/UnionTerritory',"Date","Confirmed","Deaths","Cured"]]
date_wise_data['Date'] = date_wise_data['Date'].apply(pd.to_datetime, dayfirst=True)
date_wise_data = date_wise_data.groupby(["Date"]).sum().reset_index()
def formatted_text(string):
    display(Markdown(string))
 
import scipy
def logistic(x, L, k, x0):
    return L / (1 + np.exp(-k * (x - x0))) + 1
d_df = date_wise_data.copy()
p0 = (0,0,0)
def plot_logistic_fit_data(d_df, title, p0=p0):
    d_df = d_df.sort_values(by=['Date'], ascending=True)
    d_df['x'] = np.arange(len(d_df)) + 1
    d_df['y'] = d_df['Cured']

    x = d_df['x']
    y = d_df['y']

    c2 = scipy.optimize.curve_fit(logistic,  x,  y,  p0=p0 )
    #y = logistic(x, L, k, x0)
    popt, pcov = c2

    x = range(1,d_df.shape[0] + int(popt[2]))
    y_fit = logistic(x, *popt)
    
    p_df = pd.DataFrame()
    p_df['x'] = x
    p_df['y'] = y_fit.astype(int)
    
    print("Predicted maximum number of Cured Covid-19 cases: " + str(int(popt[0])))
    print("Predicted growth rate: " + str(float(popt[1])))
    print("Predicted day of the inflexion: " + str(int(popt[2])) + "")

    x0 = int(popt[2])
    
    traceC = go.Scatter(
        x=d_df['x'], y=d_df['y'],
        name="Cured",
        marker=dict(color="Green"),
        mode = "markers+lines",
        text=d_df['Cured'],
    )

    traceP = go.Scatter(
        x=p_df['x'], y=p_df['y'],
        name="Predicted",
        marker=dict(color="black"),
        mode = "lines",
        text=p_df['y'],
    )
    
    trace_x0 = go.Scatter(
        x = [x0, x0], y = [0, p_df.loc[p_df['x']==x0,'y'].values[0]],
        name = "X0 - Inflexion point",
        marker=dict(color="blue"),
        mode = "lines",
        text = "X0 - Inflexion point"
    )

    data = [traceC, traceP, trace_x0]

    layout = dict(title = 'Logistic Curve Projection on Cured Covid Cases in India',
          xaxis = dict(title = 'Number of days since 30 January 2020,', showticklabels=True), 
          yaxis = dict(title = 'Number of cases'),
          hovermode = 'closest',plot_bgcolor='rgb(275, 270, 273)'
         )
    fig = dict(data=data, layout=layout)
    iplot(fig, filename='covid-logistic-forecast')
    
L = 250000
k = 0.25
x0 = 100
p0 = (L, k, x0)
plot_logistic_fit_data(d_df, 'India')


# In[16]:


covid_19_India = pd.read_csv(r'C:\Users\shubham.kj\Desktop\covid_19_india.csv')
date_wise_data = covid_19_India[['State/UnionTerritory',"Date","Confirmed","Deaths","Cured"]]
date_wise_data['Date'] = date_wise_data['Date'].apply(pd.to_datetime, dayfirst=True)
date_wise_data = date_wise_data.groupby(["Date"]).sum().reset_index()
def formatted_text(string):
    display(Markdown(string))
 
import scipy
def logistic(x, L, k, x0):
    return L / (1 + np.exp(-k * (x - x0))) + 1
d_df = date_wise_data.copy()
p0 = (0,0,0)
def plot_logistic_fit_data(d_df, title, p0=p0):
    d_df = d_df.sort_values(by=['Date'], ascending=True)
    d_df['x'] = np.arange(len(d_df)) + 1
    d_df['y'] = d_df['Deaths']

    x = d_df['x']
    y = d_df['y']

    c2 = scipy.optimize.curve_fit(logistic,  x,  y,  p0=p0 )
    #y = logistic(x, L, k, x0)
    popt, pcov = c2

    x = range(1,d_df.shape[0] + int(popt[2]))
    y_fit = logistic(x, *popt)
    
    p_df = pd.DataFrame()
    p_df['x'] = x
    p_df['y'] = y_fit.astype(int)
    
    print("Predicted maximum number of deaths from Covid-19: " + str(int(popt[0])))
    print("Predicted growth rate: " + str(float(popt[1])))
    print("Predicted day of the inflexion: " + str(int(popt[2])) + "")

    x0 = int(popt[2])
    
    traceC = go.Scatter(
        x=d_df['x'], y=d_df['y'],
        name="Deaths",
        marker=dict(color="Red"),
        mode = "markers+lines",
        text=d_df['Cured'],
    )

    traceP = go.Scatter(
        x=p_df['x'], y=p_df['y'],
        name="Predicted",
        marker=dict(color="blue"),
        mode = "lines",
        text=p_df['y'],
    )
    
    trace_x0 = go.Scatter(
        x = [x0, x0], y = [0, p_df.loc[p_df['x']==x0,'y'].values[0]],
        name = "X0 - Inflexion point",
        marker=dict(color="black"),
        mode = "lines",
        text = "X0 - Inflexion point"
    )

    data = [traceC, traceP, trace_x0]

    layout = dict(title = 'Logistic Curve Projection on Covid-19 Deaths in India',
          xaxis = dict(title = 'Number of days since 30 January 2020', showticklabels=True), 
          yaxis = dict(title = 'Number of cases'),
          hovermode = 'closest',plot_bgcolor='rgb(275, 270, 273)'
         )
    fig = dict(data=data, layout=layout)
    iplot(fig, filename='covid-logistic-forecast')
    
L = 10000
k = 0.25
x0 = 100
p0 = (L, k, x0)
plot_logistic_fit_data(d_df, 'India')


# In[ ]:




