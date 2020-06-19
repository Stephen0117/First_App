import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from spacy.lang.en.stop_words import STOP_WORDS
from gensim.parsing.preprocessing import remove_stopwords
import wordcloud
from wordcloud import WordCloud
import statsmodels.api as sm

#############################################################################################

#LOADING DATA

@st.cache
def load_data1():
    df = pd.read_csv("https://raw.githubusercontent.com/Stephen0117/First_App/master/covid.csv")
    df.set_index('Date', inplace=True)
    return df

def load_data2():
    df = pd.read_csv("https://raw.githubusercontent.com/Stephen0117/First_App/master/covid.csv")
    return df

def load_data3():
    df = pd.read_csv("https://raw.githubusercontent.com/Stephen0117/First_App/master/Data.csv")
    df.set_index('created_at', inplace=True)
    return df


#############################################################################################

    ##COVID 19 CASE COUNT ANALYSIS AND PREDICTION
def main(): 
    ##Covid 19 Case Count
    st.title("COVID 19 TRACKER")
    df = load_data1()
    country = st.selectbox('Select Country',df['Location'].unique().tolist(),1)
    
    ###########################################################################
    
    #Overall Covid 19 Country Data Table
    st.title("Covid 19 Status Overall Table")
    TDB = df[df['Location']==country]
    st.write(TDB)
    
    ##########################################################################
    
    #Specific Country Data Table
    st.title("Covid 19 Cumulative Confirmed Case Count")
    GDB = df[df['Location']==country]['Confirmed'].plot()
    st.pyplot()
    plt.close()

    ###########################################################################
    
    #DAILY RATE OF CHANGE CASE COUNT
    st.title("Covid 19 Rate of change (Included 10 Days Prediction)")
    df1 = load_data2()
    GDB1 = df1[df1['Location']==country]
    GDB1 = GDB1.reset_index(drop=True)
    GDB1['Confirmed_Delay'] = GDB1.iloc[1:,2]
    GDB1['Confirmed_Delay'] = GDB1['Confirmed_Delay'].shift(-1)
    GDB1['ROC'] = GDB1['Confirmed_Delay']-GDB1['Confirmed']
    GDB1 = GDB1[GDB1['ROC']>0]
    GDB1.set_index('Date', inplace=True)
    y2 = GDB1['ROC']
    
    #Training ARIMA Model
    mod = sm.tsa.statespace.SARIMAX(y2,
                                order=(1, 1, 1),
                                seasonal_order=(1, 1, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
    results2 = mod.fit()
    pred_uc2 = results2.get_forecast(steps=10)
    pred_ci2 = pred_uc2.conf_int()
    ax = y2.plot(label='observed', figsize=(14, 7))
    pred_uc2.predicted_mean.plot(ax=ax, label='Forecast')
    ax.fill_between(pred_ci2.index,
                pred_ci2.iloc[:, 0],
                pred_ci2.iloc[:, 1], color='k', alpha=.25)
    ax.set_xlabel('Date')
    ax.set_ylabel('Covid 19 Confirmed Cases Rate of Change')
    plt.legend()
    plt.show()
    st.pyplot()
    plt.close()
    
    ###########################################################################
    
    #Covid 19 Cumulative Deaths Case Count
    st.title("Covid 19 Cumulative Deaths Case Count")
    df[df['Location']==country]['Deaths'].plot()
    st.pyplot()
    plt.close()
    
    ###########################################################################
    
    #Covid 19 Cumulative Recovred Case Count
    st.title("Covid 19 Cumulative Recovered Case Count")
    df[df['Location']==country]['Recovered'].plot()
    st.pyplot()
    plt.close()
    
    
    #################################################################################################
    
    
    ## Tweets Sentiment Analysis
    
    ##SENTIMENT SCORES (POLARITY)
    st.title("Tweets Polarity Score (Included 10 days prediction)")
    Pdf = load_data3()
    J = Pdf['Polarity_Score'].groupby(['created_at']).mean()
    mod = sm.tsa.statespace.SARIMAX(J,
                                order=(1, 0, 0),
                                seasonal_order=(0, 0, 0, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
    results = mod.fit()
    pred_uc = results.get_forecast(steps=10)
    pred_ci = pred_uc.conf_int()
    ax = J.plot(label='observed', figsize=(12, 10))
    pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
    ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
    ax.set_xlabel('Date')
    ax.set_ylabel('Polarity_Score')
    plt.legend()
    plt.show()
    st.pyplot()
    plt.close()
    
    #################################################################################################
    
    
    ##WORDCLOUD
    tweets = load_data3()
    
    #Joining all text into one text variable
    All_Tweets = " ".join(tweets['text'])
    
    #Remove Stopwords
    All_Tweets = remove_stopwords(All_Tweets)
    
    #Remove "Covid", "Covid 19" & "Coronavirus" text because we would like to know words other than the subject
    All_Tweets = All_Tweets.replace('Covid', '')
    All_Tweets = All_Tweets.replace('covid', '')
    All_Tweets = All_Tweets.replace('covid-19', '')
    All_Tweets = All_Tweets.replace('wuhanvirus', '')
    All_Tweets = All_Tweets.replace('coronavirus', '')
    
    #Generate Word Cloud
    wc = WordCloud(max_font_size=50, max_words=10, background_color="black").generate(All_Tweets)
    # Display the generated image:
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    st.title("Tweets Wordcloud")
    st.pyplot()
    plt.close()

if __name__ == "__main__":
    main()