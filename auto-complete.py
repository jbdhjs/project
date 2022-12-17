#housing rental recommendation program
'''
1.Introduction:
This project is to build a housing rental recomendation
program for the users who are looking for a rental house in Pittsburgh.
While there are already many websites that provide the information, very little 
of them are specific enough for the users. The users are usually overwhelmed by the
information and they have to spend a lot of time to find the house that they want.
This project is aimed to solve this problem by providing a recommendation program
that quantifies the preferences of the users and recommends the houses in a ranked
order.

2. Data:
The data is from the website: https://www.zillow.com/
parameters: 
zid
statusType
statusText
timeOnZillow
price
pricePerSqFt
zestimate
zestimatePerSqFt
rentZestimate
area
lotSize
lotAreaUnit
beds
baths
address
addressStreet
addressCity
addressState
addressZipcode
latitude
longitude
brokerName
isZillowOwned
detailUrl
image	
sourceUrl

#2 Problem Definition
The problem is people are overwhelmed by the information of the rental houses when 
searching on internet , somtimes the filter options are unclear, and the users 
to spend a lot of time to find the rental house that they want. And also for 
business purpose, the website prioritizes the house of which the owner pays the most. 
However, the users are not aware of this and they may not get the best house for them.

#3 Background
The project is inspired when I was looking for a rental house in Pittsburgh. I found
that there are many websites that provide the information of the rental houses, but
very little of them are specific.
For example, the website: https://www.zillow.com/. User can filter the
houses by the price, the number of bedrooms, and the number of bathrooms ..., but after that 
it gives choices in a somehow random order. At least not in a obvious order.
To solve this problem, I want to build a recommendation program that quantifies the
preferences of the users and recommends the houses in a ranked order.People can configure
by setting the weights of the parameters that they care about, increasing or decreasing the
weights in round s and finally get the houses that they want.


#4 Solution Method
The data is from the website: https://www.zillow.com/ 
I choose the all rental houses and apartments in Pittsburgh. Using broswer plugin,
the data is retrived and saved in csv file. The data is at the day of 2022-12-0,
The raw data includes 6,000+ rows and 26 parameters:


The useful parameters when looking for a rental house are:


 The program will rank
the houses based on the weights and the parameters. The users can also choose
the number of houses that they want to see. The program will return the houses
in a ranked order. 

The parameters are used to quantify the preferences of the users. Every parameter
has a weight. The weights are used to quantify the preferences of the users.
The users can choose the weights of the parameters. A normalized score is provided for each house in order to
make the comparison easier.The program will rank the houses based on the scores.
Thre are minimun maximum thresholds for each parameter. They are set by the users, so that they can
avoid the extreme cases. 
the algorithm is as follows:
In a loop, for each house, the program will first check if any of the parameters
is out of the threshold. If so, the score of the house will be set to 0. If not,
the score of the house will be calculated by the following formula:
score = Sum((parameter value - minimum threshold) / (maximum threshold - minimum threshold) * weight)



#5.	What need to learn to execute your project
To build the program, I need to learn how to use the python to read the data from csv file,
and clean the data since the dat contains some missing values,for example, some houses do not
have the size number. This mainly focus on the DataFrames in pandas. I also need to learn how to
use the python to build a GUI to make the program more user-friendly for input and output.

#6.	What to deliver
The expected outcome includes a user interface that allows the users to input the weights of the parameters
and the number of houses that they want to see. The program will return the houses in a ranked order maybe 
also including the pictures of the houses, since the url of the pictures are already included in the data.



'''


# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk

#function to read the data from csv file
def read_data():
    #read the data from csv file absolute path
    df = pd.read_csv('pitts.csv')
    return df

#function to clean the data
def clean_data(df):
    #choose the useful columns, parameters: statusType, price, area, beds, baths, address, addressStreet
    df = df[['statusType','price','area','beds','baths','address','addressStreet']]
    #low case the address, statusType, addressStreet
    df['address'] = df['address'].str.lower()
    df['statusType'] = df['statusType'].str.lower()
    df['addressStreet'] = df['addressStreet'].str.lower()
    #drop the rows that have missing values
    df.dropna(inplace=True)
    #reset the index
    df.reset_index(drop=True,inplace=True)
    #add a new column 'score' and set it to 0
    df['score'] = 0
    return df
    


#function to calculate the score of each house according to the weights given by the users
def calculate_score(df,weights,thresholds,numberOfHouses):
    #create a new column 'score' and set it to 0
    df['score'] = 0
    #loop through the rows
    for i in range(len(df)):
        #check if the house is out of the threshold
        if df['price'][i] < thresholds['priceMin'][i] or df['price'][i] > thresholds['priceMax'][i] or df['area'][i] < thresholds['areaMin'][i] or df['area'][i] > thresholds['areaMax'][i]:
            #set the score to 0
            df['score'][i] = 0
            continue
        #check if the address contains the keywords
        if df['address'][i].find(weights['address']) == -1:
            #set the score to 0
            df['score'][i] = 0
            continue
        #if type is not all, check if the type is the same
        if weights['statusType'] != 'all' and df['statusType'][i] != weights['statusType']:
            #set the score to 0
            df['score'][i] = 0
            continue
        #calculate the score
        df['score'][i] = (df['price'][i] - thresholds['priceMin'][i]) / (thresholds['priceMax'][i] - thresholds['priceMin'][i]) * weights['price'] 
        + (df['area'][i] - thresholds['areaMin'][i]) / (thresholds['areaMax'][i] - thresholds['areaMin'][i]) * weights['area']
        + df['beds'][i] * weights['beds'] 
        + df['baths'][i] * weights['baths']

    #sort the dataframe by the score
    df.sort_values(by=['score'],ascending=False,inplace=True)
    # pick the top numberOfHouses houses
    df = df.head(numberOfHouses)
    #reset the index
    df.reset_index(drop=True,inplace=True)
    return df

        
#main function
def main():
    #process the data
    df = read_data()
    df = clean_data(df)
    #GUI window
    window = tk.Tk()
    window.title('Housing Rental Recommendation Program')
    window.geometry('500x300')
    #label
    tk.Label(window,text='Welcome to the Housing Rental Recommendation Program',font=('Arial',12)).pack()
    tk.Label(window,text='Please set the the parameters',font=('Arial',12)).pack()
    tk.Label(window,text='Please set the number of houses that you want to see',font=('Arial',12)).pack()
    #parameters: statusType, price, pricePerSqFt, area, beds, baths, address, addressStreet
    #parameter map
    weights = {'statusType':0,'price':0,'area':0,'beds':0,'baths':0,'address':0,'addressStreet':0, 'priceMin':0,'priceMax':0,'areaMin':0,'areaMax':0}
    threshold = {'priceMin':0,'priceMax':0,'areaMin':0,'areaMax':0}
    #input for each parameter

    #dropdown menu for statusType
    tk.Label(window,text='statusType',font=('Arial',12)).pack()
    var = tk.StringVar()
    var.set('all')
    weights['statusType'] = tk.OptionMenu(window,var,'all','house for rent','apartment for rent').pack()
    #price range
    weights['price']=tk.Entry(window).pack()
    tk.Label(window,text='price range',font=('Arial',12)).pack()
    tk.Label(window,text='minimum',font=('Arial',12)).pack()
    threshold['priceMin']  = tk.Entry(window,show=None).pack()
    tk.Label(window,text='maximum',font=('Arial',12)).pack()
    threshold['priceMax']  = tk.Entry(window,show=None).pack()
    #area range
    weights['area']=tk.Entry(window,show=None).pack()
    tk.Label(window,text='area range',font=('Arial',12)).pack()
    tk.Label(window,text='minimum',font=('Arial',12)).pack()
    threshold['areaMin'] = tk.Entry(window,show=None).pack()
    tk.Label(window,text='maximum',font=('Arial',12)).pack()
    threshold['areaMax'] = tk.Entry(window,show=None).pack()

    weights['beds']=tk.Entry(window,show=None).pack()
    weights['baths']=tk.Entry(window,show=None).pack()
    weights['address']=tk.Entry(window,show=None).pack()
    weights['addressStreet']=tk.Entry(window,show=None).pack()

    #input box for the number of houses
    number=tk.Entry(window,show=None).pack()
    #button
    tk.Button(window,text='submit',width=10,height=1,command=lambda:calculate_score(df,weights,threshold,number)).pack()
    
    #button to quit the GUI
    tk.Button(window,text='quit',width=10,height=1,command=window.quit).pack()
    window.mainloop()


if __name__ == '__main__':
    main()