import datetime
import webbrowser
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import folium

# import the data from the csv file
def load_data():
    df = pd.read_csv('all_month.csv')
    return df

# function to pick time, latitude, longitude
def pick_features(df):
    df = df[['time', 'latitude', 'longitude','mag']]
    return df

# function to drop rows that has null or 0 values in fields
def drop_null(df):
    df.dropna(inplace=True)
    df = df[(df.T != 0).any()]
    return df

# function to unify the number of digits for each column
def unify_digits(df):

    #reorganize the time from the time column as MM/DD/YYYY
    df['time'] = df['time'].str[5:7] + '/' + df['time'].str[8:10] + '/' + df['time'].str[0:4]
    #transfer the time as days from 1/1/2020
    df['time'] = (pd.to_datetime(df['time']) - pd.to_datetime('1/1/1970')).dt.days
    #round the latitude and longitude to the two digits after the decimal point
    df['latitude'] = df['latitude'].astype(float).round(3)
    df['longitude'] = df['longitude'].astype(float).round(3)
    return df

# function to create a new column to indicate the earthquake magnitude
def create_target(df):
    df['target'] = df['mag']
    return df

# function to split the data into training and testing data
def split_data(df):
    from sklearn.model_selection import train_test_split
    X = df[['time', 'latitude', 'longitude']]
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# function to build the model
def build_model(X_train,y_train):
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# function to evaluate the model using R2 score
def evaluate_model(model, X_test, y_test):
    from sklearn.metrics import r2_score
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    return r2

# function to save the model
def save_model(model):
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
        
# function to load the model
def load_model():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

# function to predict the earthquake occurrence and magnitude
def predict(model, time, latitude, longitude):
    # supress the warning message
    import warnings
    warnings.filterwarnings("ignore")
    return model.predict([[time, latitude, longitude]])

# function to output the result to the user
def output_result(result):
    print()
    if result > 0:
        print('---Earthquake will occur, the possible magnitude is ', result, "---")

    else:
        print('---Earthquake will not occur !---')
    print()
        
# function to predict the earthquake occurrence and magnitude for a single point
def single_predict(model, time, latitude, longitude):
    
    result = predict(model, time, latitude, longitude)
    output_result(result)
    
# function to predict the earthquake occurrence and magnitude for a period of time
def series_predict(model, period, latitude, longitude):
    
    result = []
    #every 10 days, predict the earthquake occurrence and magnitude
    dates =[]
    for i in range(period[0], period[1], 5):
        dates.append(pd.to_datetime('1/1/1970') + datetime.timedelta(days=i))
        result.append(predict(model, i, latitude, longitude))
    #print in the console in a table format, results rounded to 2 digits after the decimal point
    print()
    print('Date\t\tMagnitude')
    for i in range(len(dates)):
        print(dates[i].strftime('%m/%d/%Y'), '\t', round(result[i][0], 2))
    print()
    # visualize the result using matplotlib, x-axis is time, y-axis is magnitude, time only shows the month and day
    new_dates = []
    for i in range(len(dates)):
        new_dates.append(dates[i].strftime('%m/%d'))
    plt.plot(new_dates, result)
    plt.xlabel('date')
    plt.ylabel('magnitude')
    plt.show()
    
# function to predict the earthquake occurrence and magnitude for a region
def region_predict(model, time, left_corner, right_corner):
    
    result = []
    #pick ranmdom points in the region 
    random_latitude = np.random.randint(right_corner[0],left_corner[0], size=10)
    random_longtitude = np.random.randint(left_corner[1],right_corner[1], size=10)
    coordinates = []
    for i in range(10):
        coordinates.append([random_latitude[i], random_longtitude[i]])
        
    #visualize the result using folium
    m = folium.Map(location=[left_corner[0], left_corner[1]], zoom_start=5)
    for i in coordinates:
        result.append(predict(model, time, i[0], i[1]))
        if result[-1] > 0:
            folium.Marker(location=[i[0], i[1]], popup='possible magnitude: ' + str(result[-1]), icon=folium.Icon(color='red')).add_to(m)
    m.save('map.html')
    #open the map in the browser
    webbrowser.open('map.html')
    #print in the console in a table format
    print()
    print('Latitude\tLongitude\tMagnitude')
    for i in range(len(coordinates)):
        print(coordinates[i][0], '\t\t', coordinates[i][1], '\t\t', round(result[i][0], 2))
    
    print()
    
# main function
def main():
    
    print('''
    +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    +                                                           +
    +       Welcome to the earthquake prediction program!       +
    +                                                           +
    +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    ''')
    
    try:
        # use existing model if it exists
        print('Loading existing model...')
        model = load_model()
        
    except:
        print('No existing model, building a new model...')
        df = load_data()
        df = pick_features(df)
        df = drop_null(df)
        df = unify_digits(df)
        df = create_target(df)
        X_train, X_test, y_train, y_test = split_data(df)
        model = build_model(X_train, y_train )
        rss = evaluate_model(model, X_test, y_test)
        print('R SQURE Accuracy: ', rss)
        save_model(model)
        model = load_model()
        print('Model saved and loaded')
        
    while True:
        
        # three modes: single prediction, series prediction, and area prediction
        print("Which mode do you want to use? (N to exit)")
        print( "1.single prediction, 2.series prediction, and 3.region prediction")
        print()
        option = input('Enter mode: ')
        if option == 'N' or option == 'n':
            print('Take care!')
            break
        
        if option == '1':
            
            print('\n++++++++++ single prediction ++++++++++\n')
            time_str = str(input('Enter time : (ex : 2022-12-30)'))
            if time_str== '':
                time_str = '2022-12-30'
            #transform the time to days from 1970-01-01
            time = (datetime.datetime.strptime(time_str, '%Y-%m-%d') - datetime.datetime(1970,1,1)).days
            latitude = str(input('Enter latitude : (ex : 37.7749)'))
            if latitude == '':
                latitude = 37.7749
            longitude = str(input('Enter longitude : (ex : -122.4194)'))
            if longitude == '':
                longitude = -122.4194
            
            single_predict(model, time, latitude, longitude)
            
        elif option == '2':
            
            print('\n++++++++++ series prediction ++++++++++\n')
            print('Enter the period of time you want to predict : (YYYY-MM-DD)')
            period = [0, 0]
            start_str = str(input('Enter start time (ex : 2022-12-30): '))
            if start_str == '':
                start_str = '2022-12-30'
                end_str = '2023-01-30'
            else:
                end_str = str(input('Enter end time (ex : 2022-01-30): '))
                if end_str == '':
                    end_str = '2023-01-30'
            #transform the time to days from 1970-01-01
            print("start time: ", start_str, "-> end time: ", end_str)
            period[0] = (datetime.datetime.strptime(start_str, '%Y-%m-%d') - datetime.datetime(1970,1,1)).days
            period[1] = (datetime.datetime.strptime(end_str, '%Y-%m-%d') - datetime.datetime(1970,1,1)).days
            
            #check if the period is valid
            if period[0] > period[1]:
                print('Invalid period')
                continue
            
            latitude = input('Enter latitude(ex : 37.7749): ')
            if latitude == '':
                latitude = 37.7749
            else:
                latitude = float(latitude)
            longitude = input('Enter longitude(ex : -122.4194): ')
            if longitude == '':
                longitude = -122.4194
            else:
                longitude = float(longitude)
            series_predict(model, period, latitude, longitude)
            
        elif option == '3':
            
            print('\n++++++++++ region prediction ++++++++++\n')
            time_str = str(input('Enter time(ex : 2022-12-30): '))
            if time_str == '':
                time_str = '2022-12-30'
            time = (datetime.datetime.strptime(time_str, '%Y-%m-%d') - datetime.datetime(1970,1,1)).days
            left_corner, right_corner = [37, -115], [30, -110]
            print('Please enter longitude and latitude as integer')
            left_lat=input('Enter upper left corner latitude (ex : 37): ')
            left_lon=input('Enter upper left corner longitude (ex : -115): ')
            right_lat=input('Enter bottom right corner latitude (ex : 30): ')
            right_lon=input('Enter bottom right corner longitude (ex : -110): ')
            if left_lat != '' and left_lon != '' and right_lat != '' and right_lon != '':
                left_corner[0] = int(left_lat)
                left_corner[1] = int(left_lon)
                right_corner[0] = int(right_lat)
                right_corner[1] = int(right_lon)
            
            #check if the region is valid
            if left_corner[0] < right_corner[0] or left_corner[1] > right_corner[1]:
                print('Invalid region')
                continue
            region_predict(model, time, left_corner, right_corner)

        else:
            print('\nInvalid input\n')
            continue

if __name__ == '__main__':
    main()