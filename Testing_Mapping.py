import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from geopy.distance import geodesic
import folium

def main():
    lags = 12
    #Retrieve data from dataset and put into

    scats_data = '/Users/marleywetini/repos/intelligentSystems/data/Scats Data October 2006.xlsx'


    scats_data = pd.read_excel(scats_data, header=1).fillna(0)
    scats_coordinates = scats_data[['Location', 'NB_LATITUDE', 'NB_LONGITUDE']].drop_duplicates()

    # Function to calculate distance between two points (in kilometers)
    def haversine_distance(coord1, coord2):
        return geodesic(coord1, coord2).kilometers

    # Set proximity threshold (1 km)
    threshold_km = 1

    # List to store SCATS pairs that are within the threshold
    scats_neighbors = []

    # Loop through each SCATS and calculate distance to every other SCATS
    for i in range(len(scats_coordinates)):
        scats_a = scats_coordinates.iloc[i]
        coord_a = (scats_a['NB_LATITUDE'], scats_a['NB_LONGITUDE'])

        for j in range(i + 1, len(scats_coordinates)):
            scats_b = scats_coordinates.iloc[j]
            coord_b = (scats_b['NB_LATITUDE'], scats_b['NB_LONGITUDE'])

            # Calculate distance between SCATS A and SCATS B
            distance = haversine_distance(coord_a, coord_b)

            # Check if the distance is within the threshold and that lat or long is not the same
            if distance <= threshold_km:
                if -0.005 <= (scats_a['NB_LATITUDE'] - scats_b['NB_LATITUDE']) <= 0.005 or -0.005 <= (scats_a['NB_LONGITUDE'] - scats_b['NB_LONGITUDE']) <= 0.005:
                    continue
                scats_neighbors.append({
                    'Location_A': scats_a['Location'],
                    'Location_B': scats_b['Location'],
                    'Distance (km)': distance
                })

    # Convert the list to a dataframe to see neighboring SCATS
    scats_neighbors_df = pd.DataFrame(scats_neighbors)

    # Display the neighboring SCATS
    print(scats_neighbors_df)
    scats_neighbors_df.to_csv('scats_neighbors.csv', index=False)

# Create a map centered around the first SCATS location
    m = folium.Map(location=[scats_coordinates['NB_LATITUDE'].mean(), scats_coordinates['NB_LONGITUDE'].mean()], zoom_start=12)

    # Add SCATS locations as markers, including SCATS number in the popup
    for i, row in scats_coordinates.iterrows():
        folium.Marker(
            [row['NB_LATITUDE'], row['NB_LONGITUDE']],
            popup=f'SCATS {row["Location"]} - Number: {row["SCATS Number"]}'
        ).add_to(m)

    # Add lines between neighboring SCATS
    for i, row in scats_neighbors_df.iterrows():
        scats_a = scats_coordinates[scats_coordinates['Location'] == row['Location_A']].iloc[0]
        scats_b = scats_coordinates[scats_coordinates['Location'] == row['Location_B']].iloc[0]
        folium.PolyLine(locations=[
            [scats_a['NB_LATITUDE'], scats_a['NB_LONGITUDE']],
            [scats_b['NB_LATITUDE'], scats_b['NB_LONGITUDE']]
        ], color='blue').add_to(m)

    m.save('scats_map.html')

# Display the map

    # Convert 'Date' column to datetime for proper manipulation
    train_data['Date'] = pd.to_datetime(train_data['Date'], format='%d/%m/%Y')
    test_data['Date'] = pd.to_datetime(test_data['Date'], format='%d/%m/%Y')

    # Function to create lag features
    def create_lags(df, lags):
        for i in range(1, lags + 1):
            # For each traffic volume column, create a lagged version
            lagged_cols = df.filter(like='V').columns
            for col in lagged_cols:
                df[f"{col}_lag{i}"] = df[col].shift(i)
        return df
    
    # Apply lagging to the traffic data
    train_data = create_lags(train_data, lags)
    test_data = create_lags(test_data, lags)
    
    # Select only the necessary columns: Date, coordinates, and traffic data (with lags)
    necessary_columns = ['Date', 'NB_LATITUDE', 'NB_LONGITUDE'] + train_data.filter(like='V').columns.tolist() + train_data.filter(like='lag').columns.tolist()

    train_data = train_data[necessary_columns].dropna()  # Drop rows with NaN due to lagging
    test_data = test_data[necessary_columns].dropna()
    

    
if __name__ == '__main__':
    main()
