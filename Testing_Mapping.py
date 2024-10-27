import numpy as np
import pandas as pd
from geopy.distance import geodesic
import folium

def main():
    # Load the dataset
    scats_data = '/Users/marleywetini/repos/intelligentSystems/data/Scats Data October 2006.xlsx'
    scats_data = pd.read_excel(scats_data, header=1)
    
    # Extract SCATS data and drop duplicates
    scats_coordinates = scats_data[['Location', 'SCATS Number', 'NB_LATITUDE', 'NB_LONGITUDE']].drop_duplicates()
    scats_coordinates.dropna(subset=['NB_LATITUDE', 'NB_LONGITUDE'], inplace=True)

    # Function to derive direction from the "Location" column
    def extract_direction(location):
        if ' N ' in location or location.endswith(' N'):
            return 'N'
        elif ' S ' in location or location.endswith(' S'):
            return 'S'
        elif ' E ' in location or location.endswith(' E'):
            return 'E'
        elif ' W ' in location or location.endswith(' W'):
            return 'W'
        else:
            return 'Unknown'

    # Apply the function to create a "Direction" column
    scats_coordinates['Direction'] = scats_coordinates['Location'].apply(extract_direction)

    # Filter out rows where direction couldn't be determined
    scats_coordinates = scats_coordinates[scats_coordinates['Direction'] != 'Unknown']

    # Function to calculate distance between two points (in kilometers)
    def haversine_distance(coord1, coord2):
        return geodesic(coord1, coord2).kilometers

    # Set proximity threshold (1 km)
    threshold_km = 1

    # List to store SCATS pairs that are within the threshold and follow directional rules
    scats_neighbors = []

    # Set to track already connected pairs
    connected_pairs = set()

    # Loop through each SCATS and calculate distance to every other SCATS
    for i in range(len(scats_coordinates)):
        scats_a = scats_coordinates.iloc[i]
        coord_a = (scats_a['NB_LATITUDE'], scats_a['NB_LONGITUDE'])

        for j in range(i + 1, len(scats_coordinates)):
            scats_b = scats_coordinates.iloc[j]
            coord_b = (scats_b['NB_LATITUDE'], scats_b['NB_LONGITUDE'])

            # Calculate distance between SCATS A and SCATS B
            distance = haversine_distance(coord_a, coord_b)

            # Check if the distance is within the threshold and ensure correct directional connection
            if distance <= threshold_km:
                # North to South (scats_a should be north of scats_b)
                if scats_a['Direction'] == 'N' and scats_b['Direction'] == 'S' and scats_a['NB_LATITUDE'] > scats_b['NB_LATITUDE'] and scats_a['NB_LONGITUDE'] - scats_b['NB_LONGITUDE'] :
                    pair = tuple(sorted((scats_a['SCATS Number'], scats_b['SCATS Number'])))
                    if pair not in connected_pairs:
                        scats_neighbors.append({
                            'Location_A': scats_a['Location'],
                            'Location_B': scats_b['Location'],
                            'Direction': f"{scats_a['Direction']} -> {scats_b['Direction']}",
                            'Distance (km)': distance
                        })
                        connected_pairs.add(pair)

                # South to North (scats_a should be south of scats_b)
                elif scats_a['Direction'] == 'S' and scats_b['Direction'] == 'N' and scats_a['NB_LATITUDE'] < scats_b['NB_LATITUDE']:
                    pair = tuple(sorted((scats_a['SCATS Number'], scats_b['SCATS Number'])))
                    if pair not in connected_pairs:
                        scats_neighbors.append({
                            'Location_A': scats_a['Location'],
                            'Location_B': scats_b['Location'],
                            'Direction': f"{scats_a['Direction']} -> {scats_b['Direction']}",
                            'Distance (km)': distance
                        })
                        connected_pairs.add(pair)

                # East to West (scats_a should be east of scats_b)
                elif scats_a['Direction'] == 'E' and scats_b['Direction'] == 'W' and scats_a['NB_LONGITUDE'] > scats_b['NB_LONGITUDE']:
                    pair = tuple(sorted((scats_a['SCATS Number'], scats_b['SCATS Number'])))
                    if pair not in connected_pairs:
                        scats_neighbors.append({
                            'Location_A': scats_a['Location'],
                            'Location_B': scats_b['Location'],
                            'Direction': f"{scats_a['Direction']} -> {scats_b['Direction']}",
                            'Distance (km)': distance
                        })
                        connected_pairs.add(pair)

                # West to East (scats_a should be west of scats_b)
                elif scats_a['Direction'] == 'W' and scats_b['Direction'] == 'E' and scats_a['NB_LONGITUDE'] < scats_b['NB_LONGITUDE']:
                    pair = tuple(sorted((scats_a['SCATS Number'], scats_b['SCATS Number'])))
                    if pair not in connected_pairs:
                        scats_neighbors.append({
                            'Location_A': scats_a['Location'],
                            'Location_B': scats_b['Location'],
                            'Direction': f"{scats_a['Direction']} -> {scats_b['Direction']}",
                            'Distance (km)': distance
                        })
                        connected_pairs.add(pair)

    # Convert the list to a dataframe to see neighboring SCATS
    scats_neighbors_df = pd.DataFrame(scats_neighbors)

    # Display the neighboring SCATS
    print(scats_neighbors_df)
    scats_neighbors_df.to_csv('scats_neighbors.csv', index=False)

    # Create a map centered around the mean location of SCATS points
    m = folium.Map(location=[scats_coordinates['NB_LATITUDE'].mean(), scats_coordinates['NB_LONGITUDE'].mean()], zoom_start=12)

    # Add SCATS locations as markers, including SCATS number and direction in the popup
    for i, row in scats_coordinates.iterrows():
        folium.Marker(
            [row['NB_LATITUDE'], row['NB_LONGITUDE']],
            popup=f'SCATS {row["Location"]} - Number: {row["SCATS Number"]}, Direction: {row["Direction"]}'
        ).add_to(m)

    # Add lines between neighboring SCATS with directional consideration
    for i, row in scats_neighbors_df.iterrows():
        scats_a = scats_coordinates[(scats_coordinates['Location'] == row['Location_A']) & (scats_coordinates['Direction'] in row['Direction'].split(" -> "))].iloc[0]
        scats_b = scats_coordinates[(scats_coordinates['Location'] == row['Location_B']) & (scats_coordinates['Direction'] in row['Direction'].split(" -> "))].iloc[0]
        folium.PolyLine(locations=[
            [scats_a['NB_LATITUDE'], scats_a['NB_LONGITUDE']],
            [scats_b['NB_LATITUDE'], scats_b['NB_LONGITUDE']]
        ], color='blue').add_to(m)

    # Save the map to an HTML file
    m.save('scats_map.html')

if __name__ == '__main__':
    main()
