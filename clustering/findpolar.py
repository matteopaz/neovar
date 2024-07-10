from hpgeom import hpgeom as h

# find all order 5 pixels that include points 0.5 degrees away from (90, 66.54) or (-90, 66.54)

import numpy as np
from hpgeom import hpgeom as h

def find_pixels(lat, lon, order=5, distance_deg=0.5):
    # Convert latitude and longitude from degrees to radians
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    
    # Convert distance from degrees to radians
    distance_rad = np.radians(distance_deg)
    
    # Find the pixel index for the given point at the specified order
    pixel_index = h.angle_to_pixel(lon_rad, lat_rad, 2**order)
    
    # Find all neighboring pixels
    neighbors = h.neighbours(pixel_index, 2**order)
    
    # Initialize a list to hold pixels within the specified distance
    pixels_within_distance = []
    
    # Check each neighbor (and the original pixel) for distance
    for neighbor in np.append(neighbors, pixel_index):
        # Get the center point of the neighbor pixel
        neighbor_lat, neighbor_lon = h.healpix_to_lonlat(neighbor, 2**order)
        
        # Calculate the angular distance to the original point
        ang_distance = h.angular_distance(lon_rad, lat_rad, neighbor_lon, neighbor_lat)
        
        # If the distance is within the specified limit, add to the list
        if ang_distance <= distance_rad:
            pixels_within_distance.append(neighbor)
    
    return pixels_within_distance

# Example usage for the points (90, 66.54) and (-90, 66.54)
pixels_1 = find_pixels(66.54, 90)
pixels_2 = find_pixels(66.54, -90)

print("Pixels near (90, 66.54):", pixels_1)
print("Pixels near (-90, 66.54):", pixels_2)

