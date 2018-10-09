import geopandas as gpd
# from DB_CONN import *
import psycopg2
import matplotlib.pyplot as plt
con = psycopg2.connect("host='localhost' dbname='nc' user='postgres' password='' port='5444'")
cursor = con.cursor()

# plot the city of Wilmington's blocks
sql = "SELECT block.geoid10, block.geom FROM block, city WHERE ST_Intersects(block.geom, ST_Transform(city.geom, 4269)) AND city.juris = 'WM'"
df = gpd.GeoDataFrame.from_postgis(sql, con, geom_col='geom')

sql = "SELECT id, dest_type, geom FROM destinations"
dests = gpd.GeoDataFrame.from_postgis(sql, con)


# plot
fig, ax = plt.subplots()

ax.set_aspect('equal')

# plot blocks
df.plot(ax=ax, color='white', edgecolor='black')
# plot super markets
dests[dests['dest_type']=='super_market'].plot(ax=ax, marker='s', color='blue', markersize=5);
dests[dests['dest_type']=='gas_station'].plot(ax=ax, marker='o', color='red', markersize=5);


# df.plot()
plt.show()
