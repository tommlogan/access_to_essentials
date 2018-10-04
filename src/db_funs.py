'''
Here the db tables are initialized for the analysis
'''

srid = 2264

from sqlalchemy.engine import create_engine
import psycopg2
import pandas as pd
import os.path
import osgeo.ogr

engine = create_engine('postgresql+psycopg2://postgres:@localhost/nc?port=5444') ### IMPORTANT!!!
connection = engine.connect()
engine.table_names()





query = "DROP TABLE demograph;"


# add csv to nc.demograph
df = pd.read_csv('B:/research/resilience/data/nhgis/nhgis0030_ds172_2010_block.csv')
df = df[df.COUNTYA==129]
df.to_sql('demograph', engine)
# set index on gisjoin
connection.close()

con = psycopg2.connect("host='localhost' dbname='nc' user='postgres' password='' port='5444'")
cursor = con.cursor()
queries = ['CREATE INDEX "GISJOIN" ON demograph ("GISJOIN");',
        'CREATE INDEX "id" ON demograph ("BLOCKA");']
for q in queries:
    cursor.execute(q)

conn.commit()
#
# cursor.execute("Select * FROM pg_indexes WHERE tablename = 'demograph'")
# cursor.fetchall()
#
# cursor.execute("Select * FROM demograph")
# # cursor.fetchmany(5)
# colnames = [desc[0] for desc in cursor.description]

# conn.close()

# add shp
srcFile = os.path.join("data", "nhgis","NC_block_2010.shp")
shapefile = osgeo.ogr.Open(srcFile)
layer = shapefile.GetLayer(0)
for i in range(layer.GetFeatureCount()):
    feature = layer.GetFeature(i)
    name = feature.GetField("NAME").decode("Latin-1")
    wkt = feature.GetGeometryRef().ExportToWkt()
    cursor.execute("INSERT INTO countries (name,outline) " +"VALUES (%s, ST_GeometryFromText(%s, " +"4326))", (name.encode("utf8"), wkt))

connection.commit()





queries = ['CREATE INDEX "block_geom_gist" ON block USING GIST ("geom");']
for q in queries:
    cursor.execute(q)

conn.commit()
