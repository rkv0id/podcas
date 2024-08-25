import duckdb

# load VSS (vector similarity search) extension
duckdb.execute("install vss")
duckdb.execute("load vss")
