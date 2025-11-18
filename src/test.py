# src/pull_ercot_data.py

import os
from dotenv import load_dotenv
from gridstatus import Ercot, Markets
from gridstatusio import GridStatusClient

# Load .env
load_dotenv()

API_KEY = os.getenv("GRIDSTATUS_API_KEY")
if API_KEY is None:
    raise RuntimeError("GRIDSTATUS_API_KEY not found in environment variables.")

client = GridStatusClient(api_key=API_KEY)
grid = Ercot()
QUERY_LIMIT = 100
data_utc = client.get_dataset(
    dataset="ercot_spp_day_ahead_hourly",
    start="2023-04-01",
    end="2023-04-03",
    limit=QUERY_LIMIT,
)

data_utc

print(data_utc.head())