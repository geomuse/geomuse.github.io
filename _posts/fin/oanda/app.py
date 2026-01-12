import oandapyV20
import oandapyV20.endpoints.instruments as instruments
from oandapyV20.endpoints import accounts

API_KEY = "YOUR_API_KEY"
ACCOUNT_ID = "YOUR_ACCOUNT_ID"

client = oandapyV20.API(
    access_token=API_KEY,
    environment="practice"  # 或 "live"
)

r = accounts.AccountSummary(ACCOUNT_ID)
client.request(r)

print(r.response)

params = {
    "granularity": "M5",
    "count": 100
}

r = instruments.InstrumentsCandles(
    instrument="XAU_USD",
    params=params
)

client.request(r)

candles = r.response["candles"]

for c in candles:
    print(c["time"], c["mid"]["c"])