# FMP Upgrade Configuration Template

## Current FMP Configuration
```python
# In config.py
FMP_CONFIG = {
    'api_key': 'R9F8nfYK9yGdmiq7I5ETw7e6EhTuG8ve',  # Current free key
    'base_url': 'https://financialmodelingprep.com/api/v3',
    'rate_limit': 250,  # Free tier
    'tier': 'free'
}
```

## Upgraded FMP Configuration
```python
# In config.py (after upgrade)
FMP_CONFIG = {
    'api_key': 'YOUR_NEW_STARTER_KEY_HERE',  # New starter key
    'base_url': 'https://financialmodelingprep.com/api/v3',
    'rate_limit': 1000,  # Starter tier
    'tier': 'starter',
    'supports_realtime': True,
    'supports_advanced_financials': True,
    'supports_sec_filings': True,
    'supports_institutional_ownership': True
}
```

## New FMP Endpoints Available
```python
# Real-time data
'https://financialmodelingprep.com/api/v3/quote/AAPL?apikey={key}'

# Advanced financials
'https://financialmodelingprep.com/api/v3/income-statement/AAPL?apikey={key}'
'https://financialmodelingprep.com/api/v3/balance-sheet-statement/AAPL?apikey={key}'
'https://financialmodelingprep.com/api/v3/cash-flow-statement/AAPL?apikey={key}'

# SEC filings
'https://financialmodelingprep.com/api/v3/sec-filings/AAPL?apikey={key}'

# Institutional ownership
'https://financialmodelingprep.com/api/v3/institutional-holder/AAPL?apikey={key}'

# Key metrics
'https://financialmodelingprep.com/api/v3/key-metrics/AAPL?apikey={key}'
'https://financialmodelingprep.com/api/v3/ratios/AAPL?apikey={key}'
```



