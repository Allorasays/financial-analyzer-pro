# For Heroku deployment
web: streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true

# For separate API and Web services
api: python proxy.py
web: streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true

