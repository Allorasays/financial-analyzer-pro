from flask import Flask, render_template_string
import os

app = Flask(__name__)

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Financial Analyzer Pro</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            padding: 40px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        }
        .header {
            text-align: center;
            margin-bottom: 40px;
        }
        .header h1 {
            color: #333;
            font-size: 3em;
            margin-bottom: 10px;
        }
        .status {
            background: #d4edda;
            color: #155724;
            padding: 15px;
            border-radius: 8px;
            margin: 20px 0;
            text-align: center;
            font-weight: bold;
        }
        .section {
            margin: 30px 0;
        }
        .section h2 {
            color: #333;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
        }
        .market-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .market-card {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            border-left: 4px solid #667eea;
            text-align: center;
        }
        .market-card h3 {
            color: #333;
            margin-bottom: 10px;
        }
        .market-card .price {
            font-size: 1.5em;
            font-weight: bold;
            color: #28a745;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📈 Financial Analyzer Pro</h1>
            <p>Advanced Financial Analysis Platform</p>
        </div>
        
        <div class="status">
            ✅ Successfully Deployed with Flask!
        </div>
        
        <div class="section">
            <h2>📊 Market Overview</h2>
            <div class="market-grid">
                <div class="market-card">
                    <h3>S&P 500</h3>
                    <div class="price">4,500.00</div>
                    <div>+54.20 (+1.22%)</div>
                </div>
                <div class="market-card">
                    <h3>NASDAQ</h3>
                    <div class="price">14,200.00</div>
                    <div>+112.50 (+0.80%)</div>
                </div>
                <div class="market-card">
                    <h3>DOW</h3>
                    <div class="price">35,000.00</div>
                    <div>+525.00 (+1.52%)</div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>💼 Portfolio</h2>
            <p>Portfolio management features will be available in the full version.</p>
        </div>
        
        <div class="section">
            <h2>🚀 Features</h2>
            <ul>
                <li>📊 Real-time Market Data</li>
                <li>🔍 Advanced Stock Analysis</li>
                <li>📈 Interactive Charts</li>
                <li>💼 Portfolio Management</li>
                <li>⭐ Smart Watchlist</li>
                <li>🌍 Global Market Overview</li>
            </ul>
        </div>
    </div>
</body>
</html>
'''

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)


