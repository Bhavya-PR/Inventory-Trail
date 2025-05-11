import pandas as pd
import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import xgboost as xgb
import matplotlib.pyplot as plt
import warnings
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import requests
import feedparser
import ssl


warnings.filterwarnings("ignore", category=DeprecationWarning)


# Function to fetch stock data
@st.cache_data
def get_stock_data(ticker, start_date, end_date):
    stock = yf.download(ticker, start=start_date, end=end_date)
    return stock


# Ensure that this is placed at the beginning of your code with your function definitions
def final_probability(aco_prob, ml_prob, financials_passed, sentiment_prob):
    # Calculate the financials effect based on whether they pass or fail
    financials_weight = 0.20 if financials_passed else -0.05
    
    # Calculate final probability with weights
    final_prob = (0.30 * aco_prob) + (0.30 * ml_prob) + (financials_weight) + (0.20 * sentiment_prob)
    
    return final_prob

# Then, find the place where the ACO, ML, and sentiment probabilities are calculated
# For example, after those calculations:

aco_prob = 0.65  # Replace with your ACO model probability
ml_prob = 0.70   # Replace with your Random Forest or ML model probability
financials_passed = True  # Set whether the financials conditions pass or fail
sentiment_prob = 0.60  # Sentiment analysis probability

# Finally, calculate the final probability using the function
final_prob = final_probability(aco_prob, ml_prob, financials_passed, sentiment_prob)
# print(f"Final Probability: {final_prob}")

# Continue with your rest of the code...

# Function to create features for crash prediction
def create_features(data):
    data['SMA_50'] = data['Close'].rolling(window=50).mean()  # 50-day Simple Moving Average
    data['SMA_200'] = data['Close'].rolling(window=200).mean()  # 200-day Simple Moving Average
    data['Price_Change'] = data['Close'].pct_change()  # Daily price change
    data['Market_Crash'] = np.where(data['Price_Change'] < -0.05, 1, 0)  # Market crash if price drops more than 5%
    return data



# Function to train a RandomForest Classifier for stock direction prediction
def train_ml_model(stock_data):
    stock_data['Price_Change_Direction'] = np.where(stock_data['Price_Change'] > 0, 1, 0)  # 1 for up, 0 for down
    X = np.array(range(len(stock_data))).reshape(-1, 1)
    y = stock_data['Price_Change_Direction'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()


# Function to calculate sentiment score from news headlines using VADER
def get_sentiment(text):
    sentiment = analyzer.polarity_scores(text)
    return sentiment['compound']  # Return the compound score



# Function to calculate crash probability using XGBoost
def calculate_crash_probability(stock_data):
    stock_data = create_features(stock_data)
    stock_data = stock_data.dropna()  # Remove rows with NaN values


    X = stock_data[['SMA_50', 'SMA_200', 'Price_Change']]
    y = stock_data['Market_Crash']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBClassifier()
    model.fit(X_train, y_train)


    # Predicting crash probability in the next period using feature names
    latest_data = pd.DataFrame([stock_data[['SMA_50', 'SMA_200', 'Price_Change']].iloc[-1]])
    predicted_probabilities = model.predict_proba(latest_data)


    crash_probability = predicted_probabilities[0][1] * 100  # Probability of class 1 (market crash)
    return crash_probability


# Check function example
def Check_fetch_metadata():
    metadata = {
        "source": "Yahoo Finance",
        "frequency": "daily",
        "fields": ["Open", "High", "Low", "Close", "Volume", "Close"]
    }
    # print("Metadata for data source:", metadata)
    return metadata

# Function to analyze news sentiment
def analyze_news_sentiment(news_data):
    news_df = pd.DataFrame(news_data)
    news_df['Sentiment'] = news_df['Headline'].apply(get_sentiment)
    average_sentiment_prob = news_df['Sentiment'].mean() * 100  # Average sentiment
    return average_sentiment_prob, news_df[['Date', 'Headline', 'Sentiment']]


# Check function example
def Check_feature_expansion(data):
    data['Random_Feature'] = data['Close'].rolling(window=15).mean()
    data['Noise_Feature'] = np.random.normal(0, 1, len(data))
    # print("Check features added")
    return data


def calculate_aco_probability(stock_data):
    # Parameters for ACO (tune parameters to achieve desired result)
    num_ants = 40
    num_iterations = 10
    pheromone_decay = 0.95  # Slower evaporation rate of pheromones (higher retention)
    alpha = 1  # Importance of pheromone
    beta = 2   # Increased importance of heuristic (price increase history)


    # Initial pheromone levels for recent data (last 100 days for efficiency)
    recent_data = stock_data.tail(100).reset_index(drop=True)  # Limit to last 100 days
    pheromone = np.ones(len(recent_data))  # Initial pheromone

    # Tune the scaling factor to avoid overflows while maintaining significant contributions
    pheromone_scaling_factor = 1e-5

    for iteration in range(num_iterations):
        total_path_score = 0


        for ant in range(num_ants):
            # Randomly choose a start point for the ant
            start_point = np.random.randint(0, len(recent_data) - 1)
            

            # Simulate the ant moving through the stock data
            current_position = start_point
            path_length = 0
            path_score = 0


            while current_position < len(recent_data) - 1:
                # Use pheromone and price change heuristic for next move probability
                price_change_factor = np.clip(1 + recent_data['Price_Change'].iloc[current_position], -5, 5)
                next_move_prob = pheromone[current_position] ** alpha * price_change_factor
                current_position += np.random.randint(1, 5)  # Move randomly 1-5 steps instead of just 1 step
                path_length += 1
                path_score += next_move_prob


                if current_position >= len(recent_data):  # Prevent out of bounds
                    break
            
            total_path_score += path_score


            # Update pheromone based on the path's performance
            for i in range(start_point, min(start_point + path_length, len(recent_data))):
                pheromone[i] += pheromone_scaling_factor * path_score  # Scale pheromone addition


        # Pheromone evaporation after each iteration
        pheromone *= pheromone_decay
        # print(f"Iteration {iteration + 1}/{num_iterations}, Total Path Score: {total_path_score}")
        # print(f"Pheromone at last point: {pheromone[-1]}")


    # The final probability is based on the accumulated pheromone on the last period
    max_pheromone = np.max(pheromone)
    aco_probability = pheromone[-1] * 100 / max_pheromone  # Normalize by max pheromone value
    #aco_probability = np.clip(aco_probability * 1.5, 30, 70)  # Adjust to achieve desired probability range
    # print(f"Max Pheromone: {max_pheromone}, Final ACO Probability: {aco_probability:.2f}%")
    return aco_probability


# Check function example
def Check_hyperparameter_exploration(model):
    hyperparams = {"n_estimators": [50, 100, 150], "max_depth": [None, 10, 20]}
    # print("Exploring hyperparameters:", hyperparams)
    return None


# Function to evaluate financial parameters
def evaluate_financials(eps, pe_ratio, industry_pe_ratio, de_ratio, financial_results, price_book_value):
    # Convert values to floats or set default
    eps = float(eps) if eps not in ["N/A", None] else 0
    pe_ratio = float(pe_ratio) if pe_ratio not in ["N/A", None] else float('inf')
    industry_pe_ratio = float(industry_pe_ratio) if industry_pe_ratio not in ["N/A", None] else float('inf')
    de_ratio = float(de_ratio) if de_ratio not in ["N/A", None] else float('inf')
    price_book_value = float(price_book_value) if price_book_value not in ["N/A", None] else float('inf')

    # Perform comparisons
    is_eps_high = eps > 0
    is_pe_ratio_low = pe_ratio < industry_pe_ratio
    is_de_ratio_low = de_ratio < 1  # Example condition
    is_price_book_value_low = price_book_value < 1  # Example condition

    # Combine conditions into final evaluation
    return all([is_eps_high, is_pe_ratio_low, is_de_ratio_low, is_price_book_value_low])




# Check function example
def Check_sentiment_metrics(sentiment):
    metrics = {
        "positive_threshold": 0.1,
        "neutral_threshold": 0,
        "negative_threshold": -0.1
    }
    # print("Sentiment metrics:", metrics)
    return metrics



# Initialize the VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Function to get sentiment score using VADER
def get_sentiment(headline):
    sentiment = analyzer.polarity_scores(headline)
    return sentiment['compound']  # Return the compound score


# Check function example
def Check_crash_probability_logging(probability):
    # print("Logging crash probability:", probability)
    log_entry = {"timestamp": pd.Timestamp.now(), "probability": probability}
    return log_entry


def get_news_data(stock_ticker):
    if not stock_ticker:
        return f"Company '{stock_ticker}' not found in the dictionary."
    
    # URL to fetch data (Example: Yahoo Finance)
    url = f"https://finance.yahoo.com/quote/{stock_ticker}?p={stock_ticker}"
    
    try:
        # Send HTTP request to fetch the web page
        response = requests.get(url)
        response.raise_for_status()  # Raise exception for HTTP errors
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract stock price (example using Yahoo Finance structure)
        price = soup.find('fin-streamer', {'data-field': 'regularMarketPrice'}).text
        change = soup.find('fin-streamer', {'data-field': 'regularMarketChangePercent'}).text
        
        # Extract recent news headlines
        news_section = soup.find_all('li', {'class': 'js-stream-content'})
        news = []
        
        # Assume today's date and subtract days for each news item
        today = datetime.today()
        
        for idx, item in enumerate(news_section[:5]):  # Get top 5 news items
            headline = item.find('h3').text
            link = item.find('a')['href']
            
            # Create a date for each news item (starting from today)
            news_date = (today - timedelta(days=idx)).strftime('%Y-%m-%d')
            
            news.append({
                "Date": news_date,
                "Headline": headline,
                "Link": f"https://finance.yahoo.com{link}"
            })
        
        # Return the data in the desired format
        return {stock_ticker: news}
    
    except requests.exceptions.RequestException as e:
        return f"Error fetching data for '{stock_ticker}': {e}"
    

def get_top_headlines():
    url = "https://news.yahoo.com/rss"
    
    try:
        # Create an unverified SSL context
        context = ssl._create_unverified_context()
        
        # Define a custom User-Agent header to mimic a browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Fetch the RSS feed using requests with the unverified SSL context and custom headers
        response = requests.get(url, verify=False, headers=headers)
        
        # Check if the request was successful
        if response.status_code == 200:
            # Parse the RSS feed
            feed = feedparser.parse(response.text)
            
            # Get the top 3 headlines
            headlines = [entry.title for entry in feed.entries[:3]]
            
            return headlines if headlines else ["No headlines found."]
        else:
            return [f"Error fetching news: {response.status_code}"]
    
    except Exception as e:
        return [f"Error fetching news: {e}"]





# Main function to run the trading strategy
def run_trading_strategy(stock_data_dict, news_data, financial_data):
    if not isinstance(news_data, dict):  # Ensure news_data is a dictionary
        print("Error: news_data is not a dictionary. Setting it to an empty dictionary.")
        news_data = {}  # Assign an empty dictionary to prevent errors

    for stock, stock_data in stock_data_dict.items():
        st.write(f"### Stock: {stock}")

        # Calculate crash probability
        crash_probability = calculate_crash_probability(stock_data)
        st.write(f"Probability of market crash: {crash_probability:.2f}%")

        # Calculate probability of going up using ML
        model = train_ml_model(stock_data)
        prob_up = model.predict_proba([[len(stock_data)]])[0][1] * 100
        st.write(f"Probability of going up (ML): {prob_up:.2f}%")

        # Calculate probability of going up using ACO
        aco_prob_up = calculate_aco_probability(stock_data)
        st.write(f"Probability of going up (ACO): {aco_prob_up:.2f}%")

        # Evaluate financials
        try:
            eps = financial_data.get('EPS', 0)
            pe_ratio = financial_data.get('P/E Ratio', 0)
            industry_pe_ratio = financial_data.get('Industry P/E Ratio', 0)
            de_ratio = financial_data.get('D/E Ratio', 0)
            financial_results = financial_data.get('Previous Quarters Financial Results', [])
            price_book_value = financial_data.get('P/B Ratio', 0)
        except AttributeError:
            print(f"Error: financial_data is not a dictionary. Skipping financial evaluation.")
            continue

        financial_evaluation = evaluate_financials(eps, pe_ratio, industry_pe_ratio, de_ratio, financial_results, price_book_value)
        st.write(f"Financial Evaluation: {'Pass' if financial_evaluation else 'Fail'}")

        # Check if news data exists for the stock
        if stock not in news_data or not news_data[stock]:
            st.write("#### News Sentiment")
            st.write("⚠️ No Data on Headlines")  # Display message in Streamlit
            print(f"No news data found for {stock}. Skipping news analysis...")
            continue

        # Analyze news sentiment
        news_sentiment_prob, news_sentiment_df = analyze_news_sentiment(news_data[stock])
        st.write("#### News Sentiment")
        st.write(news_sentiment_df)  # Display sentiment analysis
        st.write(f"Average news sentiment impact probability: {news_sentiment_prob:.2f}%")

        # Adjust the stock recommendation based on probabilities
        if crash_probability > 80:
            recommendation = "Sell the stock (high crash risk)."
        elif crash_probability > 50 and prob_up < 75:
            recommendation = "Sell the stock (moderate crash risk)."
        elif news_sentiment_prob < 50 and prob_up < 55:
            recommendation = "Sell the stock (negative news sentiment)."
        elif prob_up >= 55 or aco_prob_up >= 60:
            recommendation = "Buy the stock!"
            if news_sentiment_prob > 75:
                recommendation += " (Positive news sentiment)."
        elif 40 <= prob_up < 60 and 40 <= aco_prob_up < 60:
            recommendation = "Hold the stock."
        else:
            recommendation = "Sell the stock."

        st.write(f"Recommendation: {recommendation}") 
        st.write("---")



# Check function example
def Check_news_sentiment_analysis(news_df):
    analysis = {"num_positive": sum(news_df['Sentiment'] > 0.1), "num_negative": sum(news_df['Sentiment'] < -0.1)}
    # print("Detailed sentiment analysis:", analysis)
    return analysis


stock_keywords = {
    'AAPL': {
        'positive': ['innovative', 'record sales', 'strong demand', 'successful', 'growth', 'market leader', 'expansion', 'positive earnings'],
        'negative': ['lawsuit', 'decline', 'supply chain issues', 'competition', 'revenue drop', 'product recall', 'disappointing sales']
    },
    'GOOGL': {
        'positive': ['dominance', 'strong performance', 'user growth', 'ad revenue increase', 'innovative products', 'successful acquisitions'],
        'negative': ['regulatory scrutiny', 'antitrust', 'data breach', 'decline in ad revenue', 'competition', 'privacy concerns']
    },
    'MSFT': {
        'positive': ['growth', 'market share', 'cloud services', 'strong earnings', 'innovation', 'leadership', 'strategic partnerships'],
        'negative': ['security breach', 'decline in sales', 'competition', 'product issues', 'layoffs', 'lawsuit']
    },
    'AMZN': {
        'positive': ['expansion', 'record profits', 'customer growth', 'innovation', 'strong logistics', 'AWS growth'],
        'negative': ['losses', 'supply chain issues', 'increased competition', 'labor disputes', 'regulatory scrutiny']
    },
    'TSLA': {
        'positive': ['market leader', 'innovation', 'record deliveries', 'expansion', 'sustainable energy', 'strong demand'],
        'negative': ['recall', 'production issues', 'competition', 'legal issues', 'negative publicity']
    },
    'FB': {
        'positive': ['user growth', 'engagement', 'strong advertising revenue', 'innovation', 'successful rebranding'],
        'negative': ['privacy issues', 'regulatory scrutiny', 'user decline', 'fake news', 'data breaches']
    },
    'NFLX': {
        'positive': ['subscriber growth', 'original content success', 'strong revenue', 'global expansion', 'critical acclaim'],
        'negative': ['subscriber losses', 'competition', 'content cost', 'market saturation', 'negative reviews']
    },
    'BRK.A': {
        'positive': ['strong portfolio', 'value investing', 'consistent performance', 'diversification', 'Warren Buffett'],
        'negative': ['underperformance', 'market volatility', 'asset decline', 'poor investments']
    },
    'NVDA': {
        'positive': ['innovation', 'strong demand', 'market leader', 'growth in AI', 'gaming success', 'record earnings'],
        'negative': ['competition', 'supply chain issues', 'decline in sales', 'market fluctuations']
    },
    'DIS': {
        'positive': ['successful franchises', 'streaming growth', 'theme park success', 'strong brand', 'innovation'],
        'negative': ['decline in box office', 'content issues', 'negative reviews', 'competition']
    },
}



# Check function example
def Check_aco_logging(path_score):
    # print("ACO path score:", path_score)
    return {"path_score": path_score, "timestamp": pd.Timestamp.now()}



# Check financial feature logging
def Check_log_financials(financial_data):
    # print("Logging financial data:", financial_data)
    return None


# Sidebar for stock selection
st.sidebar.header('Stock Input')
ticker = st.sidebar.text_input('Ticker', 'AAPL')
start_date = st.sidebar.date_input('Start Date', value=pd.to_datetime('2021-12-24'))
end_date = st.sidebar.date_input('End Date', value=pd.to_datetime('2022-12-24'))

def get_stock_financial_data(stock_symbol):
    # Fetch stock data
    stock = yf.Ticker(stock_symbol)

    # Get financial data
    financial_data = {
        'EPS': stock.info.get('epsTrailingTwelveMonths', 'N/A'),  # Earnings Per Share
        'P/E Ratio': stock.info.get('trailingPE', 'N/A'),  # Price to Earnings Ratio
        'Industry P/E Ratio': stock.info.get('peRatio', 'N/A'),  # Industry P/E Ratio
        'D/E Ratio': stock.info.get('debtToEquity', 'N/A'),  # Debt to Equity Ratio
        'Previous Quarters Financial Results': stock.financials.loc['Net Income'].tail(4).tolist(),  # Previous 4 Quarters Earnings
        'P/B Ratio': stock.info.get('priceToBook', 'N/A')  # Price to Book Ratio
    }
    
    return financial_data

# Example usage:
financial_data = get_stock_financial_data('AAPL')

# Fetch stock data
data = get_stock_data(ticker, start_date, end_date)
print(data.columns)

# Display the title and stock graph at the beginning
st.title('Stock Trading Strategy with Machine Learning and Sentiment Analysis')

# Plot stock data first (before running the strategy)
fig, ax = plt.subplots()
ax.plot(data.index, data['Close'], label=ticker, color='blue')
ax.set_title(ticker, fontsize=16)
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Close', fontsize=12)
plt.xticks(rotation=45)
st.pyplot(fig)
listt = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'FB', 'NFLX', 'BRK.A', 'NVDA', 'DIS']

print("Before")

headlines = get_top_headlines()
for idx, headline in enumerate(headlines, 1):
    print(f"{idx}. {headline}")

print("After")

# Place the Run Strategy button in the sidebar below End Date
if st.sidebar.button("Run Strategy"):
    selected_stocks = [ticker]  # Only the entered ticker is selected
    stock_data_dict = {ticker: data}  # Use the fetched stock data
    news_data = get_news_data(ticker)
    if not isinstance(news_data, dict):  # Ensure news_data is a dictionary
        print("Error: news_data is not a dictionary. Setting it to an empty dictionary.")
        news_data = {}  # Assign an empty dictionary to prevent errors

    run_trading_strategy(stock_data_dict, news_data, financial_data)
