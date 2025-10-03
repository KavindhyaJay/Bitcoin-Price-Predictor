import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(
    page_title="Bitcoin Price Predictor",
    page_icon="💰",
    layout="wide"
)

# Load model
model = load_model('D:\\ML\\Bitcoin\\Bitcoin_Price_Prediction_Model.keras')


st.markdown("<h1 style='color: gold;'>Bitcoin Price Prediction Model </h1>", unsafe_allow_html=True)
st.markdown("<h3 style='color: white;'>Bitcoin Price Data</h3>", unsafe_allow_html=True)

# Download data
data = pd.DataFrame(yf.download('BTC-USD', start='2010-01-01', end='2025-01-01'))
data = data.reset_index()
st.write(data)

col1, col2, col3 = st.columns(3)

col1.markdown("<p style='color: gold; font-weight: bold;'>Current BTC Price</p>", unsafe_allow_html=True)
col1.metric("", f"${float(data['Close'].iloc[-1]):,.2f}")

col2.markdown("<p style='color: gold; font-weight: bold;'>Max Price in Dataset</p>", unsafe_allow_html=True)
col2.metric("", f"${float(data['Close'].max()):,.2f}")

col3.markdown("<p style='color: gold; font-weight: bold;'>Min Price in Dataset</p>", unsafe_allow_html=True)
col3.metric("", f"${float(data['Close'].min()):,.2f}")


st.markdown("<br>", unsafe_allow_html=True)

st.markdown("<h2 style='color: gold;'>Bitcoin Line Chart  </h2>", unsafe_allow_html=True)
st.line_chart(data['Close'])



# Use only 'Close' column for model
train_data = data[['Close']][:-50]
test_data = data[['Close']][-700:]

# Scale Close prices
scaler = MinMaxScaler(feature_range=(0,1))
train_data_scaled = scaler.fit_transform(train_data)
test_data_scaled = scaler.transform(test_data)

# Prepare sequences
base_days = 500
x, y = [], []
for i in range(base_days, test_data_scaled.shape[0]):
    x.append(test_data_scaled[i-base_days:i, 0])
    y.append(test_data_scaled[i, 0])

x, y = np.array(x), np.array(y)
x = np.reshape(x, (x.shape[0], x.shape[1], 1))  # (samples, timesteps, features)

st.markdown("<br>", unsafe_allow_html=True)
# Predict
st.markdown("<h2 style='color: gold;'>Predicted vs Original Prices  </h2>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)
pred = model.predict(x)

# Inverse scale
pred = scaler.inverse_transform(pred)
y_actual = scaler.inverse_transform(y.reshape(-1,1))

# Prepare chart data
preds_df = pd.DataFrame(pred, columns=['Predicted Price'])
ys_df = pd.DataFrame(y_actual, columns=['Original Price'])
chart_data = pd.concat([preds_df, ys_df], axis=1)

st.write(chart_data)

st.markdown("<h2 style='color: gold;'>Predicted vs Original Prices Chart  </h2>", unsafe_allow_html=True)
compare_df = pd.DataFrame({'Predicted Price': pred.flatten(), 'Actual Price': y_actual.flatten()})
fig2 = go.Figure()
fig2.add_trace(go.Scatter(y=compare_df['Actual Price'], mode='lines', name='Actual Price', line=dict(color='lime')))
fig2.add_trace(go.Scatter(y=compare_df['Predicted Price'], mode='lines', name='Predicted Price', line=dict(color='gold')))
fig2.update_layout(xaxis_title='Day Index', yaxis_title='Price (USD)', template='plotly_dark')
st.plotly_chart(fig2, use_container_width=False)


m = y.reshape(-1,1)   
z = []

future_days = 3
last_sequence = x[-1]  

for i in range(future_days):
    inter = last_sequence.reshape(1, base_days, 1)
    pred = model.predict(inter, verbose=0)
    z.append(pred[0][0])
    
    # update sequence for next prediction
    last_sequence = np.append(last_sequence[1:], pred).reshape(-1,1)

st.markdown("<br>",unsafe_allow_html=True)
st.markdown("<h2 style='color: gold;'>Predicted Future Days Bitcoin Price  </h2>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)
z = np.array(z)
z = scaler.inverse_transform(z.reshape(-1,1))

st.line_chart(z)

