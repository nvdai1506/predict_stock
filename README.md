# Stock Price Prediction – Machine Learning Project in Python

## Xây dựng trang DashBoard phân tích chứng khoán theo các tiêu chí sau

1. Người dùng chọn một trong các phương pháp dự đoán:
    - XGBoost, RNN, LSTM (bắt buộc)
    - Transformer and Time Embeddings (nâng cao - có thể làm hoặc không có điểm cộng)
2. Người dùng chọn một hay nhiều đặc trưng để dự đoán :
    - Close, Price Of Change (bắt buộc)
    - RSI, Bolling Bands, Moving Average,...(Nâng cao)

## Author

- Lâm Khang Vỉ
- Nguyễn Văn Đại
- Lê Trần Lâm An
- Thái Nhật Minh

## Running

### Prepare

- Install python module:
    ```
    pip install flask flask-cors pandas keras tensorflow sklearn xgboost
    ```
- Install node module in frontend:
    ```
    cd frontend
    npm install
    ```

### Start server

- Start backend:
    ```
    python backend/main.py
    ```
- Start frontend:
    ```
    npm start
    ```

## Ref

- <https://data-flair.training/blogs/stock-price-prediction-machine-learning-project-in-python>
- <https://radiant-brushlands-42789.herokuapp.com/towardsdatascience.com/stock-predictions-with-state-of-the-art-transformer-and-time-embeddings-3a4485237de6?fbclid=IwAR10TqnPAfNUbOW_vQuhQ5_4O-Wre9LBc4YtHi8eu7R9a2-X9SjuS9-QO-A>
