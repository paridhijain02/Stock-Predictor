from flask import Flask, render_template, request, send_file
import os
import numpy as np
import matplotlib.pyplot as plt
import time
from nsepython import equity_history
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
app = Flask(__name__)

app.config['PLOT_FOLDER'] = 'static/plots'
if not os.path.exists(app.config['PLOT_FOLDER']):
    os.makedirs(app.config['PLOT_FOLDER'])
# Home route
@app.route('/')
def home():
    return render_template('home.html')

# Contact route
@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    name = None
    judgement = ''
    plot_file_name = None
    if request.method == 'POST':
        name = request.form.get('name')

        data = equity_history(name, "EQ", "01-01-2022", "01-12-2023")
        if 'CH_CLOSING_PRICE' not in data or len(data['CH_CLOSING_PRICE']) == 0:
            message = "No data available for the specified stock. Please check the stock name and try again."
            return render_template('prediction.html', name=name, message=message)
        close_list = data['CH_CLOSING_PRICE']
        scaler = MinMaxScaler()
        df1 = scaler.fit_transform(np.array(close_list).reshape(-1, 1))

        training_size = int(len(df1) * 0.65)
        train_data, test_data = df1[0:training_size, :], df1[training_size:len(df1), :1]

        # convert an array of values into a dataset matrix
        def create_dataset(dataset, time_step=1):
            dataX, dataY = [], []
            for i in range(len(dataset) - time_step - 1):
                a = dataset[i:(i + time_step), 0]  ###i=0, 0,1,2,3-----99   100
                dataX.append(a)
                dataY.append(dataset[i + time_step, 0])
            return np.array(dataX), np.array(dataY)

        time_step = 100
        X_train, y_train = create_dataset(train_data, time_step)
        X_test, y_test = create_dataset(test_data, time_step)

        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        # Create the Stacked LSTM model

        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(100, 1)))
        model.add(LSTM(50, return_sequences=True))
        model.add(LSTM(50))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.summary()
        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=64, verbose=1)

        x_input = test_data[len(test_data) - 100:].reshape(1, -1)
        x_input.shape

        temp_input = (list(x_input))
        temp_input = temp_input[0].tolist()

        # demonstrate prediction for next 60 days
        lst_output = []
        n_steps = 100
        i = 0
        while (i < 60):
            if (len(temp_input) > 100):
                x_input = np.array(temp_input[1:])
                x_input = x_input.reshape(1, -1)
                x_input = x_input.reshape((1, n_steps, 1))
                yhat = model.predict(x_input, verbose=0)
                temp_input.extend(yhat[0].tolist())
                temp_input = temp_input[1:]
                lst_output.extend(yhat.tolist())
                i = i + 1
            else:
                x_input = x_input.reshape((1, n_steps, 1))
                yhat = model.predict(x_input, verbose=0)
                temp_input.extend(yhat[0].tolist())
                lst_output.extend(yhat.tolist())
                i = i + 1

        df3 = df1.tolist()
        df3.extend(lst_output)
        df3 = scaler.inverse_transform(df3).tolist()
        df1 = scaler.inverse_transform(df1)

        df4 = []
        # __________________df4 is future prediction of 60 days__________________
        for i in range(60):
            df4.append(df3[-(1 + i)])
        df5 = []
        # __________________df5 is last 60 days of data__________________
        for i in range(60):
            df5.append(df1[-(1 + i)])


        if np.sum(df4) >= np.sum(df5):
            judgement = 'You should buy these stocks'
        else:
            judgement = "You should sell these stocks"


        plt.figure(figsize=(10, 5))
        plt.plot(df3, label='Predicted Prices')
        plt.plot(df1, label='Actual Prices')
        plt.title('Stock Price Prediction')
        plt.xlabel('Days')
        plt.ylabel('Price')
        plt.legend()

        # Generate a unique filename using epoch time
        epoch_time = int(time.time())  # Get current epoch time
        plot_file_name = f'stock_prediction_{epoch_time}.png'
        plot_file_path = os.path.join('static', plot_file_name)

        # Save the plot to a file
        plt.savefig(plot_file_path)
        plt.close()

    return render_template('prediction.html', name=name, message=judgement, plot_url=plot_file_name)

def plot_graph(data, save_path):
    # Plotting the graph using Matplotlib
    plt.figure(figsize=(10, 5))
    plt.plot(data, label='Closing Price', color='blue')
    plt.title('Closing Prices Over Time')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)

    # Save the plot as a PNG file
    plt.savefig(save_path)
    plt.close()

if __name__ == '__main__':
    app.run(debug=True)
