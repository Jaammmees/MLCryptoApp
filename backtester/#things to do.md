#things to do

backtesting
- once model building and data processing done, attempt to backtest and generate
- need many configurable parameters
    - needs to take in how many minutes next it should be predicting,
    - whether it wants to backtest the whole period of the data? i know is already chosen in data processing but can do it again,

- use case should be
    - have processed training data, and have a scaler.pkl and column_names.pkl saved
    - train a model on this processed training data,
    - then when going to backtest, load the scaler.pkl, the column_names.pkl, the model, which allows it to process any data
      that has a OHLCV at a minimum to predict on it,

realtime
- once the rest is done, we can start trying to simulate real-time

building model
- allow for multiple metrics

data processing
- add options to change column names too if needed.

train model
- show data preview
- get rid of sequence generation, just show the sequence length given the model,
- try to show progress in the window itself,

#bugs to fix and error messages to do
whenever an error is counted, output it as a message box with ctk dialog
could wrap in a whole try catch block,
