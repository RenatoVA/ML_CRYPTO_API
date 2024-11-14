import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np
import joblib


class Model:
    def __init__(self):
        self.gaming = pd.read_csv('data/processed/gaming.csv')
        self.meme = pd.read_csv('data/processed/meme.csv')
        self.ai = pd.read_csv('data/processed/ai.csv')
        self.rwa = pd.read_csv('data/processed/rwa.csv')
        self.df = pd.concat([self.gaming, self.meme, self.ai, self.rwa], ignore_index=True)
        self.features = ['open', 'high', 'low', 'close', 'volume', 'close_percentage', 'volume_percentage', 'daily_change', 'up_down',
            'mv_7', 'mv_14', 'mv_21', 'volat_7', 'volat_14', 'volat_21']
        self.model_folder = 'data/models/'
        self.scaler_folder = 'data/scalers/'
        pass
    def get_dfcategories(self,name):
        if name == 'gaming':
            return self.gaming
        elif name == 'meme':
            return self.meme
        elif name == 'ai':
            return self.ai
        elif name == 'rwa':
            return self.rwa
        else:
            raise ValueError('Invalid category')
    def add_ons(self, df):
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df['close_percentage'] = df.groupby('symbol')['close'].pct_change()
        df['volume_percentage'] = df.groupby('symbol')['volume'].pct_change()

        df['mv_7'] = df.groupby('symbol')['close'].transform(lambda x: x.rolling(window=7).mean())
        df['mv_14'] = df.groupby('symbol')['close'].transform(lambda x: x.rolling(window=14).mean())
        df['mv_21'] = df.groupby('symbol')['close'].transform(lambda x: x.rolling(window=21).mean())

        df['volat_7'] = df.groupby('symbol')['close'].transform(lambda x: x.rolling(window=7).std())
        df['volat_14'] = df.groupby('symbol')['close'].transform(lambda x: x.rolling(window=14).std())
        df['volat_21'] = df.groupby('symbol')['close'].transform(lambda x: x.rolling(window=21).std())

        df['daily_change'] = df['high'] - df['low']
        df['up_down'] = df['close']/df['open']
        return df
    def data_scaler(self, df, scaler):
        df[self.features] = scaler.transform(df[self.features])
        return df
    def eval(self):
        self.gaming = self.add_ons(self.gaming)
        self.meme = self.add_ons(self.meme)
        self.ai = self.add_ons(self.ai)
        self.rwa = self.add_ons(self.rwa)
        self.df = self.add_ons(self.df)
    def load_model_by_token(self,token):
        model_path = f"{self.model_folder}{token}_model.joblib"
        scaler_path = f"{self.scaler_folder}{token}_scaler.joblib"
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    def load_models_by_category(self,category):
        models = {}
        scalers = {}
        for token in category:
            model, scaler = self.load_model_by_token(token)
            models[token]=model
            scalers[token]=scaler
        return models, scalers
    def predict_by_token(self, days, token, input_data):
        model, scaler = self.load_model_by_token(token)
        input_data=input_data[input_data['symbol']==token]
        X_last = input_data[self.features].iloc[-1].copy().values.reshape(1, -1)
        ultima_fecha = input_data['timestamp'].max()
        predicciones_futuras = []
        for i in range(days):
            y_next = model.predict(X_last)
            y_next_descaled = scaler.inverse_transform([[0] * (len(self.features) - 1) + [val] for val in y_next])[:, -1]
            fecha_prediccion = ultima_fecha + pd.Timedelta(days=i + 1)
            predicciones_futuras.append({'fecha': fecha_prediccion, 'prediccion': y_next_descaled[0]})
            X_last = np.concatenate([X_last[:, 1:], [[y_next[0]]]], axis=1)
        return predicciones_futuras
    def predict_by_category(self, days, category):
        df_category = self.get_dfcategories(category)
        tokens = df_category['symbol'].unique()
        models, scalers = self.load_models_by_category(tokens)
        future_prices_per_token = {}
        for token in tokens:
            model=models[token]
            input_data=df_category[df_category['symbol']==token]
            X_last = input_data[self.features].iloc[-1].copy().values.reshape(1, -1)
            ultima_fecha = input_data['timestamp'].max()
            predicciones_futuras = []

            for i in range(days):
                y_next = model.predict(X_last)

                y_next_descaled = scalers[token].inverse_transform([[0] * (len(self.features) - 1) + [val] for val in y_next])[:, -1]
                fecha_prediccion = ultima_fecha + pd.Timedelta(days=i + 1)
                predicciones_futuras.append({'fecha': fecha_prediccion, 'prediccion': y_next_descaled[0]})
                X_last = np.concatenate([X_last[:, 1:], [[y_next[0]]]], axis=1)
            future_prices_per_token[token] = pd.DataFrame(predicciones_futuras)
        return future_prices_per_token
    def top_5_tokens_by_category(self, category, days=30):
        future_prices = self.predict_by_category(days, category)
        price_changes = []

        for token, predictions in future_prices.items():
            initial_price = predictions['prediccion'].iloc[0]
            final_price = predictions['prediccion'].iloc[-1]
            price_change_percentage = ((final_price - initial_price) / initial_price) * 100
            price_changes.append((token, price_change_percentage))
        top_5_tokens = sorted(price_changes, key=lambda x: x[1], reverse=True)[:5]
        return top_5_tokens
if __name__ == "__main__":
    model = Model()
    model.eval()
    print(model.top_5_tokens_by_category('gaming'))
