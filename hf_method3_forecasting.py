"""
HF Analytics Method 3: Time Series Forecasting
Uses Transformer models for price prediction
Models: amazon/chronos-t5-small, google/timesfm-1.0-200m
Enhances: backtest_engine.py, seasonality_analyzer.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Union
import json


class HFTimeSeriesForecaster:
    """
    Time series forecasting using Transformer models
    Predicts future prices based on historical patterns
    """
    
    def __init__(self, model_name: str = "amazon/chronos-t5-small"):
        self.model_name = model_name
        self.model = None
        self.device = None
        self.forecast_cache = {}
        
        print(f"Initializing HF Time Series Forecaster: {model_name}")
    
    def load_model(self):
        """Load forecasting model with proper error handling"""
        try:
            from chronos import ChronosPipeline
            import torch
            
            print(f"Loading model: {self.model_name}")
            
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"Using device: {self.device}")
            
            self.model = ChronosPipeline.from_pretrained(
                self.model_name,
                device_map=self.device,
                torch_dtype=torch.bfloat16 if self.device == 'cuda' else torch.float32
            )
            
            print("✓ Chronos model loaded successfully")
            return True
            
        except ImportError as e:
            print("⚠️  Chronos library not installed")
            print("   Install: pip install chronos-forecasting")
            print("   Falling back to statistical methods")
            return False
        except Exception as e:
            print(f"✗ Model loading error: {e}")
            print("   Falling back to statistical methods")
            return False
    
    def forecast(self, historical_data: Union[pd.Series, np.ndarray], 
                 forecast_horizon: int = 7,
                 num_samples: int = 20,
                 temperature: float = 1.0) -> Dict:
        """
        Generate forecast with multiple methods
        
        Args:
            historical_data: Historical price series
            forecast_horizon: Number of periods to forecast
            num_samples: Number of forecast samples (for uncertainty)
            temperature: Sampling temperature (higher = more uncertainty)
        
        Returns:
            Dictionary with forecast statistics
        """
        if isinstance(historical_data, pd.Series):
            data_values = historical_data.values
            data_index = historical_data.index
        else:
            data_values = historical_data
            data_index = None
        
        if len(data_values) < 10:
            return {
                'error': 'Insufficient data (need at least 10 points)',
                'method': 'error'
            }
        
        cache_key = f"{hash(data_values.tobytes())}_{forecast_horizon}"
        if cache_key in self.forecast_cache:
            print("Using cached forecast")
            return self.forecast_cache[cache_key]
        
        if self.model:
            result = self._chronos_forecast(
                data_values, 
                forecast_horizon, 
                num_samples, 
                temperature
            )
        else:
            result = self._statistical_forecast(
                data_values, 
                forecast_horizon
            )
        
        result['forecast_dates'] = self._generate_forecast_dates(
            data_index, 
            forecast_horizon
        )
        
        self.forecast_cache[cache_key] = result
        
        return result
    
    def _chronos_forecast(self, data: np.ndarray, horizon: int,
                         num_samples: int, temperature: float) -> Dict:
        """Chronos transformer forecast"""
        try:
            import torch
            
            context_length = min(512, len(data))
            context = torch.tensor(data[-context_length:], dtype=torch.float32)
            
            print(f"Forecasting with Chronos (context: {context_length} points)...")
            
            with torch.no_grad():
                forecast_samples = self.model.predict(
                    context.unsqueeze(0),
                    prediction_length=horizon,
                    num_samples=num_samples,
                    temperature=temperature
                )
            
            forecast_array = forecast_samples[0].cpu().numpy()
            
            forecast_mean = forecast_array.mean(axis=0)
            forecast_median = np.median(forecast_array, axis=0)
            forecast_std = forecast_array.std(axis=0)
            
            percentile_10 = np.percentile(forecast_array, 10, axis=0)
            percentile_25 = np.percentile(forecast_array, 25, axis=0)
            percentile_75 = np.percentile(forecast_array, 75, axis=0)
            percentile_90 = np.percentile(forecast_array, 90, axis=0)
            
            return {
                'forecast_mean': forecast_mean.tolist(),
                'forecast_median': forecast_median.tolist(),
                'forecast_std': forecast_std.tolist(),
                'confidence_50': {
                    'lower': percentile_25.tolist(),
                    'upper': percentile_75.tolist()
                },
                'confidence_80': {
                    'lower': percentile_10.tolist(),
                    'upper': percentile_90.tolist()
                },
                'confidence_95': {
                    'lower': (forecast_mean - 1.96 * forecast_std).tolist(),
                    'upper': (forecast_mean + 1.96 * forecast_std).tolist()
                },
                'horizon': horizon,
                'num_samples': num_samples,
                'method': 'chronos',
                'model': self.model_name
            }
            
        except Exception as e:
            print(f"Chronos forecast error: {e}")
            return self._statistical_forecast(data, horizon)
    
    def _statistical_forecast(self, data: np.ndarray, horizon: int) -> Dict:
        """Statistical fallback forecasting methods"""
        methods_tried = []
        
        try:
            result = self._exponential_smoothing(data, horizon)
            methods_tried.append('exponential_smoothing')
            return result
        except Exception as e:
            print(f"Exponential smoothing failed: {e}")
        
        try:
            result = self._moving_average_forecast(data, horizon)
            methods_tried.append('moving_average')
            return result
        except Exception as e:
            print(f"Moving average failed: {e}")
        
        return self._naive_forecast(data, horizon)
    
    def _exponential_smoothing(self, data: np.ndarray, horizon: int,
                               alpha: float = 0.3) -> Dict:
        """Exponential smoothing forecast"""
        level = data[-1]
        trend = np.mean(np.diff(data[-10:]))
        
        forecast = []
        
        for h in range(horizon):
            forecast_value = level + (h + 1) * trend
            forecast.append(forecast_value)
            level = alpha * forecast_value + (1 - alpha) * level
        
        forecast = np.array(forecast)
        
        recent_std = np.std(data[-20:])
        forecast_std = recent_std * np.sqrt(np.arange(1, horizon + 1))
        
        return {
            'forecast_mean': forecast.tolist(),
            'forecast_median': forecast.tolist(),
            'forecast_std': forecast_std.tolist(),
            'confidence_50': {
                'lower': (forecast - 0.67 * forecast_std).tolist(),
                'upper': (forecast + 0.67 * forecast_std).tolist()
            },
            'confidence_80': {
                'lower': (forecast - 1.28 * forecast_std).tolist(),
                'upper': (forecast + 1.28 * forecast_std).tolist()
            },
            'confidence_95': {
                'lower': (forecast - 1.96 * forecast_std).tolist(),
                'upper': (forecast + 1.96 * forecast_std).tolist()
            },
            'horizon': horizon,
            'method': 'exponential_smoothing',
            'parameters': {'alpha': alpha}
        }
    
    def _moving_average_forecast(self, data: np.ndarray, 
                                 horizon: int, window: int = 10) -> Dict:
        """Moving average forecast"""
        ma = np.mean(data[-window:])
        trend = np.mean(np.diff(data[-window:]))
        
        forecast = [ma + (h + 1) * trend for h in range(horizon)]
        forecast = np.array(forecast)
        
        std = np.std(data[-window:])
        forecast_std = np.array([std * np.sqrt(h + 1) for h in range(horizon)])
        
        return {
            'forecast_mean': forecast.tolist(),
            'forecast_median': forecast.tolist(),
            'forecast_std': forecast_std.tolist(),
            'confidence_80': {
                'lower': (forecast - 1.28 * forecast_std).tolist(),
                'upper': (forecast + 1.28 * forecast_std).tolist()
            },
            'horizon': horizon,
            'method': 'moving_average',
            'parameters': {'window': window}
        }
    
    def _naive_forecast(self, data: np.ndarray, horizon: int) -> Dict:
        """Naive forecast (last observation carried forward)"""
        last_value = data[-1]
        forecast = np.full(horizon, last_value)
        
        std = np.std(data[-20:])
        forecast_std = np.array([std * np.sqrt(h + 1) for h in range(horizon)])
        
        return {
            'forecast_mean': forecast.tolist(),
            'forecast_median': forecast.tolist(),
            'forecast_std': forecast_std.tolist(),
            'confidence_80': {
                'lower': (forecast - 1.28 * forecast_std).tolist(),
                'upper': (forecast + 1.28 * forecast_std).tolist()
            },
            'horizon': horizon,
            'method': 'naive'
        }
    
    def _generate_forecast_dates(self, historical_index: Optional[pd.Index],
                                horizon: int) -> List[str]:
        """Generate dates for forecast period"""
        if historical_index is None or len(historical_index) == 0:
            return [f"T+{i+1}" for i in range(horizon)]
        
        try:
            last_date = pd.to_datetime(historical_index[-1])
            
            if len(historical_index) > 1:
                freq = pd.infer_freq(historical_index[-10:])
            else:
                freq = 'D'
            
            if freq is None:
                freq = 'D'
            
            forecast_index = pd.date_range(
                start=last_date, 
                periods=horizon + 1, 
                freq=freq
            )[1:]
            
            return [d.strftime('%Y-%m-%d') for d in forecast_index]
            
        except Exception as e:
            print(f"Date generation error: {e}")
            return [f"T+{i+1}" for i in range(horizon)]
    
    def forecast_multiple_horizons(self, historical_data: Union[pd.Series, np.ndarray],
                                   horizons: List[int] = [1, 5, 10, 20]) -> Dict:
        """Generate forecasts for multiple horizons"""
        results = {}
        
        for horizon in horizons:
            print(f"Forecasting {horizon} periods ahead...")
            forecast = self.forecast(historical_data, forecast_horizon=horizon)
            results[f"horizon_{horizon}"] = forecast
        
        return results
    
    def evaluate_forecast_accuracy(self, historical_data: pd.Series,
                                  test_size: int = 10) -> Dict:
        """Evaluate forecast accuracy on held-out data"""
        if len(historical_data) < test_size + 20:
            return {'error': 'Insufficient data for evaluation'}
        
        train_data = historical_data.iloc[:-test_size]
        test_data = historical_data.iloc[-test_size:]
        
        forecast = self.forecast(train_data, forecast_horizon=test_size)
        
        predictions = np.array(forecast['forecast_mean'])
        actuals = test_data.values
        
        mae = np.mean(np.abs(predictions - actuals))
        rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
        mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
        
        return {
            'mae': round(mae, 4),
            'rmse': round(rmse, 4),
            'mape': round(mape, 2),
            'method': forecast['method'],
            'test_size': test_size
        }
    
    def save_results(self, results: Dict, filepath: str):
        """Save forecast results"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved: {filepath}")


if __name__ == "__main__":
    import yfinance as yf
    
    print("="*80)
    print("HF METHOD 3: TIME SERIES FORECASTING")
    print("="*80)
    
    forecaster = HFTimeSeriesForecaster()
    model_loaded = forecaster.load_model()
    
    print("\n" + "="*80)
    print("TEST 1: REAL DATA FORECAST")
    print("="*80)
    
    symbol = 'GC=F'
    print(f"\nFetching {symbol} data...")
    
    ticker = yf.Ticker(symbol)
    df = ticker.history(period='6mo')
    
    if not df.empty:
        prices = df['Close']
        
        print(f"Historical data: {len(prices)} days")
        print(f"Current price: ${prices.iloc[-1]:.2f}")
        
        print("\n--- Single Horizon Forecast ---")
        forecast = forecaster.forecast(prices, forecast_horizon=7)
        
        print(f"\nMethod: {forecast['method']}")
        print(f"Forecast horizon: {forecast['horizon']} days")
        
        if 'forecast_mean' in forecast:
            print("\nPredictions:")
            for i, (pred, date) in enumerate(zip(forecast['forecast_mean'], 
                                                 forecast['forecast_dates'])):
                print(f"  {date}: ${pred:.2f}")
            
            if 'confidence_80' in forecast:
                print("\n80% Confidence Intervals:")
                for i, date in enumerate(forecast['forecast_dates']):
                    lower = forecast['confidence_80']['lower'][i]
                    upper = forecast['confidence_80']['upper'][i]
                    print(f"  {date}: ${lower:.2f} - ${upper:.2f}")
        
        print("\n--- Multiple Horizons ---")
        multi_forecast = forecaster.forecast_multiple_horizons(
            prices, 
            horizons=[1, 5, 10]
        )
        
        for horizon_key, result in multi_forecast.items():
            if 'forecast_mean' in result:
                horizon = result['horizon']
                final_pred = result['forecast_mean'][-1]
                print(f"{horizon_key}: ${final_pred:.2f}")
        
        print("\n--- Forecast Accuracy Evaluation ---")
        accuracy = forecaster.evaluate_forecast_accuracy(prices, test_size=10)
        
        if 'mae' in accuracy:
            print(f"Method: {accuracy['method']}")
            print(f"MAE: ${accuracy['mae']:.2f}")
            print(f"RMSE: ${accuracy['rmse']:.2f}")
            print(f"MAPE: {accuracy['mape']:.2f}%")
        
        forecaster.save_results(forecast, 'hf_forecast_results.json')
    
    print("\n" + "="*80)
    print("TEST 2: SYNTHETIC DATA")
    print("="*80)
    
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    trend = np.linspace(1800, 2000, 100)
    noise = np.random.normal(0, 10, 100)
    synthetic_prices = pd.Series(trend + noise, index=dates)
    
    forecast = forecaster.forecast(synthetic_prices, forecast_horizon=10)
    
    print(f"\nSynthetic forecast ({forecast['method']}):")
    print(f"  Starting: ${synthetic_prices.iloc[-1]:.2f}")
    print(f"  10-day ahead: ${forecast['forecast_mean'][-1]:.2f}")
    print(f"  Std deviation: ${forecast['forecast_std'][-1]:.2f}")
    
    print("\n" + "="*80)
    print("✓ Time series forecasting complete")
    print("="*80)
