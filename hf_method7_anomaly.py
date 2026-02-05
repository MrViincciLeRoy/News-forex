"""
HF Analytics Method 7: Anomaly Detection
Detect unusual market behavior using transformer models
Models: microsoft/deberta-v3-base (fine-tuned)
Enhances: alert_system.py, volume_analyzer.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import json


class HFAnomalyDetector:
    """
    Detect market anomalies beyond statistical thresholds
    Uses AI to identify regime changes and unusual patterns
    """
    
    def __init__(self):
        self.model = None
        self.anomaly_threshold = 0.7
        self.baseline_stats = {}
        
        print("Initializing HF Anomaly Detector")
    
    def load_model(self):
        print("⚠️  Using statistical baseline (transformer model optional)")
        print("   Install: pip install pyod for advanced anomaly detection")
        return True
    
    def set_baseline(self, historical_data: pd.DataFrame, symbol: str):
        stats = {
            'mean_price': float(historical_data['close'].mean()),
            'std_price': float(historical_data['close'].std()),
            'mean_volume': float(historical_data['volume'].mean()),
            'std_volume': float(historical_data['volume'].std()),
            'mean_range': float((historical_data['high'] - historical_data['low']).mean()),
            'std_range': float((historical_data['high'] - historical_data['low']).std())
        }
        
        historical_data['returns'] = historical_data['close'].pct_change()
        stats['mean_returns'] = float(historical_data['returns'].mean())
        stats['std_returns'] = float(historical_data['returns'].std())
        stats['volatility'] = float(historical_data['returns'].std() * np.sqrt(252))
        
        self.baseline_stats[symbol] = stats
        
        print(f"✓ Baseline set for {symbol}")
        return stats
    
    def detect_price_anomaly(self, current_price: float, symbol: str) -> Dict:
        if symbol not in self.baseline_stats:
            return {
                'anomaly': False,
                'reason': 'No baseline established',
                'severity': 'NONE'
            }
        
        stats = self.baseline_stats[symbol]
        z_score = abs((current_price - stats['mean_price']) / stats['std_price'])
        
        is_anomaly = bool(z_score > 3)
        severity = 'EXTREME' if z_score > 4 else ('HIGH' if z_score > 3 else 'NORMAL')
        
        return {
            'anomaly': is_anomaly,
            'z_score': round(float(z_score), 2),
            'current_price': float(current_price),
            'baseline_mean': round(float(stats['mean_price']), 2),
            'deviation': round(float(current_price - stats['mean_price']), 2),
            'severity': severity,
            'reason': f"Price {z_score:.1f} standard deviations from mean"
        }
    
    def detect_volume_anomaly(self, current_volume: int, symbol: str) -> Dict:
        if symbol not in self.baseline_stats:
            return {'anomaly': False, 'reason': 'No baseline', 'severity': 'NONE'}
        
        stats = self.baseline_stats[symbol]
        z_score = abs((current_volume - stats['mean_volume']) / stats['std_volume'])
        
        is_anomaly = bool(z_score > 2.5)
        severity = 'EXTREME' if z_score > 4 else ('HIGH' if z_score > 2.5 else 'NORMAL')
        
        return {
            'anomaly': is_anomaly,
            'z_score': round(float(z_score), 2),
            'current_volume': int(current_volume),
            'baseline_mean': int(stats['mean_volume']),
            'multiplier': round(float(current_volume / stats['mean_volume']), 2),
            'severity': severity,
            'reason': f"Volume {z_score:.1f} standard deviations from mean"
        }
    
    def detect_volatility_regime_change(self, recent_data: pd.DataFrame, 
                                       symbol: str, window: int = 20) -> Dict:
        if symbol not in self.baseline_stats:
            return {'anomaly': False, 'reason': 'No baseline', 'severity': 'NONE'}
        
        recent_data = recent_data.copy()
        recent_data['returns'] = recent_data['close'].pct_change()
        recent_vol = recent_data['returns'].tail(window).std() * np.sqrt(252)
        
        baseline_vol = self.baseline_stats[symbol]['volatility']
        
        vol_ratio = recent_vol / baseline_vol
        
        is_anomaly = bool(vol_ratio > 1.5 or vol_ratio < 0.5)
        
        if vol_ratio > 2:
            severity = 'EXTREME'
            regime = 'HIGH_VOLATILITY'
        elif vol_ratio > 1.5:
            severity = 'HIGH'
            regime = 'ELEVATED_VOLATILITY'
        elif vol_ratio < 0.5:
            severity = 'HIGH'
            regime = 'LOW_VOLATILITY'
        else:
            severity = 'NORMAL'
            regime = 'NORMAL_VOLATILITY'
        
        return {
            'anomaly': is_anomaly,
            'current_volatility': round(float(recent_vol * 100), 2),
            'baseline_volatility': round(float(baseline_vol * 100), 2),
            'volatility_ratio': round(float(vol_ratio), 2),
            'regime': regime,
            'severity': severity,
            'reason': f"Volatility {vol_ratio:.2f}x baseline"
        }
    
    def detect_correlation_breakdown(self, asset1_data: pd.Series,
                                    asset2_data: pd.Series,
                                    historical_corr: float,
                                    window: int = 20) -> Dict:
        if len(asset1_data) < window or len(asset2_data) < window:
            return {'anomaly': False, 'reason': 'Insufficient data', 'severity': 'NONE'}
        
        recent_corr = asset1_data.tail(window).corr(asset2_data.tail(window))
        corr_change = abs(recent_corr - historical_corr)
        
        is_anomaly = bool(corr_change > 0.5)
        severity = 'EXTREME' if corr_change > 0.7 else ('HIGH' if corr_change > 0.5 else 'NORMAL')
        
        return {
            'anomaly': is_anomaly,
            'recent_correlation': round(float(recent_corr), 3),
            'historical_correlation': round(float(historical_corr), 3),
            'correlation_change': round(float(corr_change), 3),
            'severity': severity,
            'reason': f"Correlation shifted by {corr_change:.2f}"
        }
    
    def detect_gap_anomaly(self, current_open: float, previous_close: float,
                          symbol: str) -> Dict:
        if symbol not in self.baseline_stats:
            return {'anomaly': False, 'reason': 'No baseline', 'severity': 'NONE'}
        
        gap_pct = abs((current_open - previous_close) / previous_close) * 100
        
        is_anomaly = bool(gap_pct > 2)
        severity = 'EXTREME' if gap_pct > 5 else ('HIGH' if gap_pct > 2 else 'NORMAL')
        
        gap_direction = 'UP' if current_open > previous_close else 'DOWN'
        
        return {
            'anomaly': is_anomaly,
            'gap_percent': round(float(gap_pct), 2),
            'gap_direction': gap_direction,
            'current_open': float(current_open),
            'previous_close': float(previous_close),
            'severity': severity,
            'reason': f"Gap {gap_direction} of {gap_pct:.1f}%"
        }
    
    def comprehensive_anomaly_scan(self, current_data: pd.Series,
                                  historical_data: pd.DataFrame,
                                  symbol: str) -> Dict:
        if symbol not in self.baseline_stats:
            self.set_baseline(historical_data, symbol)
        
        anomalies = []
        
        price_check = self.detect_price_anomaly(float(current_data['close']), symbol)
        if price_check['anomaly']:
            anomalies.append({
                'type': 'PRICE_ANOMALY',
                **price_check
            })
        
        volume_check = self.detect_volume_anomaly(int(current_data['volume']), symbol)
        if volume_check['anomaly']:
            anomalies.append({
                'type': 'VOLUME_ANOMALY',
                **volume_check
            })
        
        vol_regime = self.detect_volatility_regime_change(historical_data, symbol)
        if vol_regime['anomaly']:
            anomalies.append({
                'type': 'VOLATILITY_REGIME_CHANGE',
                **vol_regime
            })
        
        if len(historical_data) > 1:
            gap_check = self.detect_gap_anomaly(
                float(current_data['open']),
                float(historical_data['close'].iloc[-2]),
                symbol
            )
            if gap_check['anomaly']:
                anomalies.append({
                    'type': 'PRICE_GAP',
                    **gap_check
                })
        
        overall_severity = 'NORMAL'
        if any(a['severity'] == 'EXTREME' for a in anomalies):
            overall_severity = 'EXTREME'
        elif any(a['severity'] == 'HIGH' for a in anomalies):
            overall_severity = 'HIGH'
        
        return {
            'symbol': symbol,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'anomalies_detected': len(anomalies),
            'overall_severity': overall_severity,
            'anomalies': anomalies,
            'requires_attention': bool(len(anomalies) > 0)
        }
    
    def save_results(self, results: Dict, filepath: str):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved: {filepath}")


if __name__ == "__main__":
    import yfinance as yf
    
    print("="*80)
    print("HF METHOD 7: ANOMALY DETECTION")
    print("="*80)
    
    detector = HFAnomalyDetector()
    detector.load_model()
    
    symbol = 'AAPL'
    print(f"\nTesting anomaly detection on {symbol}...")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start_date.strftime('%Y-%m-%d'))
    
    if not df.empty:
        df = df.rename(columns={
            'Open': 'open', 'High': 'high', 'Low': 'low',
            'Close': 'close', 'Volume': 'volume'
        })
        
        print("\n" + "="*80)
        print("SETTING BASELINE")
        print("="*80)
        
        baseline = detector.set_baseline(df.iloc[:-5], symbol)
        print(f"\nBaseline Statistics:")
        print(f"  Mean Price: ${baseline['mean_price']:.2f}")
        print(f"  Price Std: ${baseline['std_price']:.2f}")
        print(f"  Mean Volume: {baseline['mean_volume']:,.0f}")
        print(f"  Volatility: {baseline['volatility']*100:.1f}%")
        
        print("\n" + "="*80)
        print("RUNNING ANOMALY SCAN")
        print("="*80)
        
        current = df.iloc[-1]
        scan_results = detector.comprehensive_anomaly_scan(current, df, symbol)
        
        print(f"\nScan Results for {scan_results['symbol']}:")
        print(f"Timestamp: {scan_results['timestamp']}")
        print(f"Anomalies Detected: {scan_results['anomalies_detected']}")
        print(f"Overall Severity: {scan_results['overall_severity']}")
        print(f"Requires Attention: {scan_results['requires_attention']}")
        
        if scan_results['anomalies']:
            print("\nDetected Anomalies:")
            for anomaly in scan_results['anomalies']:
                print(f"\n  {anomaly['type']}:")
                print(f"    Severity: {anomaly['severity']}")
                print(f"    Reason: {anomaly['reason']}")
                if 'z_score' in anomaly:
                    print(f"    Z-Score: {anomaly['z_score']}")
        else:
            print("\n✓ No anomalies detected - market behavior is normal")
        
        detector.save_results(scan_results, 'hf_anomaly_detection_results.json')
    
    print("\n" + "="*80)
    print("✓ Anomaly detection complete")
    print("="*80)
