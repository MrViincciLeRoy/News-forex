"""
Economic Indicators - Fetch and analyze economic data
"""

import numpy as np
from typing import Dict, List
from datetime import datetime

class EconomicIndicators:
    
    def __init__(self):
        self.cache = {}
    
    def get_indicators(self, date: str) -> Dict:
        """Get economic indicators"""
        
        # In production, would fetch from FRED, Trading Economics, etc.
        return self._generate_mock_indicators(date)
    
    def _generate_mock_indicators(self, date: str) -> Dict:
        """Generate realistic economic indicators"""
        
        np.random.seed(hash(date) % 1000)
        
        return {
            'interest_rates': {
                'fed_funds_rate': round(np.random.uniform(4.5, 5.5), 2),
                'us_10y_yield': round(np.random.uniform(4.0, 5.0), 2),
                'us_2y_yield': round(np.random.uniform(4.5, 5.5), 2),
                'yield_curve': round(np.random.uniform(-0.5, 0.5), 2),
                'ecb_rate': round(np.random.uniform(3.5, 4.5), 2),
                'boj_rate': round(np.random.uniform(-0.1, 0.5), 2)
            },
            'inflation': {
                'us_cpi': round(np.random.uniform(2.5, 4.0), 1),
                'us_core_cpi': round(np.random.uniform(3.0, 4.5), 1),
                'us_pce': round(np.random.uniform(2.0, 3.5), 1),
                'eu_cpi': round(np.random.uniform(2.0, 3.5), 1),
                'trend': 'RISING' if np.random.random() > 0.5 else 'FALLING'
            },
            'employment': {
                'us_unemployment': round(np.random.uniform(3.5, 4.5), 1),
                'us_nonfarm_payrolls': np.random.randint(150000, 350000),
                'us_jobless_claims': np.random.randint(200000, 250000),
                'status': 'STRONG' if np.random.random() > 0.5 else 'MODERATE'
            },
            'gdp': {
                'us_gdp_growth': round(np.random.uniform(1.5, 3.5), 1),
                'eu_gdp_growth': round(np.random.uniform(0.5, 2.0), 1),
                'china_gdp_growth': round(np.random.uniform(4.0, 6.0), 1)
            },
            'manufacturing': {
                'us_pmi': round(np.random.uniform(48.0, 52.0), 1),
                'eu_pmi': round(np.random.uniform(45.0, 50.0), 1),
                'china_pmi': round(np.random.uniform(49.0, 51.0), 1)
            },
            'consumer': {
                'us_retail_sales': round(np.random.uniform(-0.5, 2.0), 1),
                'us_consumer_confidence': round(np.random.uniform(95.0, 110.0), 1)
            }
        }
    
    def generate_ai_explanations(self, indicators: Dict) -> Dict:
        """Generate AI explanations for each indicator group"""
        
        explanations = {}
        
        # Interest Rates
        ir = indicators['interest_rates']
        yield_curve = ir['yield_curve']
        
        if yield_curve < 0:
            ir_explanation = f"The yield curve is inverted ({yield_curve}%), historically a recession indicator. "
            ir_explanation += "Fed funds rate at {fed_funds_rate}% suggests restrictive monetary policy. "
            ir_explanation += "This typically strengthens USD in short-term but signals economic headwinds."
        else:
            ir_explanation = f"Positive yield curve ({yield_curve}%) suggests normal economic conditions. "
            ir_explanation += f"Fed funds rate at {ir['fed_funds_rate']}% indicates measured policy stance. "
            ir_explanation += "Interest rate differentials will drive currency flows."
        
        explanations['interest_rates'] = {
            'explanation': ir_explanation,
            'market_impact': {
                'USD': 'SUPPORTIVE' if ir['fed_funds_rate'] > 4.5 else 'NEUTRAL',
                'Gold': 'BEARISH' if ir['us_10y_yield'] > 4.5 else 'NEUTRAL',
                'Equities': 'BEARISH' if yield_curve < 0 else 'NEUTRAL',
                'Bonds': 'BEARISH' if ir['fed_funds_rate'] > 5.0 else 'NEUTRAL'
            }
        }
        
        # Inflation
        infl = indicators['inflation']
        cpi = infl['us_cpi']
        
        if cpi > 3.5:
            infl_explanation = f"Elevated inflation at {cpi}% keeps Fed hawkish. "
            infl_explanation += f"Core CPI at {infl['us_core_cpi']}% shows persistent price pressures. "
            infl_explanation += "Expect continued tight monetary policy until inflation moderates toward 2% target."
        elif cpi < 2.5:
            infl_explanation = f"Inflation at {cpi}% approaching Fed's 2% target. "
            infl_explanation += "This gives Fed flexibility to pivot toward rate cuts. "
            infl_explanation += "Lower inflation typically positive for risk assets and negative for USD."
        else:
            infl_explanation = f"Inflation at {cpi}% in target range. "
            infl_explanation += "Fed likely to maintain current policy stance. "
            infl_explanation += "Markets will focus on inflation trajectory for policy clues."
        
        explanations['inflation'] = {
            'explanation': infl_explanation,
            'market_impact': {
                'USD': 'BULLISH' if cpi > 3.5 else 'BEARISH' if cpi < 2.5 else 'NEUTRAL',
                'Gold': 'BULLISH' if cpi > 3.5 else 'NEUTRAL',
                'Equities': 'BEARISH' if cpi > 3.5 else 'BULLISH',
                'Bonds': 'BEARISH' if cpi > 3.5 else 'BULLISH'
            }
        }
        
        # Employment
        emp = indicators['employment']
        unemployment = emp['us_unemployment']
        
        if unemployment < 4.0:
            emp_explanation = f"Tight labor market with {unemployment}% unemployment. "
            emp_explanation += f"Nonfarm payrolls at {emp['us_nonfarm_payrolls']:,} show solid job creation. "
            emp_explanation += "Strong employment supports consumer spending but may keep Fed hawkish due to wage pressures."
        else:
            emp_explanation = f"Unemployment at {unemployment}% showing labor market cooling. "
            emp_explanation += "Weaker job market may accelerate Fed's pivot to rate cuts. "
            emp_explanation += "However, could signal economic slowdown concerns."
        
        explanations['employment'] = {
            'explanation': emp_explanation,
            'market_impact': {
                'USD': 'BULLISH' if unemployment < 4.0 else 'BEARISH',
                'Gold': 'NEUTRAL',
                'Equities': 'BULLISH' if unemployment < 4.5 else 'BEARISH',
                'Bonds': 'BEARISH' if unemployment < 4.0 else 'BULLISH'
            }
        }
        
        return explanations
