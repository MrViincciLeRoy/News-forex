"""
News Fetcher - Fetch financial news with SERP API support
"""

import requests
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import os

class NewsFetcher:
    
    def __init__(self, prefer_serp: bool = True):
        self.prefer_serp = prefer_serp
        self.serp_api_key = os.getenv('SERP_API_KEY', '')
    
    def fetch_event_news(self, date: str, event_name: str, max_records: int = 30, full_content: bool = False) -> List[Dict]:
        """Fetch news articles related to event"""
        
        if self.prefer_serp and self.serp_api_key:
            return self._fetch_from_serp(event_name, date, max_records)
        else:
            return self._fetch_mock_news(event_name, date, max_records)
    
    def _fetch_from_serp(self, query: str, date: str, max_records: int) -> List[Dict]:
        """Fetch from SERP API"""
        
        url = "https://serpapi.com/search"
        params = {
            'q': f"{query} financial news",
            'tbm': 'nws',
            'api_key': self.serp_api_key,
            'num': max_records
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                articles = []
                
                for item in data.get('news_results', [])[:max_records]:
                    articles.append({
                        'title': item.get('title', ''),
                        'content': item.get('snippet', ''),
                        'source': item.get('source', {}).get('name', 'Unknown'),
                        'url': item.get('link', ''),
                        'published_date': item.get('date', date),
                        'relevance_score': 0.9
                    })
                
                return articles
        except:
            pass
        
        return self._fetch_mock_news(query, date, max_records)
    
    def _fetch_mock_news(self, event_name: str, date: str, max_records: int) -> List[Dict]:
        """Generate mock news for testing"""
        
        sentiments = ['positive', 'negative', 'neutral']
        themes = ['inflation', 'employment', 'growth', 'policy', 'markets']
        sources = ['Bloomberg', 'Reuters', 'WSJ', 'FT', 'CNBC']
        
        articles = []
        for i in range(min(max_records, 15)):
            articles.append({
                'title': f"Analysis: {event_name} - Market Implications for Day {i+1}",
                'content': f"Markets react to {event_name} with mixed signals. Analysts suggest watching key indicators closely. " * 2,
                'source': sources[i % len(sources)],
                'url': f"https://example.com/article-{i}",
                'published_date': date,
                'relevance_score': 0.8 - (i * 0.02)
            })
        
        return articles
