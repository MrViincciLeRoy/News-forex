"""
Enhanced News Fetcher with Multiple Sources
Fixes timeout issues by trying multiple news APIs and RSS feeds
"""

import requests
from datetime import datetime, timedelta
import json
import time
from typing import List, Dict, Optional
import feedparser
from bs4 import BeautifulSoup
import os


class EnhancedNewsFetcher:
    """
    Multi-source news fetcher with fallbacks:
    1. NewsAPI (if API key available)
    2. Google News RSS
    3. Bing News API
    4. Yahoo Finance RSS
    5. Reuters RSS
    """
    
    def __init__(self):
        self.newsapi_key = os.environ.get('NEWSAPI_KEY', '')
        self.bing_key = os.environ.get('BING_NEWS_KEY', '')
        
        # RSS feeds (no API key needed!)
        self.rss_feeds = {
            'google_news': 'https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en',
            'yahoo_finance': 'https://finance.yahoo.com/rss/headline',
            'reuters_business': 'https://www.reutersagency.com/feed/?taxonomy=best-topics&post_type=best',
        }
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def fetch_event_news(self, date: str, event_name: str, max_records: int = 10) -> List[Dict]:
        """
        Fetch news for specific event with multiple fallbacks
        
        Args:
            date: Date string 'YYYY-MM-DD'
            event_name: Event name (e.g., 'Non-Farm Payrolls')
            max_records: Maximum articles to return
            
        Returns:
            List of article dictionaries
        """
        print(f"\nFetching news for event: {event_name}")
        
        date_obj = datetime.strptime(date, '%Y-%m-%d')
        
        # Generate search query
        query = self._generate_search_query(event_name, date_obj)
        print(f"  Search query: '{query}'")
        
        all_articles = []
        
        # Try multiple sources in order
        sources_tried = []
        
        # 1. Try Google News RSS (most reliable, no API key needed)
        try:
            print(f"  [1/4] Trying Google News RSS...")
            articles = self._fetch_google_news_rss(query, date_obj, max_records)
            if articles:
                all_articles.extend(articles)
                sources_tried.append(f"Google RSS ({len(articles)})")
                print(f"  ✓ Google News RSS: {len(articles)} articles")
        except Exception as e:
            print(f"  ✗ Google News RSS failed: {str(e)[:50]}")
        
        # 2. Try NewsAPI if key available
        if self.newsapi_key and len(all_articles) < max_records:
            try:
                print(f"  [2/4] Trying NewsAPI...")
                articles = self._fetch_newsapi(query, date_obj, max_records - len(all_articles))
                if articles:
                    all_articles.extend(articles)
                    sources_tried.append(f"NewsAPI ({len(articles)})")
                    print(f"  ✓ NewsAPI: {len(articles)} articles")
            except Exception as e:
                print(f"  ✗ NewsAPI failed: {str(e)[:50]}")
        
        # 3. Try Yahoo Finance RSS
        if len(all_articles) < max_records:
            try:
                print(f"  [3/4] Trying Yahoo Finance RSS...")
                articles = self._fetch_yahoo_finance_rss(event_name, date_obj, max_records - len(all_articles))
                if articles:
                    all_articles.extend(articles)
                    sources_tried.append(f"Yahoo RSS ({len(articles)})")
                    print(f"  ✓ Yahoo Finance RSS: {len(articles)} articles")
            except Exception as e:
                print(f"  ✗ Yahoo Finance RSS failed: {str(e)[:50]}")
        
        # 4. Try Bing News if key available
        if self.bing_key and len(all_articles) < max_records:
            try:
                print(f"  [4/4] Trying Bing News API...")
                articles = self._fetch_bing_news(query, date_obj, max_records - len(all_articles))
                if articles:
                    all_articles.extend(articles)
                    sources_tried.append(f"Bing ({len(articles)})")
                    print(f"  ✓ Bing News: {len(articles)} articles")
            except Exception as e:
                print(f"  ✗ Bing News failed: {str(e)[:50]}")
        
        # Remove duplicates
        all_articles = self._deduplicate_articles(all_articles)
        
        # Sort by relevance/date
        all_articles = sorted(
            all_articles,
            key=lambda x: (x.get('relevance_score', 0), x.get('published_date', '')),
            reverse=True
        )[:max_records]
        
        print(f"\n  ✓ Total fetched: {len(all_articles)} articles")
        print(f"  Sources: {', '.join(sources_tried) if sources_tried else 'None'}")
        
        return all_articles
    
    def _generate_search_query(self, event_name: str, date_obj: datetime) -> str:
        """Generate optimized search query"""
        
        # Event-specific keywords
        event_keywords = {
            'non-farm payrolls': ['jobs report', 'employment', 'unemployment', 'NFP'],
            'cpi': ['inflation', 'consumer prices', 'CPI'],
            'fomc': ['federal reserve', 'interest rates', 'fed meeting'],
            'gdp': ['economic growth', 'GDP', 'gross domestic product'],
            'retail sales': ['consumer spending', 'retail'],
        }
        
        event_lower = event_name.lower()
        
        # Find matching keywords
        keywords = [event_name]
        for key, terms in event_keywords.items():
            if key in event_lower:
                keywords.extend(terms)
                break
        
        # Add date context
        month_year = date_obj.strftime('%B %Y')
        keywords.append(month_year)
        
        return ' OR '.join(keywords[:3])  # Limit to avoid too long queries
    
    def _fetch_google_news_rss(self, query: str, date_obj: datetime, max_records: int) -> List[Dict]:
        """Fetch from Google News RSS (no API key needed!)"""
        
        url = self.rss_feeds['google_news'].format(query=requests.utils.quote(query))
        
        response = self.session.get(url, timeout=10)
        response.raise_for_status()
        
        feed = feedparser.parse(response.content)
        
        articles = []
        for entry in feed.entries[:max_records * 2]:  # Get extra for filtering
            
            # Parse date
            pub_date = None
            if hasattr(entry, 'published_parsed'):
                pub_date = datetime(*entry.published_parsed[:6])
            
            # Check if within date range (±3 days)
            if pub_date:
                days_diff = abs((pub_date.date() - date_obj.date()).days)
                if days_diff > 3:
                    continue
            
            article = {
                'title': entry.get('title', ''),
                'content': entry.get('summary', ''),
                'url': entry.get('link', ''),
                'published_date': pub_date.isoformat() if pub_date else date_obj.isoformat(),
                'source': 'Google News RSS',
                'relevance_score': 0.8
            }
            
            articles.append(article)
            
            if len(articles) >= max_records:
                break
        
        return articles
    
    def _fetch_newsapi(self, query: str, date_obj: datetime, max_records: int) -> List[Dict]:
        """Fetch from NewsAPI (requires API key)"""
        
        from_date = (date_obj - timedelta(days=1)).strftime('%Y-%m-%d')
        to_date = (date_obj + timedelta(days=1)).strftime('%Y-%m-%d')
        
        url = 'https://newsapi.org/v2/everything'
        params = {
            'q': query,
            'from': from_date,
            'to': to_date,
            'language': 'en',
            'sortBy': 'relevancy',
            'pageSize': max_records,
            'apiKey': self.newsapi_key
        }
        
        response = self.session.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        articles = []
        for item in data.get('articles', []):
            article = {
                'title': item.get('title', ''),
                'content': item.get('description', '') or item.get('content', ''),
                'url': item.get('url', ''),
                'published_date': item.get('publishedAt', date_obj.isoformat()),
                'source': f"NewsAPI - {item.get('source', {}).get('name', 'Unknown')}",
                'relevance_score': 0.9
            }
            articles.append(article)
        
        return articles
    
    def _fetch_yahoo_finance_rss(self, event_name: str, date_obj: datetime, max_records: int) -> List[Dict]:
        """Fetch from Yahoo Finance RSS"""
        
        url = self.rss_feeds['yahoo_finance']
        
        response = self.session.get(url, timeout=10)
        response.raise_for_status()
        
        feed = feedparser.parse(response.content)
        
        articles = []
        event_keywords = event_name.lower().split()
        
        for entry in feed.entries[:max_records * 3]:
            
            # Filter by keyword relevance
            title = entry.get('title', '').lower()
            summary = entry.get('summary', '').lower()
            
            relevance = sum(1 for kw in event_keywords if kw in title or kw in summary)
            if relevance == 0:
                continue
            
            pub_date = None
            if hasattr(entry, 'published_parsed'):
                pub_date = datetime(*entry.published_parsed[:6])
            
            article = {
                'title': entry.get('title', ''),
                'content': entry.get('summary', ''),
                'url': entry.get('link', ''),
                'published_date': pub_date.isoformat() if pub_date else date_obj.isoformat(),
                'source': 'Yahoo Finance RSS',
                'relevance_score': 0.7 + (relevance * 0.1)
            }
            
            articles.append(article)
            
            if len(articles) >= max_records:
                break
        
        return articles
    
    def _fetch_bing_news(self, query: str, date_obj: datetime, max_records: int) -> List[Dict]:
        """Fetch from Bing News API (requires API key)"""
        
        url = 'https://api.bing.microsoft.com/v7.0/news/search'
        headers = {'Ocp-Apim-Subscription-Key': self.bing_key}
        params = {
            'q': query,
            'count': max_records,
            'mkt': 'en-US',
            'freshness': 'Week'
        }
        
        response = self.session.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        articles = []
        for item in data.get('value', []):
            article = {
                'title': item.get('name', ''),
                'content': item.get('description', ''),
                'url': item.get('url', ''),
                'published_date': item.get('datePublished', date_obj.isoformat()),
                'source': f"Bing - {item.get('provider', [{}])[0].get('name', 'Unknown')}",
                'relevance_score': 0.85
            }
            articles.append(article)
        
        return articles
    
    def _deduplicate_articles(self, articles: List[Dict]) -> List[Dict]:
        """Remove duplicate articles based on title similarity"""
        
        unique = []
        seen_titles = set()
        
        for article in articles:
            title = article.get('title', '').lower().strip()
            
            # Simple deduplication based on title
            title_key = ''.join(c for c in title if c.isalnum())[:50]
            
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                unique.append(article)
        
        return unique
    
    def fetch_news(self, date: str, max_records: int = 10) -> List[Dict]:
        """Fetch general financial news for a date"""
        
        date_obj = datetime.strptime(date, '%Y-%m-%d')
        
        # Generic financial news query
        query = f"markets finance economy stocks {date_obj.strftime('%B %Y')}"
        
        return self.fetch_event_news(date, query, max_records)
    
    def get_affected_symbols(self, event_name: str, text: str) -> List[str]:
        """Extract financial symbols from event and text"""
        
        symbols = set()
        
        # Event-specific symbol mappings
        event_symbols = {
            'non-farm payrolls': ['DX-Y.NYB', 'GC=F', '^GSPC', 'EURUSD=X', '^TNX'],
            'cpi': ['GC=F', 'DX-Y.NYB', '^GSPC', 'TLT', 'EURUSD=X'],
            'fomc': ['^TNX', 'DX-Y.NYB', '^GSPC', 'GC=F', 'EURUSD=X'],
            'gdp': ['^GSPC', 'DX-Y.NYB', '^TNX', 'EURUSD=X'],
            'retail sales': ['^GSPC', 'EURUSD=X', 'GC=F'],
        }
        
        event_lower = event_name.lower()
        for key, symbol_list in event_symbols.items():
            if key in event_lower:
                symbols.update(symbol_list)
                break
        
        # Text-based extraction
        text_upper = text.upper()
        
        symbol_keywords = {
            'GOLD': 'GC=F',
            'DOLLAR': 'DX-Y.NYB',
            'EUR': 'EURUSD=X',
            'S&P': '^GSPC',
            'TREASURY': '^TNX',
            'NASDAQ': '^IXIC',
        }
        
        for keyword, symbol in symbol_keywords.items():
            if keyword in text_upper:
                symbols.add(symbol)
        
        return list(symbols)


class NewsFetcher(EnhancedNewsFetcher):
    """Backward compatibility alias"""
    pass


if __name__ == "__main__":
    fetcher = EnhancedNewsFetcher()
    
    print("="*80)
    print("ENHANCED NEWS FETCHER TEST")
    print("="*80)
    
    # Test 1: Non-Farm Payrolls
    print("\nTest 1: Non-Farm Payrolls (2024-11-01)")
    print("-" * 80)
    
    articles = fetcher.fetch_event_news('2024-11-01', 'Non-Farm Payrolls', max_records=5)
    
    print(f"\n✓ Fetched {len(articles)} articles\n")
    
    for i, article in enumerate(articles[:3], 1):
        print(f"{i}. {article['title']}")
        print(f"   Source: {article['source']}")
        print(f"   URL: {article['url'][:60]}...")
        print()
    
    # Test 2: Symbol extraction
    print("\nTest 2: Symbol Extraction")
    print("-" * 80)
    
    if articles:
        text = ' '.join([a['title'] + ' ' + a['content'] for a in articles])
        symbols = fetcher.get_affected_symbols('Non-Farm Payrolls', text)
        print(f"Extracted symbols: {', '.join(symbols)}")
    
    print("\n" + "="*80)
    print("✓ TEST COMPLETE")
    print("="*80)
