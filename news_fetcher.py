"""
Enhanced News Fetcher with SERP Integration
Uses SerpAPI for comprehensive news gathering with full article content
"""

import requests
from datetime import datetime, timedelta
import json
import time
from typing import List, Dict, Optional
import feedparser
from bs4 import BeautifulSoup
import os

try:
    from serp_news_fetcher import SerpNewsFetcher
    SERP_AVAILABLE = True
except ImportError:
    SERP_AVAILABLE = False


class NewsFetcher:
    """
    Multi-source news fetcher with SERP integration
    Priority: SERP API > RSS feeds > fallback sources
    """
    
    def __init__(self, prefer_serp=True):
        self.prefer_serp = prefer_serp and SERP_AVAILABLE
        self.serp_fetcher = SerpNewsFetcher() if SERP_AVAILABLE else None
        
        self.newsapi_key = os.environ.get('NEWSAPI_KEY', '')
        self.bing_key = os.environ.get('BING_NEWS_KEY', '')
        
        self.rss_feeds = {
            'google_news': 'https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en',
            'yahoo_finance': 'https://finance.yahoo.com/rss/headline',
            'reuters_business': 'https://www.reutersagency.com/feed/?taxonomy=best-topics&post_type=best',
        }
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def fetch_event_news(
        self,
        date: str,
        event_name: str,
        max_records: int = 20,
        full_content: bool = True
    ) -> List[Dict]:
        """
        Fetch news for specific event with SERP priority
        
        Args:
            date: Date string 'YYYY-MM-DD'
            event_name: Event name (e.g., 'Non-Farm Payrolls')
            max_records: Maximum articles (default 20)
            full_content: Extract full article content (default True)
            
        Returns:
            List of article dictionaries with full content
        """
        print(f"\n{'='*80}")
        print(f"FETCHING NEWS: {event_name}")
        print(f"{'='*80}")
        print(f"Date: {date}")
        print(f"Max Articles: {max_records}")
        print(f"Full Content: {full_content}")
        print(f"SERP Available: {SERP_AVAILABLE}")
        
        all_articles = []
        
        # PRIORITY 1: Try SERP if available
        if self.prefer_serp and self.serp_fetcher:
            try:
                print(f"\n[1/3] Trying SERP API...")
                print("-"*80)
                
                serp_results = self.serp_fetcher.fetch_comprehensive_news(
                    query=event_name,
                    date=date,
                    max_articles=max_records,
                    full_content=full_content
                )
                
                if serp_results and serp_results.get('articles'):
                    articles = self._convert_serp_format(serp_results['articles'])
                    all_articles.extend(articles)
                    
                    print(f"\n✓ SERP: {len(articles)} articles")
                    print(f"  Full content: {serp_results['statistics']['full_content_extracted']}")
                    print(f"  Sources: {serp_results['statistics']['sources_count']}")
                    
                    if len(all_articles) >= max_records:
                        return self._finalize_articles(all_articles[:max_records])
            
            except Exception as e:
                print(f"\n✗ SERP failed: {str(e)[:60]}")
        
        # PRIORITY 2: RSS feeds
        if len(all_articles) < max_records:
            try:
                print(f"\n[2/3] Trying RSS feeds...")
                print("-"*80)
                
                rss_articles = self._fetch_from_rss(event_name, date, max_records - len(all_articles))
                
                if rss_articles:
                    all_articles.extend(rss_articles)
                    print(f"✓ RSS: {len(rss_articles)} articles")
            
            except Exception as e:
                print(f"✗ RSS failed: {str(e)[:60]}")
        
        # PRIORITY 3: Other APIs
        if len(all_articles) < max_records:
            try:
                print(f"\n[3/3] Trying fallback APIs...")
                print("-"*80)
                
                fallback_articles = self._fetch_fallback(event_name, date, max_records - len(all_articles))
                
                if fallback_articles:
                    all_articles.extend(fallback_articles)
                    print(f"✓ Fallback: {len(fallback_articles)} articles")
            
            except Exception as e:
                print(f"✗ Fallback failed: {str(e)[:60]}")
        
        final_articles = self._finalize_articles(all_articles[:max_records])
        
        print(f"\n{'='*80}")
        print(f"✓ FETCH COMPLETE: {len(final_articles)} total articles")
        print(f"{'='*80}\n")
        
        return final_articles
    
    def _convert_serp_format(self, serp_articles: List[Dict]) -> List[Dict]:
        """Convert SERP format to standard format"""
        converted = []
        
        for article in serp_articles:
            converted.append({
                'title': article.get('title', ''),
                'content': article.get('full_content', article.get('snippet', '')),
                'url': article.get('url', ''),
                'published_date': article.get('date', ''),
                'source': f"SERP - {article.get('source', 'Unknown')}",
                'relevance_score': 0.95,
                'content_length': article.get('content_length', 0),
                'extraction_method': 'serp_api'
            })
        
        return converted
    
    def _fetch_from_rss(self, event_name: str, date: str, max_records: int) -> List[Dict]:
        """Fetch from RSS feeds"""
        date_obj = datetime.strptime(date, '%Y-%m-%d')
        all_articles = []
        
        # Google News RSS
        try:
            query = self._generate_search_query(event_name, date_obj)
            url = self.rss_feeds['google_news'].format(query=requests.utils.quote(query))
            
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            feed = feedparser.parse(response.content)
            
            for entry in feed.entries[:max_records * 2]:
                pub_date = None
                if hasattr(entry, 'published_parsed'):
                    pub_date = datetime(*entry.published_parsed[:6])
                
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
                    'relevance_score': 0.8,
                    'extraction_method': 'rss'
                }
                
                all_articles.append(article)
                
                if len(all_articles) >= max_records:
                    break
        
        except Exception as e:
            pass
        
        return all_articles
    
    def _fetch_fallback(self, event_name: str, date: str, max_records: int) -> List[Dict]:
        """Fallback to NewsAPI or Bing if available"""
        articles = []
        
        if self.newsapi_key:
            try:
                date_obj = datetime.strptime(date, '%Y-%m-%d')
                from_date = (date_obj - timedelta(days=1)).strftime('%Y-%m-%d')
                to_date = (date_obj + timedelta(days=1)).strftime('%Y-%m-%d')
                
                url = 'https://newsapi.org/v2/everything'
                params = {
                    'q': event_name,
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
                
                for item in data.get('articles', []):
                    article = {
                        'title': item.get('title', ''),
                        'content': item.get('description', '') or item.get('content', ''),
                        'url': item.get('url', ''),
                        'published_date': item.get('publishedAt', date),
                        'source': f"NewsAPI - {item.get('source', {}).get('name', 'Unknown')}",
                        'relevance_score': 0.85,
                        'extraction_method': 'newsapi'
                    }
                    articles.append(article)
            
            except Exception as e:
                pass
        
        return articles
    
    def _generate_search_query(self, event_name: str, date_obj: datetime) -> str:
        """Generate optimized search query"""
        event_keywords = {
            'non-farm payrolls': ['jobs report', 'employment', 'unemployment', 'NFP'],
            'cpi': ['inflation', 'consumer prices', 'CPI'],
            'fomc': ['federal reserve', 'interest rates', 'fed meeting'],
            'gdp': ['economic growth', 'GDP', 'gross domestic product'],
            'retail sales': ['consumer spending', 'retail'],
        }
        
        event_lower = event_name.lower()
        keywords = [event_name]
        
        for key, terms in event_keywords.items():
            if key in event_lower:
                keywords.extend(terms)
                break
        
        month_year = date_obj.strftime('%B %Y')
        keywords.append(month_year)
        
        return ' OR '.join(keywords[:3])
    
    def _finalize_articles(self, articles: List[Dict]) -> List[Dict]:
        """Deduplicate and sort articles"""
        seen_urls = set()
        seen_titles = set()
        unique = []
        
        for article in articles:
            url = article.get('url', '')
            title = article.get('title', '').lower().strip()
            
            url_key = url.split('?')[0]
            title_key = ''.join(c for c in title if c.isalnum())[:100]
            
            if url_key not in seen_urls and title_key not in seen_titles:
                seen_urls.add(url_key)
                seen_titles.add(title_key)
                unique.append(article)
        
        unique = sorted(
            unique,
            key=lambda x: (x.get('relevance_score', 0), len(x.get('content', ''))),
            reverse=True
        )
        
        return unique
    
    def fetch_news(self, date: str, max_records: int = 20) -> List[Dict]:
        """Fetch general financial news for a date"""
        date_obj = datetime.strptime(date, '%Y-%m-%d')
        query = f"markets finance economy stocks {date_obj.strftime('%B %Y')}"
        
        return self.fetch_event_news(date, query, max_records)
    
    def get_affected_symbols(self, event_name: str, text: str) -> List[str]:
        """Extract financial symbols from event and text"""
        symbols = set()
        
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


if __name__ == "__main__":
    fetcher = NewsFetcher(prefer_serp=True)
    
    print("="*80)
    print("NEWS FETCHER TEST WITH SERP INTEGRATION")
    print("="*80)
    
    print("\nTest: Non-Farm Payrolls (2024-11-01)")
    print("-" * 80)
    
    articles = fetcher.fetch_event_news(
        date='2024-11-01',
        event_name='Non-Farm Payrolls',
        max_records=20,
        full_content=True
    )
    
    print(f"\n✓ Fetched {len(articles)} articles\n")
    
    for i, article in enumerate(articles[:3], 1):
        print(f"{i}. {article['title'][:70]}...")
        print(f"   Source: {article['source']}")
        print(f"   Content: {len(article.get('content', ''))} chars")
        print(f"   Method: {article.get('extraction_method', 'unknown')}")
        print()
    
    symbols = fetcher.get_affected_symbols(
        'Non-Farm Payrolls',
        ' '.join([a['title'] + ' ' + a.get('content', '') for a in articles])
    )
    print(f"Extracted symbols: {', '.join(symbols)}")
    
    print("\n" + "="*80)
    print("✓ TEST COMPLETE")
    print("="*80)
