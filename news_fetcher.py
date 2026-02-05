"""
News Fetcher
Retrieves financial news from GDELT API for any given date or date range
Provides clean, structured news data for downstream analysis
"""

import pandas as pd
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import json
import time


class NewsFetcher:
    """
    Fetches financial news from GDELT Project API
    Supports single date, date range, and event-specific queries
    """
    
    def __init__(self):
        self.base_url = "https://api.gdeltproject.org/api/v2/doc/doc"
        self.cache = {}
        self.rate_limit_delay = 0.5  # seconds between requests
        
        # Predefined search queries for common financial events
        self.event_queries = {
            'Non-Farm Payrolls': 'nonfarm payrolls jobs report employment',
            'NFP': 'nonfarm payrolls jobs report employment',
            'Consumer Price Index': 'CPI inflation consumer prices',
            'CPI': 'CPI inflation consumer prices',
            'Producer Price Index': 'PPI producer prices inflation',
            'PPI': 'PPI producer prices inflation',
            'Retail Sales': 'retail sales consumer spending economy',
            'ISM Manufacturing': 'ISM manufacturing PMI economy index',
            'ISM Services': 'ISM services PMI economy index',
            'FOMC': 'Federal Reserve FOMC interest rate decision',
            'Fed': 'Federal Reserve interest rate monetary policy',
            'Jobless Claims': 'unemployment jobless claims weekly',
            'GDP': 'GDP gross domestic product economy growth',
            'Unemployment': 'unemployment rate jobs labor market',
            'Housing': 'housing starts building permits real estate',
            'Gold': 'gold prices precious metals XAU',
            'Oil': 'oil prices crude WTI Brent energy',
            'Dollar': 'dollar DXY currency forex exchange rate',
            'Stock Market': 'stock market S&P Dow NASDAQ equity',
            'Treasury': 'treasury bonds yields interest rates',
            'Inflation': 'inflation prices CPI PPI cost of living',
            'Central Bank': 'central bank monetary policy rate decision'
        }
    
    def fetch_news(self, date: str, query: str = 'economy finance market', 
                   max_records: int = 10, language: str = 'english',
                   timespan: str = 'day') -> List[Dict]:
        """
        Fetch news for a specific date
        
        Args:
            date: Date in 'YYYY-MM-DD' format
            query: Search query terms
            max_records: Maximum number of articles to retrieve
            language: Article language filter
            timespan: Time window ('day', 'hour', 'week')
            
        Returns:
            List of news articles with metadata
        """
        # Parse date
        try:
            date_obj = datetime.strptime(date, '%Y-%m-%d')
        except ValueError:
            print(f"Invalid date format: {date}. Use YYYY-MM-DD")
            return []
        
        # Create cache key
        cache_key = f"{date}_{query}_{max_records}"
        if cache_key in self.cache:
            print(f"  Using cached results for {date}")
            return self.cache[cache_key]
        
        # Build datetime range
        if timespan == 'day':
            start_datetime = date_obj.strftime('%Y%m%d') + '000000'
            end_datetime = date_obj.strftime('%Y%m%d') + '235959'
        elif timespan == 'hour':
            start_datetime = date_obj.strftime('%Y%m%d%H') + '0000'
            end_datetime = date_obj.strftime('%Y%m%d%H') + '5959'
        elif timespan == 'week':
            start_datetime = (date_obj - timedelta(days=3)).strftime('%Y%m%d') + '000000'
            end_datetime = (date_obj + timedelta(days=3)).strftime('%Y%m%d') + '235959'
        else:
            start_datetime = date_obj.strftime('%Y%m%d') + '000000'
            end_datetime = date_obj.strftime('%Y%m%d') + '235959'
        
        # API parameters
        params = {
            'query': query,
            'mode': 'artlist',
            'maxrecords': max_records * 2,  # Request more, filter later
            'format': 'json',
            'startdatetime': start_datetime,
            'enddatetime': end_datetime,
            'sourcelang': language
        }
        
        print(f"  Fetching news for {date} | Query: '{query}'")
        
        try:
            response = requests.get(self.base_url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                articles = []
                
                if 'articles' in data and data['articles']:
                    for article in data['articles'][:max_records]:
                        articles.append({
                            'title': article.get('title', 'No title'),
                            'url': article.get('url', ''),
                            'source': article.get('domain', 'Unknown'),
                            'language': article.get('language', 'unknown'),
                            'date': date,
                            'seendate': article.get('seendate', ''),
                            'socialimage': article.get('socialimage', ''),
                            'domain_country': article.get('sourcecountry', '')
                        })
                    
                    print(f"  ✓ Retrieved {len(articles)} articles")
                    
                    # Cache results
                    self.cache[cache_key] = articles
                    
                    # Rate limiting
                    time.sleep(self.rate_limit_delay)
                    
                    return articles
                else:
                    print(f"  ✗ No articles found for {date}")
                    return []
            else:
                print(f"  ✗ Request failed: HTTP {response.status_code}")
                return []
                
        except requests.exceptions.Timeout:
            print(f"  ✗ Request timeout for {date}")
            return []
        except requests.exceptions.RequestException as e:
            print(f"  ✗ Request error: {e}")
            return []
        except json.JSONDecodeError:
            print(f"  ✗ Failed to parse JSON response")
            return []
    
    def fetch_news_range(self, start_date: str, end_date: str, 
                         query: str = 'economy finance market',
                         max_records_per_day: int = 5) -> Dict[str, List[Dict]]:
        """
        Fetch news for a date range
        
        Args:
            start_date: Start date 'YYYY-MM-DD'
            end_date: End date 'YYYY-MM-DD'
            query: Search query
            max_records_per_day: Articles per day
            
        Returns:
            Dictionary mapping dates to article lists
        """
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        if start > end:
            print("Start date must be before end date")
            return {}
        
        delta = (end - start).days + 1
        print(f"\nFetching news from {start_date} to {end_date} ({delta} days)")
        
        results = {}
        
        for i in range(delta):
            current_date = start + timedelta(days=i)
            date_str = current_date.strftime('%Y-%m-%d')
            
            # Skip weekends for financial news (optional)
            if current_date.weekday() >= 5:  # Saturday=5, Sunday=6
                print(f"  Skipping weekend: {date_str}")
                continue
            
            articles = self.fetch_news(date_str, query, max_records_per_day)
            if articles:
                results[date_str] = articles
        
        print(f"\n✓ Fetched news for {len(results)} days")
        return results
    
    def fetch_event_news(self, event_date: str, event_name: str,
                         max_records: int = 10, 
                         custom_query: Optional[str] = None) -> List[Dict]:
        """
        Fetch news for a specific economic event
        
        Args:
            event_date: Date of event 'YYYY-MM-DD'
            event_name: Name of event (e.g., 'Non-Farm Payrolls')
            max_records: Maximum articles to retrieve
            custom_query: Optional custom search query
            
        Returns:
            List of relevant articles
        """
        # Get predefined query or use custom
        if custom_query:
            query = custom_query
        else:
            # Try to match event name to predefined queries
            query = None
            for key, value in self.event_queries.items():
                if key.lower() in event_name.lower():
                    query = value
                    break
            
            if not query:
                # Fallback to event name
                query = event_name
        
        print(f"\nFetching news for event: {event_name}")
        
        articles = self.fetch_news(event_date, query, max_records)
        
        # Enhance with event metadata
        for article in articles:
            article['event_name'] = event_name
            article['event_date'] = event_date
        
        return articles
    
    def fetch_multiple_events(self, events: List[Dict],
                             max_records_per_event: int = 5) -> Dict:
        """
        Fetch news for multiple events
        
        Args:
            events: List of event dicts with 'date' and 'event' keys
            max_records_per_event: Articles per event
            
        Returns:
            Dictionary with event data and news
        """
        print(f"\nFetching news for {len(events)} events")
        
        results = []
        
        for i, event in enumerate(events, 1):
            event_date = event.get('date')
            event_name = event.get('event')
            
            if not event_date or not event_name:
                print(f"  Skipping invalid event: {event}")
                continue
            
            print(f"\n[{i}/{len(events)}] {event_name} on {event_date}")
            
            articles = self.fetch_event_news(
                event_date, 
                event_name, 
                max_records_per_event
            )
            
            results.append({
                'event_date': event_date,
                'event_name': event_name,
                'articles_count': len(articles),
                'articles': articles
            })
        
        return {
            'total_events': len(events),
            'events_with_news': len([r for r in results if r['articles_count'] > 0]),
            'total_articles': sum(r['articles_count'] for r in results),
            'events': results
        }
    
    def get_trending_topics(self, date: str, num_queries: int = 5) -> List[Dict]:
        """
        Fetch news for multiple trending financial topics on a date
        
        Args:
            date: Date 'YYYY-MM-DD'
            num_queries: Number of different topics to query
            
        Returns:
            List of results for different topics
        """
        trending_queries = [
            'stock market trading S&P500 Dow',
            'Federal Reserve interest rates FOMC',
            'inflation CPI prices economy',
            'gold silver commodities precious metals',
            'dollar currency forex exchange rates',
            'oil energy crude prices',
            'Bitcoin cryptocurrency digital assets',
            'employment jobs labor market'
        ]
        
        results = []
        
        for query in trending_queries[:num_queries]:
            articles = self.fetch_news(date, query, max_records=3)
            if articles:
                results.append({
                    'topic': query,
                    'article_count': len(articles),
                    'articles': articles
                })
        
        return results
    
    def save_to_json(self, data: Dict, filename: str):
        """Save news data to JSON file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Saved to {filename}")
    
    def save_to_csv(self, articles: List[Dict], filename: str):
        """Save articles to CSV file"""
        if not articles:
            print("No articles to save")
            return
        
        df = pd.DataFrame(articles)
        df.to_csv(filename, index=False, encoding='utf-8')
        print(f"✓ Saved to {filename}")
    
    def get_news_summary(self, articles: List[Dict]) -> Dict:
        """Generate summary statistics for news articles"""
        if not articles:
            return {
                'total_articles': 0,
                'unique_sources': 0,
                'date_range': None
            }
        
        sources = [a.get('source', 'Unknown') for a in articles]
        dates = [a.get('date', '') for a in articles if a.get('date')]
        
        return {
            'total_articles': len(articles),
            'unique_sources': len(set(sources)),
            'top_sources': pd.Series(sources).value_counts().head(5).to_dict(),
            'date_range': {
                'start': min(dates) if dates else None,
                'end': max(dates) if dates else None
            },
            'sample_titles': [a.get('title', '') for a in articles[:5]]
        }


if __name__ == "__main__":
    print("="*80)
    print("NEWS FETCHER - GDELT API")
    print("="*80)
    
    fetcher = NewsFetcher()
    
    # Test 1: Single date news fetch
    print("\n" + "="*80)
    print("TEST 1: Fetch news for a specific date")
    print("="*80)
    
    test_date = '2024-11-01'
    articles = fetcher.fetch_news(
        date=test_date,
        query='economy finance market',
        max_records=10
    )
    
    if articles:
        print(f"\nRetrieved {len(articles)} articles:")
        for i, article in enumerate(articles[:5], 1):
            print(f"\n{i}. {article['title']}")
            print(f"   Source: {article['source']}")
            print(f"   URL: {article['url'][:60]}...")
    
    # Test 2: Event-specific news
    print("\n" + "="*80)
    print("TEST 2: Fetch news for specific economic events")
    print("="*80)
    
    test_events = [
        {'date': '2024-11-01', 'event': 'Non-Farm Payrolls'},
        {'date': '2024-10-10', 'event': 'Consumer Price Index'},
        {'date': '2024-09-18', 'event': 'FOMC Decision'}
    ]
    
    event_results = fetcher.fetch_multiple_events(test_events, max_records_per_event=3)
    
    print(f"\nEvents processed: {event_results['events_with_news']}/{event_results['total_events']}")
    print(f"Total articles: {event_results['total_articles']}")
    
    for event in event_results['events']:
        if event['articles_count'] > 0:
            print(f"\n{event['event_name']} ({event['event_date']}): {event['articles_count']} articles")
            for article in event['articles'][:2]:
                print(f"  • {article['title']}")
    
    # Test 3: Date range fetch
    print("\n" + "="*80)
    print("TEST 3: Fetch news for date range")
    print("="*80)
    
    range_results = fetcher.fetch_news_range(
        start_date='2024-11-01',
        end_date='2024-11-05',
        query='gold prices inflation',
        max_records_per_day=3
    )
    
    print(f"\nDates with news: {len(range_results)}")
    for date, articles in range_results.items():
        print(f"  {date}: {len(articles)} articles")
    
    # Test 4: Trending topics
    print("\n" + "="*80)
    print("TEST 4: Get trending topics for a date")
    print("="*80)
    
    trending = fetcher.get_trending_topics('2024-11-01', num_queries=3)
    
    for topic_data in trending:
        print(f"\nTopic: {topic_data['topic']}")
        print(f"Articles: {topic_data['article_count']}")
        for article in topic_data['articles'][:2]:
            print(f"  • {article['title']}")
    
    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    # Save single date results
    fetcher.save_to_json(
        {'date': test_date, 'articles': articles},
        'news_single_date.json'
    )
    
    # Save event results
    fetcher.save_to_json(event_results, 'news_events.json')
    
    # Save to CSV
    all_articles = []
    for event in event_results['events']:
        all_articles.extend(event['articles'])
    
    if all_articles:
        fetcher.save_to_csv(all_articles, 'news_articles.csv')
    
    # Generate summary
    print("\n" + "="*80)
    print("NEWS SUMMARY")
    print("="*80)
    
    summary = fetcher.get_news_summary(all_articles)
    print(f"\nTotal Articles: {summary['total_articles']}")
    print(f"Unique Sources: {summary['unique_sources']}")
    print(f"\nTop Sources:")
    for source, count in list(summary.get('top_sources', {}).items())[:5]:
        print(f"  {source}: {count}")
    
    print("\n" + "="*80)
    print("✓ News fetching complete")
    print("="*80)
