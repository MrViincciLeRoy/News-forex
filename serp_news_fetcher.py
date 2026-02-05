import requests
from bs4 import BeautifulSoup
import json
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import re
from urllib.parse import urlparse, quote
from api_key_manager import APIKeyManager


class SerpNewsFetcher:
    def __init__(self):
        self.key_manager = APIKeyManager()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
        
        self.search_strategies = [
            'standard',
            'exact_phrase',
            'special_chars',
            'ai_search',
            'date_specific'
        ]
    
    def fetch_comprehensive_news(
        self,
        query: str,
        date: str,
        max_articles: int = 20,
        full_content: bool = True
    ) -> Dict:
        print(f"\n{'='*80}")
        print(f"COMPREHENSIVE NEWS FETCH")
        print(f"{'='*80}")
        print(f"Query: {query}")
        print(f"Date: {date}")
        print(f"Max Articles: {max_articles}")
        print(f"Full Content: {full_content}")
        
        date_obj = datetime.strptime(date, '%Y-%m-%d')
        
        results = {
            'query': query,
            'date': date,
            'timestamp': datetime.now().isoformat(),
            'articles': [],
            'sources': {},
            'statistics': {
                'total_found': 0,
                'full_content_extracted': 0,
                'strategies_used': [],
                'sources_count': 0
            }
        }
        
        all_articles = []
        strategy_attempts = []
        
        for strategy in self.search_strategies:
            if len(all_articles) >= max_articles:
                break
            
            print(f"\n🔍 Strategy: {strategy}")
            print("-"*80)
            
            try:
                articles = self._execute_strategy(
                    strategy,
                    query,
                    date_obj,
                    max_articles - len(all_articles)
                )
                
                if articles:
                    all_articles.extend(articles)
                    strategy_attempts.append(strategy)
                    print(f"  ✓ Found {len(articles)} articles")
                else:
                    print(f"  ⊘ No results")
            
            except Exception as e:
                print(f"  ✗ Error: {str(e)[:60]}")
        
        all_articles = self._deduplicate_articles(all_articles)
        print(f"\n📊 After deduplication: {len(all_articles)} unique articles")
        
        if full_content:
            print(f"\n📥 Extracting full content...")
            print("-"*80)
            
            for i, article in enumerate(all_articles[:max_articles], 1):
                try:
                    print(f"  [{i}/{min(len(all_articles), max_articles)}] {article['title'][:50]}...")
                    
                    full_text = self._extract_full_article(article['url'])
                    
                    if full_text:
                        article['full_content'] = full_text
                        article['content_length'] = len(full_text)
                        results['statistics']['full_content_extracted'] += 1
                        print(f"      ✓ {len(full_text)} chars")
                    else:
                        article['full_content'] = article.get('snippet', '')
                        article['content_length'] = len(article.get('snippet', ''))
                        print(f"      ⊘ Using snippet")
                    
                    time.sleep(0.5)
                
                except Exception as e:
                    print(f"      ✗ {str(e)[:40]}")
                    article['full_content'] = article.get('snippet', '')
                    article['content_length'] = len(article.get('snippet', ''))
        
        results['articles'] = all_articles[:max_articles]
        results['statistics']['total_found'] = len(results['articles'])
        results['statistics']['strategies_used'] = strategy_attempts
        
        sources = {}
        for article in results['articles']:
            source = article.get('source', 'Unknown')
            sources[source] = sources.get(source, 0) + 1
        
        results['sources'] = sources
        results['statistics']['sources_count'] = len(sources)
        
        print(f"\n{'='*80}")
        print(f"✓ FETCH COMPLETE")
        print(f"{'='*80}")
        print(f"Total Articles: {results['statistics']['total_found']}")
        print(f"Full Content: {results['statistics']['full_content_extracted']}")
        print(f"Sources: {results['statistics']['sources_count']}")
        print(f"Strategies: {', '.join(strategy_attempts)}")
        print(f"{'='*80}\n")
        
        return results
    
    def _execute_strategy(
        self,
        strategy: str,
        query: str,
        date_obj: datetime,
        max_results: int
    ) -> List[Dict]:
        if strategy == 'standard':
            return self._standard_search(query, date_obj, max_results)
        
        elif strategy == 'exact_phrase':
            return self._exact_phrase_search(query, date_obj, max_results)
        
        elif strategy == 'special_chars':
            return self._special_chars_search(query, date_obj, max_results)
        
        elif strategy == 'ai_search':
            return self._ai_search(query, date_obj, max_results)
        
        elif strategy == 'date_specific':
            return self._date_specific_search(query, date_obj, max_results)
        
        return []
    
    def _standard_search(
        self,
        query: str,
        date_obj: datetime,
        max_results: int
    ) -> List[Dict]:
        api_key = self.key_manager.get_best_key('serp')
        if not api_key:
            api_key = self.key_manager.get_key('serp')
        
        if not api_key:
            print("  ⊘ No SERP API key available")
            return []
        
        params = {
            'engine': 'google',
            'q': query,
            'api_key': api_key[0] if isinstance(api_key, tuple) else api_key,
            'num': max_results,
            'tbm': 'nws',
            'tbs': f'cdr:1,cd_min:{(date_obj - timedelta(days=2)).strftime("%m/%d/%Y")},cd_max:{(date_obj + timedelta(days=2)).strftime("%m/%d/%Y")}'
        }
        
        try:
            response = self.session.get(
                'https://serpapi.com/search',
                params=params,
                timeout=15
            )
            
            self.key_manager.record_usage(
                'serp',
                api_key[0] if isinstance(api_key, tuple) else api_key,
                response.status_code == 200
            )
            
            response.raise_for_status()
            data = response.json()
            
            articles = []
            for item in data.get('news_results', []):
                article = self._parse_serp_result(item, 'standard')
                if article:
                    articles.append(article)
            
            return articles
        
        except Exception as e:
            print(f"  Error: {str(e)[:60]}")
            return []
    
    def _exact_phrase_search(
        self,
        query: str,
        date_obj: datetime,
        max_results: int
    ) -> List[Dict]:
        exact_query = f'"{query}"'
        
        api_key = self.key_manager.get_best_key('serp')
        if not api_key:
            api_key = self.key_manager.get_key('serp')
        
        if not api_key:
            return []
        
        params = {
            'engine': 'google',
            'q': exact_query,
            'api_key': api_key[0] if isinstance(api_key, tuple) else api_key,
            'num': max_results,
            'tbm': 'nws'
        }
        
        try:
            response = self.session.get(
                'https://serpapi.com/search',
                params=params,
                timeout=15
            )
            
            self.key_manager.record_usage(
                'serp',
                api_key[0] if isinstance(api_key, tuple) else api_key,
                response.status_code == 200
            )
            
            response.raise_for_status()
            data = response.json()
            
            articles = []
            for item in data.get('news_results', []):
                article = self._parse_serp_result(item, 'exact_phrase')
                if article:
                    articles.append(article)
            
            return articles
        
        except Exception as e:
            print(f"  Error: {str(e)[:60]}")
            return []
    
    def _special_chars_search(
        self,
        query: str,
        date_obj: datetime,
        max_results: int
    ) -> List[Dict]:
        keywords = query.split()
        
        special_queries = [
            f'{keywords[0]} OR {keywords[1] if len(keywords) > 1 else keywords[0]}',
            f'{query} (market OR economy OR finance)',
            f'"{keywords[0]}" AND {keywords[1] if len(keywords) > 1 else "news"}'
        ]
        
        all_articles = []
        
        for special_query in special_queries:
            if len(all_articles) >= max_results:
                break
            
            api_key = self.key_manager.get_best_key('serp')
            if not api_key:
                api_key = self.key_manager.get_key('serp')
            
            if not api_key:
                continue
            
            params = {
                'engine': 'google',
                'q': special_query,
                'api_key': api_key[0] if isinstance(api_key, tuple) else api_key,
                'num': max_results // len(special_queries),
                'tbm': 'nws'
            }
            
            try:
                response = self.session.get(
                    'https://serpapi.com/search',
                    params=params,
                    timeout=15
                )
                
                self.key_manager.record_usage(
                    'serp',
                    api_key[0] if isinstance(api_key, tuple) else api_key,
                    response.status_code == 200
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    for item in data.get('news_results', []):
                        article = self._parse_serp_result(item, 'special_chars')
                        if article:
                            all_articles.append(article)
                
                time.sleep(1)
            
            except Exception as e:
                print(f"  Error with query '{special_query[:30]}...': {str(e)[:40]}")
        
        return all_articles
    
    def _ai_search(
        self,
        query: str,
        date_obj: datetime,
        max_results: int
    ) -> List[Dict]:
        enhanced_query = f"{query} analysis impact market news {date_obj.strftime('%B %Y')}"
        
        api_key = self.key_manager.get_best_key('serp')
        if not api_key:
            api_key = self.key_manager.get_key('serp')
        
        if not api_key:
            return []
        
        params = {
            'engine': 'google',
            'q': enhanced_query,
            'api_key': api_key[0] if isinstance(api_key, tuple) else api_key,
            'num': max_results,
            'tbm': 'nws',
            'gl': 'us',
            'hl': 'en'
        }
        
        try:
            response = self.session.get(
                'https://serpapi.com/search',
                params=params,
                timeout=15
            )
            
            self.key_manager.record_usage(
                'serp',
                api_key[0] if isinstance(api_key, tuple) else api_key,
                response.status_code == 200
            )
            
            response.raise_for_status()
            data = response.json()
            
            articles = []
            for item in data.get('news_results', []):
                article = self._parse_serp_result(item, 'ai_search')
                if article:
                    articles.append(article)
            
            return articles
        
        except Exception as e:
            print(f"  Error: {str(e)[:60]}")
            return []
    
    def _date_specific_search(
        self,
        query: str,
        date_obj: datetime,
        max_results: int
    ) -> List[Dict]:
        date_query = f'{query} {date_obj.strftime("%B %d %Y")}'
        
        api_key = self.key_manager.get_best_key('serp')
        if not api_key:
            api_key = self.key_manager.get_key('serp')
        
        if not api_key:
            return []
        
        params = {
            'engine': 'google',
            'q': date_query,
            'api_key': api_key[0] if isinstance(api_key, tuple) else api_key,
            'num': max_results,
            'tbm': 'nws'
        }
        
        try:
            response = self.session.get(
                'https://serpapi.com/search',
                params=params,
                timeout=15
            )
            
            self.key_manager.record_usage(
                'serp',
                api_key[0] if isinstance(api_key, tuple) else api_key,
                response.status_code == 200
            )
            
            response.raise_for_status()
            data = response.json()
            
            articles = []
            for item in data.get('news_results', []):
                article = self._parse_serp_result(item, 'date_specific')
                if article:
                    articles.append(article)
            
            return articles
        
        except Exception as e:
            print(f"  Error: {str(e)[:60]}")
            return []
    
    def _parse_serp_result(self, item: Dict, strategy: str) -> Optional[Dict]:
        try:
            article = {
                'title': item.get('title', ''),
                'url': item.get('link', ''),
                'snippet': item.get('snippet', ''),
                'source': item.get('source', 'Unknown'),
                'date': item.get('date', ''),
                'thumbnail': item.get('thumbnail', ''),
                'strategy': strategy,
                'timestamp': datetime.now().isoformat()
            }
            
            if not article['title'] or not article['url']:
                return None
            
            return article
        
        except Exception as e:
            return None
    
    def _extract_full_article(self, url: str) -> Optional[str]:
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'form']):
                tag.decompose()
            
            article_selectors = [
                'article',
                '[role="article"]',
                '.article-body',
                '.article-content',
                '.post-content',
                '.entry-content',
                '#article-body',
                '.story-body'
            ]
            
            article_content = None
            for selector in article_selectors:
                article_content = soup.select_one(selector)
                if article_content:
                    break
            
            if not article_content:
                article_content = soup.find('body')
            
            if not article_content:
                return None
            
            paragraphs = article_content.find_all('p')
            
            text_parts = []
            for p in paragraphs:
                text = p.get_text(strip=True)
                if len(text) > 50:
                    text_parts.append(text)
            
            full_text = '\n\n'.join(text_parts)
            
            full_text = re.sub(r'\n{3,}', '\n\n', full_text)
            full_text = re.sub(r' {2,}', ' ', full_text)
            
            if len(full_text) < 200:
                return None
            
            return full_text
        
        except Exception as e:
            return None
    
    def _deduplicate_articles(self, articles: List[Dict]) -> List[Dict]:
        seen_urls = set()
        seen_titles = set()
        unique = []
        
        for article in articles:
            url = article.get('url', '')
            title = article.get('title', '').lower().strip()
            
            url_key = urlparse(url).path
            title_key = re.sub(r'\W+', '', title)[:100]
            
            if url_key not in seen_urls and title_key not in seen_titles:
                seen_urls.add(url_key)
                seen_titles.add(title_key)
                unique.append(article)
        
        return unique
    
    def save_results(self, results: Dict, filename: Optional[str] = None) -> str:
        if not filename:
            query_clean = re.sub(r'\W+', '_', results['query'])[:30]
            date_clean = results['date'].replace('-', '_')
            filename = f"news_{query_clean}_{date_clean}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Saved: {filename}")
        
        csv_file = filename.replace('.json', '.csv')
        self._save_csv(results, csv_file)
        
        return filename
    
    def _save_csv(self, results: Dict, filename: str):
        import csv
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['title', 'url', 'source', 'date', 'snippet', 'content_length', 'strategy']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            writer.writeheader()
            for article in results['articles']:
                writer.writerow({
                    'title': article.get('title', ''),
                    'url': article.get('url', ''),
                    'source': article.get('source', ''),
                    'date': article.get('date', ''),
                    'snippet': article.get('snippet', '')[:200],
                    'content_length': article.get('content_length', 0),
                    'strategy': article.get('strategy', '')
                })
        
        print(f"💾 Saved: {filename}")


if __name__ == "__main__":
    print("="*80)
    print("SERP NEWS FETCHER TEST")
    print("="*80)
    
    fetcher = SerpNewsFetcher()
    
    results = fetcher.fetch_comprehensive_news(
        query='Non-Farm Payrolls',
        date='2024-11-01',
        max_articles=10,
        full_content=True
    )
    
    output_file = fetcher.save_results(results)
    
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"Total Articles: {results['statistics']['total_found']}")
    print(f"Full Content: {results['statistics']['full_content_extracted']}")
    print(f"Sources: {results['statistics']['sources_count']}")
    print(f"\nTop Sources:")
    for source, count in sorted(results['sources'].items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {source}: {count}")
    
    print("\n" + "="*80)
    print("✓ TEST COMPLETE")
    print("="*80)
