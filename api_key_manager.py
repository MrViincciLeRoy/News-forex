import os
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class APIKeyManager:
    def __init__(self, state_file='api_key_state.json'):
        self.state_file = state_file
        self.keys = {
            'serp': [],
            'alpha_vantage': [],
            'fred': [],
            'newsapi': [],
            'bing': []
        }
        self.usage = {}
        self.limits = {
            'serp': {'calls_per_month': 100, 'calls_per_hour': 50},
            'alpha_vantage': {'calls_per_day': 500, 'calls_per_minute': 5},
            'fred': {'calls_per_day': 1000},
            'newsapi': {'calls_per_day': 100},
            'bing': {'calls_per_month': 1000}
        }
        
        self._load_keys()
        self._load_state()
    
    def _load_keys(self):
        for i in range(1, 11):
            for service in ['serp', 'alpha_vantage', 'fred', 'newsapi', 'bing']:
                key = os.environ.get(f'{service.upper()}_API_KEY_{i}', '')
                if not key:
                    key = os.environ.get(f'{service.upper()}_API_KEY', '')
                
                if key and key not in self.keys[service]:
                    self.keys[service].append(key)
        
        for service, keys in self.keys.items():
            print(f"  {service.upper()}: {len(keys)} key(s)")
    
    def _load_state(self):
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    self.usage = json.load(f)
                self._clean_old_usage()
            except:
                self.usage = {}
        else:
            self.usage = {}
    
    def _save_state(self):
        with open(self.state_file, 'w') as f:
            json.dump(self.usage, f, indent=2)
    
    def _clean_old_usage(self):
        now = datetime.now()
        cutoff = (now - timedelta(days=32)).isoformat()
        
        for service in list(self.usage.keys()):
            for key_hash in list(self.usage[service].keys()):
                usage_data = self.usage[service][key_hash]
                usage_data['hourly'] = {
                    ts: count for ts, count in usage_data.get('hourly', {}).items()
                    if ts > (now - timedelta(hours=2)).isoformat()
                }
                usage_data['daily'] = {
                    ts: count for ts, count in usage_data.get('daily', {}).items()
                    if ts > (now - timedelta(days=2)).isoformat()
                }
                usage_data['monthly'] = {
                    ts: count for ts, count in usage_data.get('monthly', {}).items()
                    if ts > cutoff
                }
    
    def _get_key_hash(self, key: str) -> str:
        return f"{key[:8]}...{key[-4:]}"
    
    def _init_usage_tracking(self, service: str, key_hash: str):
        if service not in self.usage:
            self.usage[service] = {}
        
        if key_hash not in self.usage[service]:
            self.usage[service][key_hash] = {
                'hourly': {},
                'daily': {},
                'monthly': {},
                'last_used': None,
                'total_calls': 0,
                'errors': 0
            }
    
    def _can_use_key(self, service: str, key_hash: str) -> Tuple[bool, str]:
        if service not in self.usage or key_hash not in self.usage[service]:
            return True, "OK"
        
        usage = self.usage[service][key_hash]
        now = datetime.now()
        
        limits = self.limits.get(service, {})
        
        if 'calls_per_hour' in limits:
            hour_key = now.strftime('%Y-%m-%d-%H')
            hourly_calls = usage['hourly'].get(hour_key, 0)
            if hourly_calls >= limits['calls_per_hour']:
                return False, f"Hourly limit ({limits['calls_per_hour']})"
        
        if 'calls_per_day' in limits:
            day_key = now.strftime('%Y-%m-%d')
            daily_calls = usage['daily'].get(day_key, 0)
            if daily_calls >= limits['calls_per_day']:
                return False, f"Daily limit ({limits['calls_per_day']})"
        
        if 'calls_per_month' in limits:
            month_key = now.strftime('%Y-%m')
            monthly_calls = usage['monthly'].get(month_key, 0)
            if monthly_calls >= limits['calls_per_month']:
                return False, f"Monthly limit ({limits['calls_per_month']})"
        
        if 'calls_per_minute' in limits:
            if usage.get('last_used'):
                last_used = datetime.fromisoformat(usage['last_used'])
                if (now - last_used).total_seconds() < 60 / limits['calls_per_minute']:
                    return False, "Rate limit"
        
        return True, "OK"
    
    def get_key(self, service: str) -> Optional[str]:
        if service not in self.keys or not self.keys[service]:
            return None
        
        for key in self.keys[service]:
            key_hash = self._get_key_hash(key)
            self._init_usage_tracking(service, key_hash)
            
            can_use, reason = self._can_use_key(service, key_hash)
            if can_use:
                return key
        
        return None
    
    def record_usage(self, service: str, key: str, success: bool = True):
        key_hash = self._get_key_hash(key)
        self._init_usage_tracking(service, key_hash)
        
        now = datetime.now()
        usage = self.usage[service][key_hash]
        
        hour_key = now.strftime('%Y-%m-%d-%H')
        day_key = now.strftime('%Y-%m-%d')
        month_key = now.strftime('%Y-%m')
        
        usage['hourly'][hour_key] = usage['hourly'].get(hour_key, 0) + 1
        usage['daily'][day_key] = usage['daily'].get(day_key, 0) + 1
        usage['monthly'][month_key] = usage['monthly'].get(month_key, 0) + 1
        
        usage['last_used'] = now.isoformat()
        usage['total_calls'] += 1
        
        if not success:
            usage['errors'] += 1
        
        self._save_state()
    
    def get_usage_stats(self, service: str) -> Dict:
        if service not in self.usage:
            return {}
        
        stats = {
            'total_keys': len(self.keys.get(service, [])),
            'keys': []
        }
        
        for key_hash, usage in self.usage[service].items():
            now = datetime.now()
            
            key_stats = {
                'key': key_hash,
                'total_calls': usage['total_calls'],
                'errors': usage['errors'],
                'last_used': usage['last_used'],
                'current_usage': {}
            }
            
            hour_key = now.strftime('%Y-%m-%d-%H')
            day_key = now.strftime('%Y-%m-%d')
            month_key = now.strftime('%Y-%m')
            
            key_stats['current_usage']['hourly'] = usage['hourly'].get(hour_key, 0)
            key_stats['current_usage']['daily'] = usage['daily'].get(day_key, 0)
            key_stats['current_usage']['monthly'] = usage['monthly'].get(month_key, 0)
            
            can_use, reason = self._can_use_key(service, key_hash)
            key_stats['available'] = can_use
            key_stats['status'] = reason
            
            stats['keys'].append(key_stats)
        
        return stats
    
    def reset_key(self, service: str, key: str):
        key_hash = self._get_key_hash(key)
        if service in self.usage and key_hash in self.usage[service]:
            del self.usage[service][key_hash]
            self._save_state()
    
    def get_best_key(self, service: str) -> Optional[Tuple[str, Dict]]:
        if service not in self.keys or not self.keys[service]:
            return None
        
        best_key = None
        best_score = -1
        best_stats = None
        
        for key in self.keys[service]:
            key_hash = self._get_key_hash(key)
            self._init_usage_tracking(service, key_hash)
            
            can_use, reason = self._can_use_key(service, key_hash)
            if not can_use:
                continue
            
            usage = self.usage[service][key_hash]
            
            error_rate = usage['errors'] / max(usage['total_calls'], 1)
            
            score = 100 - (error_rate * 50)
            
            now = datetime.now()
            month_key = now.strftime('%Y-%m')
            monthly_usage = usage['monthly'].get(month_key, 0)
            monthly_limit = self.limits.get(service, {}).get('calls_per_month', 1000)
            
            score -= (monthly_usage / monthly_limit) * 30
            
            if usage.get('last_used'):
                last_used = datetime.fromisoformat(usage['last_used'])
                minutes_ago = (now - last_used).total_seconds() / 60
                score += min(minutes_ago, 20)
            else:
                score += 20
            
            if score > best_score:
                best_score = score
                best_key = key
                best_stats = {
                    'score': score,
                    'error_rate': error_rate,
                    'monthly_usage': monthly_usage,
                    'total_calls': usage['total_calls']
                }
        
        return (best_key, best_stats) if best_key else None


if __name__ == "__main__":
    print("="*80)
    print("API KEY MANAGER TEST")
    print("="*80)
    
    manager = APIKeyManager()
    
    print("\nKey Summary:")
    print("-"*80)
    for service in ['serp', 'alpha_vantage', 'fred']:
        stats = manager.get_usage_stats(service)
        print(f"\n{service.upper()}:")
        print(f"  Total Keys: {stats.get('total_keys', 0)}")
        
        for key_stat in stats.get('keys', []):
            print(f"  {key_stat['key']}: {key_stat['total_calls']} calls, {key_stat['status']}")
    
    print("\n" + "="*80)
    print("Test: Get best SERP key")
    print("-"*80)
    
    result = manager.get_best_key('serp')
    if result:
        key, stats = result
        print(f"Best key: {manager._get_key_hash(key)}")
        print(f"Score: {stats['score']:.2f}")
        print(f"Total calls: {stats['total_calls']}")
    else:
        print("No SERP keys available")
    
    print("\n" + "="*80)
