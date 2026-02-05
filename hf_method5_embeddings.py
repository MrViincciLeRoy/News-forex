"""
HF Analytics Method 5: Correlation Discovery with Embeddings
Find semantic relationships between markets using sentence embeddings
Models: sentence-transformers/all-MiniLM-L6-v2, BAAI/bge-large-en-v1.5
Enhances: correlation_analyzer.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import json


class HFCorrelationDiscovery:
    """
    Discover hidden correlations using semantic embeddings
    Goes beyond numerical correlation to find conceptual relationships
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.embeddings_cache = {}
        
        print(f"Initializing HF Correlation Discovery: {model_name}")
    
    def load_model(self):
        """Load sentence transformer model"""
        try:
            from sentence_transformers import SentenceTransformer
            
            print(f"Loading model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            
            print("✓ Model loaded successfully")
            return True
            
        except ImportError:
            print("⚠️  sentence-transformers not installed")
            print("   Install: pip install sentence-transformers")
            return False
        except Exception as e:
            print(f"✗ Error: {e}")
            return False
    
    def create_event_description(self, event: Dict) -> str:
        """Create rich text description of event for embedding"""
        parts = [event.get('event', '')]
        
        if event.get('description'):
            parts.append(event['description'])
        
        if event.get('category'):
            parts.append(f"Category: {event['category']}")
        
        if event.get('impact'):
            parts.append(f"Impact: {event['impact']}")
        
        return '. '.join(parts)
    
    def embed_events(self, events: List[Dict]) -> Dict[str, np.ndarray]:
        """Create embeddings for events"""
        if not self.model:
            print("Model not loaded, cannot create embeddings")
            return {}
        
        embeddings = {}
        
        print(f"Creating embeddings for {len(events)} events...")
        
        for event in events:
            event_id = event.get('event', '') + '_' + event.get('date', '')
            
            if event_id in self.embeddings_cache:
                embeddings[event_id] = self.embeddings_cache[event_id]
                continue
            
            description = self.create_event_description(event)
            embedding = self.model.encode(description)
            
            embeddings[event_id] = embedding
            self.embeddings_cache[event_id] = embedding
        
        print(f"✓ Created {len(embeddings)} embeddings")
        return embeddings
    
    def find_similar_events(self, target_event: Dict, 
                           candidate_events: List[Dict],
                           top_k: int = 5) -> List[Tuple[Dict, float]]:
        """Find events semantically similar to target"""
        if not self.model:
            return []
        
        all_events = [target_event] + candidate_events
        embeddings = self.embed_events(all_events)
        
        if not embeddings:
            return []
        
        target_id = target_event.get('event', '') + '_' + target_event.get('date', '')
        target_embedding = embeddings[target_id]
        
        similarities = []
        
        for event in candidate_events:
            event_id = event.get('event', '') + '_' + event.get('date', '')
            
            if event_id == target_id:
                continue
            
            event_embedding = embeddings.get(event_id)
            if event_embedding is None:
                continue
            
            similarity = self._cosine_similarity(target_embedding, event_embedding)
            similarities.append((event, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between vectors"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    
    def cluster_events(self, events: List[Dict], 
                      n_clusters: int = 5) -> Dict[int, List[Dict]]:
        """Cluster events by semantic similarity"""
        if not self.model:
            return {}
        
        embeddings = self.embed_events(events)
        
        if not embeddings:
            return {}
        
        try:
            from sklearn.cluster import KMeans
            
            embedding_matrix = np.array(list(embeddings.values()))
            event_ids = list(embeddings.keys())
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(embedding_matrix)
            
            clusters = {}
            for event, label in zip(events, labels):
                label = int(label)
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(event)
            
            return clusters
            
        except ImportError:
            print("⚠️  sklearn not installed for clustering")
            return {}
        except Exception as e:
            print(f"Clustering error: {e}")
            return {}
    
    def find_cross_market_relationships(self, 
                                       events_by_market: Dict[str, List[Dict]],
                                       threshold: float = 0.5) -> List[Dict]:
        """Find relationships between events in different markets"""
        if not self.model:
            return []
        
        relationships = []
        markets = list(events_by_market.keys())
        
        print(f"Analyzing cross-market relationships across {len(markets)} markets...")
        
        for i, market1 in enumerate(markets):
            for market2 in markets[i+1:]:
                events1 = events_by_market[market1]
                events2 = events_by_market[market2]
                
                embeddings1 = self.embed_events(events1)
                embeddings2 = self.embed_events(events2)
                
                for event1 in events1:
                    event1_id = event1.get('event', '') + '_' + event1.get('date', '')
                    emb1 = embeddings1.get(event1_id)
                    
                    if emb1 is None:
                        continue
                    
                    for event2 in events2:
                        event2_id = event2.get('event', '') + '_' + event2.get('date', '')
                        emb2 = embeddings2.get(event2_id)
                        
                        if emb2 is None:
                            continue
                        
                        similarity = self._cosine_similarity(emb1, emb2)
                        
                        if similarity > threshold:
                            relationships.append({
                                'market1': market1,
                                'event1': event1.get('event', ''),
                                'date1': event1.get('date', ''),
                                'market2': market2,
                                'event2': event2.get('event', ''),
                                'date2': event2.get('date', ''),
                                'similarity': round(similarity, 4),
                                'relationship_strength': 'STRONG' if similarity > 0.7 else 'MODERATE'
                            })
        
        relationships.sort(key=lambda x: x['similarity'], reverse=True)
        
        print(f"✓ Found {len(relationships)} cross-market relationships")
        return relationships
    
    def discover_hidden_correlations(self, events: List[Dict],
                                    price_data: Dict[str, pd.Series] = None) -> Dict:
        """Discover semantic + numerical correlations"""
        semantic_clusters = self.cluster_events(events)
        
        result = {
            'total_events': len(events),
            'semantic_clusters': len(semantic_clusters),
            'clusters': {}
        }
        
        for cluster_id, cluster_events in semantic_clusters.items():
            cluster_desc = self._describe_cluster(cluster_events)
            
            result['clusters'][cluster_id] = {
                'description': cluster_desc,
                'event_count': len(cluster_events),
                'events': [e.get('event', '') for e in cluster_events[:5]],
                'date_range': {
                    'start': min([e.get('date', '') for e in cluster_events]),
                    'end': max([e.get('date', '') for e in cluster_events])
                }
            }
            
            if price_data:
                result['clusters'][cluster_id]['price_correlation'] = \
                    self._analyze_cluster_price_impact(cluster_events, price_data)
        
        return result
    
    def _describe_cluster(self, events: List[Dict]) -> str:
        """Generate description for event cluster"""
        event_names = [e.get('event', '').lower() for e in events]
        
        if any('cpi' in name or 'inflation' in name for name in event_names):
            return "Inflation & Price Data Events"
        elif any('payroll' in name or 'employment' in name or 'job' in name for name in event_names):
            return "Employment & Labor Market Events"
        elif any('fed' in name or 'fomc' in name or 'rate' in name for name in event_names):
            return "Monetary Policy Events"
        elif any('gdp' in name or 'growth' in name for name in event_names):
            return "Economic Growth Events"
        else:
            return "Mixed Economic Events"
    
    def _analyze_cluster_price_impact(self, events: List[Dict],
                                     price_data: Dict[str, pd.Series]) -> Dict:
        """Analyze price movements around clustered events"""
        impacts = {}
        
        for symbol, prices in price_data.items():
            total_change = 0
            count = 0
            
            for event in events:
                event_date = event.get('date')
                if not event_date:
                    continue
                
                try:
                    date = pd.to_datetime(event_date)
                    
                    if date in prices.index:
                        before = prices.loc[date - timedelta(days=1)]
                        after = prices.loc[date + timedelta(days=1)]
                        change = ((after - before) / before) * 100
                        
                        total_change += change
                        count += 1
                except:
                    continue
            
            if count > 0:
                impacts[symbol] = round(total_change / count, 2)
        
        return impacts
    
    def save_results(self, results: Dict, filepath: str):
        """Save correlation discovery results"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved: {filepath}")


if __name__ == "__main__":
    print("="*80)
    print("HF METHOD 5: CORRELATION DISCOVERY WITH EMBEDDINGS")
    print("="*80)
    
    discovery = HFCorrelationDiscovery()
    model_loaded = discovery.load_model()
    
    test_events = [
        {
            'event': 'Non-Farm Payrolls',
            'description': 'Monthly employment report',
            'date': '2024-11-01',
            'category': 'employment'
        },
        {
            'event': 'Consumer Price Index',
            'description': 'Inflation measurement',
            'date': '2024-10-10',
            'category': 'inflation'
        },
        {
            'event': 'Unemployment Rate',
            'description': 'Labor market indicator',
            'date': '2024-11-01',
            'category': 'employment'
        },
        {
            'event': 'Producer Price Index',
            'description': 'Wholesale inflation',
            'date': '2024-10-12',
            'category': 'inflation'
        },
        {
            'event': 'FOMC Rate Decision',
            'description': 'Federal Reserve policy',
            'date': '2024-09-18',
            'category': 'monetary_policy'
        },
        {
            'event': 'Retail Sales',
            'description': 'Consumer spending',
            'date': '2024-11-15',
            'category': 'growth'
        }
    ]
    
    if model_loaded:
        print("\n" + "="*80)
        print("TEST 1: FIND SIMILAR EVENTS")
        print("="*80)
        
        target = test_events[0]  # NFP
        similar = discovery.find_similar_events(target, test_events[1:], top_k=3)
        
        print(f"\nEvents similar to '{target['event']}':")
        for event, score in similar:
            print(f"  {event['event']}: {score:.4f} similarity")
        
        print("\n" + "="*80)
        print("TEST 2: CLUSTER EVENTS")
        print("="*80)
        
        clusters = discovery.cluster_events(test_events, n_clusters=3)
        
        for cluster_id, events in clusters.items():
            print(f"\nCluster {cluster_id}:")
            for event in events:
                print(f"  - {event['event']}")
        
        print("\n" + "="*80)
        print("TEST 3: CROSS-MARKET RELATIONSHIPS")
        print("="*80)
        
        events_by_market = {
            'US_Employment': test_events[:3],
            'US_Inflation': test_events[1:4],
            'US_Monetary': test_events[4:]
        }
        
        relationships = discovery.find_cross_market_relationships(
            events_by_market, 
            threshold=0.4
        )
        
        print(f"\nFound {len(relationships)} cross-market relationships:")
        for rel in relationships[:5]:
            print(f"\n{rel['market1']}: {rel['event1']}")
            print(f"  ↔ {rel['market2']}: {rel['event2']}")
            print(f"  Similarity: {rel['similarity']:.4f} ({rel['relationship_strength']})")
        
        print("\n" + "="*80)
        print("TEST 4: HIDDEN CORRELATIONS")
        print("="*80)
        
        correlations = discovery.discover_hidden_correlations(test_events)
        
        print(f"\nDiscovered {correlations['semantic_clusters']} semantic clusters:")
        for cluster_id, data in correlations['clusters'].items():
            print(f"\nCluster {cluster_id}: {data['description']}")
            print(f"  Events: {data['event_count']}")
            print(f"  Sample: {', '.join(data['events'][:3])}")
        
        discovery.save_results(correlations, 'hf_correlation_discovery_results.json')
    
    print("\n" + "="*80)
    print("✓ Correlation discovery complete")
    print("="*80)
