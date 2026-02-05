"""
Smart Pipeline Optimizer with Memory Management
Prevents OOM errors, optimizes HF model loading, and adapts to system resources
"""

import os
import sys
import psutil
import gc
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
import json


class MemoryMonitor:
    """Real-time memory monitoring with alerts"""
    
    def __init__(self, threshold_percent=85, check_interval=5):
        self.threshold_percent = threshold_percent
        self.check_interval = check_interval
        self.monitoring = False
        self.alert_callback = None
        self.stats = []
        
    def get_memory_info(self) -> Dict:
        """Get current memory statistics"""
        vm = psutil.virtual_memory()
        process = psutil.Process()
        
        return {
            'total_gb': round(vm.total / (1024**3), 2),
            'available_gb': round(vm.available / (1024**3), 2),
            'used_gb': round(vm.used / (1024**3), 2),
            'percent_used': vm.percent,
            'process_mb': round(process.memory_info().rss / (1024**2), 2),
            'is_critical': vm.percent > self.threshold_percent
        }
    
    def start_monitoring(self, callback=None):
        """Start background memory monitoring"""
        self.monitoring = True
        self.alert_callback = callback
        
        def monitor_loop():
            while self.monitoring:
                info = self.get_memory_info()
                self.stats.append({
                    'timestamp': datetime.now().isoformat(),
                    **info
                })
                
                if info['is_critical'] and self.alert_callback:
                    self.alert_callback(info)
                
                time.sleep(self.check_interval)
        
        thread = threading.Thread(target=monitor_loop, daemon=True)
        thread.start()
    
    def stop_monitoring(self):
        """Stop background monitoring"""
        self.monitoring = False
    
    def get_peak_usage(self) -> float:
        """Get peak memory usage during monitoring"""
        if not self.stats:
            return 0.0
        return max(s['percent_used'] for s in self.stats)


class SmartModelLoader:
    """Intelligent HF model loading with memory awareness"""
    
    def __init__(self, memory_monitor: MemoryMonitor):
        self.monitor = memory_monitor
        self.loaded_models = {}
        self.load_order = []
        self.fallback_mode = False
        
        # Model priority (load critical ones first)
        self.model_priority = {
            'sentiment': 1,      # Most useful
            'ner': 2,           # Symbol extraction
            'classification': 3, # Event categorization
            'qa': 4,            # Question answering
            'multimodal': 5,    # Combined analysis
            'zeroshot': 6,      # Dynamic categorization
            'causal': 7,        # Explanations
            'embeddings': 8,    # Correlations
            'forecasting': 9,   # Predictions
            'anomaly': 10       # Statistical (no model)
        }
        
        # Memory requirements (estimated MB per model)
        self.model_memory_req = {
            'sentiment': 500,
            'ner': 400,
            'classification': 800,
            'qa': 600,
            'multimodal': 700,
            'zeroshot': 900,
            'causal': 1200,
            'embeddings': 300,
            'forecasting': 1500,
            'anomaly': 50
        }
    
    def can_load_model(self, model_name: str) -> bool:
        """Check if we have enough memory to load model"""
        info = self.monitor.get_memory_info()
        required_mb = self.model_memory_req.get(model_name, 500)
        available_mb = info['available_gb'] * 1024
        
        # Need at least 2x the model size + 1GB buffer
        safety_margin_mb = (required_mb * 2) + 1024
        
        can_load = available_mb > safety_margin_mb
        
        if not can_load:
            print(f"⚠️  Insufficient memory for {model_name}")
            print(f"   Required: ~{safety_margin_mb}MB, Available: {available_mb:.0f}MB")
        
        return can_load
    
    def load_model_safe(self, model_name: str, loader_func) -> Optional[Any]:
        """Safely load a model with memory checks"""
        
        # Check memory before loading
        if not self.can_load_model(model_name):
            print(f"⊘ Skipping {model_name} - insufficient memory")
            self.fallback_mode = True
            return None
        
        try:
            print(f"Loading {model_name}...")
            before_mem = self.monitor.get_memory_info()
            
            model = loader_func()
            
            after_mem = self.monitor.get_memory_info()
            actual_usage = after_mem['process_mb'] - before_mem['process_mb']
            
            print(f"✓ {model_name} loaded ({actual_usage:.0f}MB)")
            
            self.loaded_models[model_name] = {
                'model': model,
                'memory_mb': actual_usage,
                'timestamp': datetime.now().isoformat()
            }
            self.load_order.append(model_name)
            
            return model
            
        except MemoryError as e:
            print(f"✗ {model_name} failed: Out of memory")
            self.force_cleanup()
            self.fallback_mode = True
            return None
            
        except Exception as e:
            print(f"✗ {model_name} failed: {str(e)[:100]}")
            return None
    
    def force_cleanup(self):
        """Aggressive memory cleanup"""
        print("\n🧹 Emergency memory cleanup...")
        
        # Unload least important models first
        sorted_models = sorted(
            self.loaded_models.items(),
            key=lambda x: self.model_priority.get(x[0], 99),
            reverse=True
        )
        
        for name, data in sorted_models:
            if self.model_priority.get(name, 99) > 5:  # Keep top 5
                print(f"   Unloading {name}")
                del data['model']
                del self.loaded_models[name]
        
        gc.collect()
        
        info = self.monitor.get_memory_info()
        print(f"   Memory after cleanup: {info['percent_used']}%")
    
    def get_loaded_summary(self) -> Dict:
        """Get summary of loaded models"""
        total_mb = sum(m['memory_mb'] for m in self.loaded_models.values())
        
        return {
            'total_models': len(self.loaded_models),
            'total_memory_mb': round(total_mb, 2),
            'models': list(self.loaded_models.keys()),
            'fallback_mode': self.fallback_mode,
            'load_order': self.load_order
        }


class SmartPipelineOptimizer:
    """Main optimizer coordinating memory and model management"""
    
    def __init__(self, config_path='smart_config.json'):
        self.config = self._load_config(config_path)
        self.monitor = MemoryMonitor(
            threshold_percent=self.config['memory_threshold'],
            check_interval=self.config['check_interval']
        )
        self.model_loader = SmartModelLoader(self.monitor)
        self.session_stats = {
            'start_time': datetime.now().isoformat(),
            'warnings': [],
            'errors': []
        }
        
        # Detect environment
        self.is_ci = os.environ.get('CI') == 'true'
        self.is_github_actions = os.environ.get('GITHUB_ACTIONS') == 'true'
        
        print("="*80)
        print("SMART PIPELINE OPTIMIZER")
        print("="*80)
        print(f"Environment: {'CI/CD' if self.is_ci else 'Local'}")
        print(f"Memory Threshold: {self.config['memory_threshold']}%")
        
        # Start monitoring
        self.monitor.start_monitoring(callback=self._memory_alert)
        
        initial = self.monitor.get_memory_info()
        print(f"System Memory: {initial['total_gb']}GB total, "
              f"{initial['available_gb']}GB available ({initial['percent_used']}% used)")
        print("="*80 + "\n")
    
    def _load_config(self, path: str) -> Dict:
        """Load or create configuration"""
        default_config = {
            'memory_threshold': 85,
            'check_interval': 5,
            'max_models': 10,
            'enable_auto_cleanup': True,
            'ci_mode_models': ['sentiment', 'ner', 'classification'],
            'local_mode_models': 'all',
            'batch_sizes': {
                'small_memory': 5,
                'medium_memory': 10,
                'large_memory': 20
            }
        }
        
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    config = json.load(f)
                return {**default_config, **config}
            except:
                pass
        
        return default_config
    
    def _memory_alert(self, info: Dict):
        """Handle memory alerts"""
        msg = (f"⚠️  HIGH MEMORY: {info['percent_used']}% "
               f"({info['used_gb']}/{info['total_gb']}GB)")
        print(f"\n{msg}")
        
        self.session_stats['warnings'].append({
            'timestamp': datetime.now().isoformat(),
            'type': 'memory',
            'message': msg,
            'details': info
        })
        
        if self.config['enable_auto_cleanup']:
            self.model_loader.force_cleanup()
    
    def get_optimization_strategy(self) -> Dict:
        """Determine optimal strategy based on resources"""
        mem_info = self.monitor.get_memory_info()
        
        # Determine memory tier
        if mem_info['available_gb'] > 8:
            tier = 'large'
            max_models = 10
            batch_size = self.config['batch_sizes']['large_memory']
        elif mem_info['available_gb'] > 4:
            tier = 'medium'
            max_models = 6
            batch_size = self.config['batch_sizes']['medium_memory']
        else:
            tier = 'small'
            max_models = 3
            batch_size = self.config['batch_sizes']['small_memory']
        
        # CI mode restrictions
        if self.is_ci:
            max_models = min(max_models, 3)
            allowed_models = self.config['ci_mode_models']
        else:
            allowed_models = 'all'
        
        strategy = {
            'memory_tier': tier,
            'max_models': max_models,
            'batch_size': batch_size,
            'allowed_models': allowed_models,
            'use_fallbacks': mem_info['available_gb'] < 3,
            'enable_caching': mem_info['available_gb'] > 6
        }
        
        print("\n📊 Optimization Strategy:")
        print(f"   Tier: {tier.upper()}")
        print(f"   Max Models: {max_models}")
        print(f"   Batch Size: {batch_size}")
        print(f"   Fallback Mode: {strategy['use_fallbacks']}")
        print()
        
        return strategy
    
    def should_load_hf_model(self, model_name: str) -> bool:
        """Decide if a specific model should be loaded"""
        strategy = self.get_optimization_strategy()
        
        # Check if model is allowed
        if strategy['allowed_models'] != 'all':
            if model_name not in strategy['allowed_models']:
                return False
        
        # Check if we're at max models
        if len(self.model_loader.loaded_models) >= strategy['max_models']:
            return False
        
        # Check if we should use fallbacks
        if strategy['use_fallbacks']:
            # Only load highest priority models
            priority = self.model_loader.model_priority.get(model_name, 99)
            return priority <= 3
        
        return True
    
    def load_hf_method(self, method_name: str, module_name: str, 
                       class_name: str) -> Optional[Any]:
        """Load a HuggingFace method with optimization"""
        
        # Check if we should load this model
        if not self.should_load_hf_model(method_name):
            print(f"⊘ Skipping {method_name} (optimization)")
            return None
        
        def loader():
            # Dynamic import
            module = __import__(module_name, fromlist=[class_name])
            analyzer_class = getattr(module, class_name)
            
            # Create instance
            instance = analyzer_class()
            
            # Load model if it has the method
            if hasattr(instance, 'load_model'):
                try:
                    instance.load_model()
                except Exception as e:
                    print(f"   Model load warning: {str(e)[:80]}")
            
            return instance
        
        return self.model_loader.load_model_safe(method_name, loader)
    
    def optimize_batch_processing(self, items: List, process_func) -> List:
        """Process items in optimized batches"""
        strategy = self.get_optimization_strategy()
        batch_size = strategy['batch_size']
        
        results = []
        total = len(items)
        
        for i in range(0, total, batch_size):
            batch = items[i:i+batch_size]
            
            print(f"Processing batch {i//batch_size + 1}/{(total-1)//batch_size + 1} "
                  f"({len(batch)} items)")
            
            # Check memory before each batch
            mem_info = self.monitor.get_memory_info()
            if mem_info['is_critical']:
                print("⚠️  Memory critical, forcing cleanup...")
                self.model_loader.force_cleanup()
                gc.collect()
            
            # Process batch
            try:
                batch_results = process_func(batch)
                results.extend(batch_results)
            except MemoryError:
                print("✗ Batch failed: OOM - reducing batch size")
                # Try with smaller batches
                for item in batch:
                    try:
                        result = process_func([item])
                        results.extend(result)
                    except:
                        print(f"   Skipped 1 item due to memory")
        
        return results
    
    def get_session_report(self) -> Dict:
        """Generate session performance report"""
        self.monitor.stop_monitoring()
        
        mem_stats = self.monitor.get_memory_info()
        peak_mem = self.monitor.get_peak_usage()
        model_summary = self.model_loader.get_loaded_summary()
        
        report = {
            'session': {
                'start_time': self.session_stats['start_time'],
                'end_time': datetime.now().isoformat(),
                'environment': 'CI/CD' if self.is_ci else 'Local',
                'warnings': len(self.session_stats['warnings']),
                'errors': len(self.session_stats['errors'])
            },
            'memory': {
                'current_usage_percent': mem_stats['percent_used'],
                'peak_usage_percent': peak_mem,
                'current_available_gb': mem_stats['available_gb'],
                'process_memory_mb': mem_stats['process_mb']
            },
            'models': model_summary,
            'recommendations': []
        }
        
        # Generate recommendations
        if peak_mem > 90:
            report['recommendations'].append(
                "⚠️  Peak memory exceeded 90% - consider reducing batch sizes"
            )
        
        if model_summary['fallback_mode']:
            report['recommendations'].append(
                "ℹ️  Fallback mode was used - some models were skipped"
            )
        
        if model_summary['total_models'] < 3:
            report['recommendations'].append(
                "ℹ️  Limited models loaded - increase available memory for better results"
            )
        
        return report
    
    def print_report(self):
        """Print formatted session report"""
        report = self.get_session_report()
        
        print("\n" + "="*80)
        print("SMART OPTIMIZER SESSION REPORT")
        print("="*80)
        
        print(f"\nSession:")
        print(f"  Environment: {report['session']['environment']}")
        print(f"  Duration: {report['session']['start_time']} → {report['session']['end_time']}")
        print(f"  Warnings: {report['session']['warnings']}")
        
        print(f"\nMemory:")
        print(f"  Peak Usage: {report['memory']['peak_usage_percent']}%")
        print(f"  Current Usage: {report['memory']['current_usage_percent']}%")
        print(f"  Process Memory: {report['memory']['process_memory_mb']}MB")
        
        print(f"\nModels:")
        print(f"  Loaded: {report['models']['total_models']}")
        print(f"  Total Memory: {report['models']['total_memory_mb']}MB")
        print(f"  Models: {', '.join(report['models']['models'])}")
        print(f"  Fallback Mode: {report['models']['fallback_mode']}")
        
        if report['recommendations']:
            print(f"\nRecommendations:")
            for rec in report['recommendations']:
                print(f"  {rec}")
        
        print("\n" + "="*80)
    
    def save_report(self, filepath='optimizer_report.json'):
        """Save session report to file"""
        report = self.get_session_report()
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n✓ Report saved: {filepath}")


# Example integration with comprehensive pipeline
def create_optimized_pipeline(output_dir='optimized_output'):
    """Create comprehensive pipeline with smart optimization"""
    
    # Initialize optimizer
    optimizer = SmartPipelineOptimizer()
    
    # Get optimization strategy
    strategy = optimizer.get_optimization_strategy()
    
    print("\n🔧 Initializing Optimized Pipeline")
    print("-" * 80)
    
    # Core modules (always load these)
    core_modules = {}
    
    # Only load HF methods if resources allow
    hf_methods = {}
    
    if not strategy['use_fallbacks']:
        # Load HF methods based on priority and memory
        hf_methods = {
            'sentiment': optimizer.load_hf_method(
                'sentiment', 'hf_method1_sentiment', 'HFSentimentAnalyzer'
            ),
            'ner': optimizer.load_hf_method(
                'ner', 'hf_method2_ner', 'HFEntityExtractor'
            ),
            'classification': optimizer.load_hf_method(
                'classification', 'hf_method4_classification', 'HFEventClassifier'
            )
        }
        
        # Filter out None values
        hf_methods = {k: v for k, v in hf_methods.items() if v is not None}
    
    print("\n✓ Pipeline Initialized")
    print(f"   Core Modules: {len(core_modules)}")
    print(f"   HF Methods: {len(hf_methods)}")
    print(f"   Strategy: {strategy['memory_tier'].upper()}")
    
    return optimizer, hf_methods


if __name__ == "__main__":
    print("="*80)
    print("SMART PIPELINE OPTIMIZER - TEST")
    print("="*80)
    
    # Create optimizer
    optimizer = SmartPipelineOptimizer()
    
    # Simulate some work
    print("\n🔬 Running optimization tests...\n")
    
    # Test 1: Memory monitoring
    print("Test 1: Memory Monitoring")
    mem = optimizer.monitor.get_memory_info()
    print(f"✓ Current memory: {mem['percent_used']}%")
    
    # Test 2: Strategy determination
    print("\nTest 2: Optimization Strategy")
    strategy = optimizer.get_optimization_strategy()
    print(f"✓ Strategy determined: {strategy['memory_tier']}")
    
    # Test 3: Model loading decision
    print("\nTest 3: Model Loading Decisions")
    for model in ['sentiment', 'ner', 'forecasting', 'causal']:
        should_load = optimizer.should_load_hf_model(model)
        print(f"   {model}: {'✓ Load' if should_load else '⊘ Skip'}")
    
    # Test 4: Batch optimization
    print("\nTest 4: Batch Processing")
    test_items = list(range(25))
    
    def dummy_process(batch):
        return [x * 2 for x in batch]
    
    results = optimizer.optimize_batch_processing(test_items, dummy_process)
    print(f"✓ Processed {len(results)} items in optimized batches")
    
    # Test 5: Session report
    print("\nTest 5: Session Report")
    optimizer.print_report()
    optimizer.save_report('test_optimizer_report.json')
    
    print("\n" + "="*80)
    print("✓ All tests completed successfully")
    print("="*80)
