"""
GameMatch MLOps Monitoring and Model Performance Tracking
Production-ready monitoring for recommendation system performance
"""

import pandas as pd
import numpy as np
import json
import time
import logging
import os
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
import threading
import psutil
try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    
# Microsoft SQL Server support
try:
    import pyodbc
    MSSQL_AVAILABLE = True
except ImportError:
    try:
        import pymssql
        MSSQL_AVAILABLE = True
        MSSQL_DRIVER = 'pymssql'
    except ImportError:
        MSSQL_AVAILABLE = False
        MSSQL_DRIVER = None

logger = logging.getLogger(__name__)

@dataclass
class ModelMetrics:
    """Model performance metrics"""
    model_version: str
    timestamp: datetime
    response_time_ms: float
    recommendation_count: int
    user_satisfaction_score: float
    click_through_rate: float
    conversion_rate: float
    diversity_score: float
    coverage_score: float
    novelty_score: float
    confidence_distribution: Dict[str, float]
    genre_distribution: Dict[str, int]
    price_distribution: Dict[str, int]

@dataclass
class SystemMetrics:
    """System resource metrics"""
    timestamp: datetime
    cpu_usage_percent: float
    memory_usage_gb: float
    memory_usage_percent: float
    gpu_usage_percent: float
    gpu_memory_gb: float
    disk_usage_percent: float
    network_io_mb: float
    response_queue_size: int
    active_connections: int

@dataclass
class BusinessMetrics:
    """Business impact metrics"""
    timestamp: datetime
    total_recommendations: int
    unique_users: int
    user_engagement_score: float
    recommendation_adoption_rate: float
    user_retention_rate: float
    revenue_impact_estimate: float
    cost_per_recommendation: float
    model_accuracy: float

class RecommendationTracker:
    """Track recommendation performance and user feedback"""
    
    def __init__(self, server: str = None, port: int = None, database: str = None, 
                 username: str = None, password: str = None):
        # Use environment variables with secure defaults
        self.server = server or os.getenv('MSSQL_SERVER', 'localhost')
        self.port = port or int(os.getenv('MSSQL_PORT', '1433'))
        self.database = database or os.getenv('MSSQL_DATABASE', 'gamematch')
        self.username = username or os.getenv('MSSQL_USERNAME', 'sa')
        self.password = password or os.getenv('MSSQL_PASSWORD')
        
        if not self.password:
            logger.warning("MSSQL_PASSWORD not set, monitoring will be disabled")
            self.password = None
            self.connection_string = None
        else:
            try:
                self.connection_string = self._build_connection_string()
                self._init_database()
                logger.info("Database monitoring initialized successfully")
            except Exception as e:
                logger.warning(f"Database initialization failed: {e}")
                self.connection_string = None
    
    def _build_connection_string(self) -> str:
        """Build MSSQL connection string"""
        if not MSSQL_AVAILABLE:
            raise ImportError("No MSSQL driver available. Install pyodbc or pymssql.")
        
        if 'pyodbc' in globals():
            if self.username and self.password:
                return f"DRIVER={{ODBC Driver 18 for SQL Server}};SERVER={self.server},{self.port};DATABASE={self.database};UID={self.username};PWD={self.password};TrustServerCertificate=yes"
            else:
                return f"DRIVER={{ODBC Driver 18 for SQL Server}};SERVER={self.server},{self.port};DATABASE={self.database};Trusted_Connection=yes;TrustServerCertificate=yes"
        else:  # pymssql
            if self.username and self.password:
                return f"server={self.server};port={self.port};database={self.database};user={self.username};password={self.password}"
            else:
                return f"server={self.server};port={self.port};database={self.database};trusted_connection=yes"
    
    def _get_connection(self):
        """Get MSSQL database connection"""
        if 'pyodbc' in globals():
            return pyodbc.connect(self.connection_string)
        else:  # pymssql
            return pymssql.connect(
                server=self.server, 
                port=self.port,
                database=self.database,
                user=self.username, 
                password=self.password
            )
        
    def _init_database(self):
        """Initialize MSSQL database for tracking"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Create recommendations table
            cursor.execute('''
                IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='recommendations' AND xtype='U')
                CREATE TABLE recommendations (
                    id INT IDENTITY(1,1) PRIMARY KEY,
                    timestamp DATETIME2 DEFAULT GETDATE(),
                    query NVARCHAR(MAX),
                    user_id NVARCHAR(255),
                    game_ids NVARCHAR(MAX),
                    response_time_ms FLOAT,
                    confidence_score FLOAT,
                    model_version NVARCHAR(255)
                )
            ''')
            
            # Create user_feedback table
            cursor.execute('''
                IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='user_feedback' AND xtype='U')
                CREATE TABLE user_feedback (
                    id INT IDENTITY(1,1) PRIMARY KEY,
                    recommendation_id INT,
                    timestamp DATETIME2 DEFAULT GETDATE(),
                    feedback_type NVARCHAR(255),
                    rating INT,
                    clicked_games NVARCHAR(MAX),
                    purchased_games NVARCHAR(MAX),
                    time_spent_seconds INT,
                    FOREIGN KEY (recommendation_id) REFERENCES recommendations (id)
                )
            ''')
            
            # Create model_performance table
            cursor.execute('''
                IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='model_performance' AND xtype='U')
                CREATE TABLE model_performance (
                    id INT IDENTITY(1,1) PRIMARY KEY,
                    timestamp DATETIME2 DEFAULT GETDATE(),
                    model_version NVARCHAR(255),
                    metric_name NVARCHAR(255),
                    metric_value FLOAT,
                    metadata NVARCHAR(MAX)
                )
            ''')
            
            conn.commit()
    
    def log_recommendation(self, 
                         query: str, 
                         user_id: str, 
                         game_ids: List[int],
                         response_time_ms: float,
                         confidence_score: float,
                         model_version: str) -> int:
        """Log a recommendation request"""
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO recommendations 
                (query, user_id, game_ids, response_time_ms, confidence_score, model_version)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (query, user_id, json.dumps(game_ids), response_time_ms, confidence_score, model_version))
            
            # Get the inserted ID (SQL Server specific)
            cursor.execute("SELECT @@IDENTITY")
            result = cursor.fetchone()
            conn.commit()
            return int(result[0]) if result else 0
    
    def log_user_feedback(self,
                         recommendation_id: int,
                         feedback_type: str,
                         rating: Optional[int] = None,
                         clicked_games: Optional[List[int]] = None,
                         purchased_games: Optional[List[int]] = None,
                         time_spent_seconds: Optional[int] = None):
        """Log user feedback on recommendations"""
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO user_feedback 
                (recommendation_id, feedback_type, rating, clicked_games, purchased_games, time_spent_seconds)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (recommendation_id, feedback_type, rating, 
                  json.dumps(clicked_games) if clicked_games else None,
                  json.dumps(purchased_games) if purchased_games else None,
                  time_spent_seconds))
            conn.commit()
    
    def log_metric(self, model_version: str, metric_name: str, metric_value: float, metadata: Dict = None):
        """Log a performance metric"""
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO model_performance (model_version, metric_name, metric_value, metadata)
                VALUES (?, ?, ?, ?)
            ''', (model_version, metric_name, metric_value, json.dumps(metadata) if metadata else None))
            conn.commit()

class PerformanceMonitor:
    """Monitor system and model performance in real-time"""
    
    def __init__(self, tracker: RecommendationTracker):
        self.tracker = tracker
        self.metrics_history = defaultdict(lambda: deque(maxlen=1000))
        self.monitoring_active = False
        self.monitor_thread = None
        
    def start_monitoring(self, interval_seconds: int = 60):
        """Start continuous monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, args=(interval_seconds,))
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        logger.info(f"Started performance monitoring (interval: {interval_seconds}s)")
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Performance monitoring stopped")
    
    def _monitoring_loop(self, interval_seconds: int):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Collect system metrics
                system_metrics = self._collect_system_metrics()
                self.metrics_history['system'].append(system_metrics)
                
                # Log critical metrics to database
                self.tracker.log_metric(
                    "system", "cpu_usage", system_metrics.cpu_usage_percent
                )
                self.tracker.log_metric(
                    "system", "memory_usage", system_metrics.memory_usage_percent
                )
                
                # Collect model metrics (if recommendations happened recently)
                model_metrics = self._calculate_model_metrics()
                if model_metrics:
                    self.metrics_history['model'].append(model_metrics)
                
                time.sleep(interval_seconds)
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(interval_seconds)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        
        # CPU and Memory
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        # GPU (if available)
        gpu_usage = 0
        gpu_memory = 0
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                gpu_usage = gpu.load * 100
                gpu_memory = gpu.memoryUsed
        except:
            pass
        
        # Disk
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        
        # Network
        network = psutil.net_io_counters()
        network_mb = (network.bytes_sent + network.bytes_recv) / (1024 * 1024)
        
        return SystemMetrics(
            timestamp=datetime.now(),
            cpu_usage_percent=cpu_percent,
            memory_usage_gb=memory.used / (1024**3),
            memory_usage_percent=memory.percent,
            gpu_usage_percent=gpu_usage,
            gpu_memory_gb=gpu_memory / 1024,
            disk_usage_percent=disk_percent,
            network_io_mb=network_mb,
            response_queue_size=0,  # Would be populated by application
            active_connections=0    # Would be populated by application
        )
    
    def _calculate_model_metrics(self) -> Optional[ModelMetrics]:
        """Calculate model performance metrics from recent data"""
        
        # Get recent recommendations (last hour)
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=1)
        
        with self.tracker._get_connection() as conn:
            # Get recommendations
            recommendations_df = pd.read_sql_query('''
                SELECT * FROM recommendations 
                WHERE timestamp >= ? AND timestamp <= ?
            ''', conn, params=(start_time, end_time))
            
            if recommendations_df.empty:
                return None
            
            # Get feedback
            feedback_df = pd.read_sql_query('''
                SELECT uf.*, r.timestamp as rec_timestamp
                FROM user_feedback uf
                JOIN recommendations r ON uf.recommendation_id = r.id
                WHERE r.timestamp >= ? AND r.timestamp <= ?
            ''', conn, params=(start_time, end_time))
        
        # Calculate metrics
        avg_response_time = recommendations_df['response_time_ms'].mean()
        recommendation_count = len(recommendations_df)
        
        # User satisfaction from feedback
        satisfaction_scores = feedback_df[feedback_df['rating'].notna()]['rating']
        user_satisfaction = satisfaction_scores.mean() / 5.0 if len(satisfaction_scores) > 0 else 0.5
        
        # Click-through rate
        clicked_feedback = feedback_df[feedback_df['clicked_games'].notna()]
        ctr = len(clicked_feedback) / recommendation_count if recommendation_count > 0 else 0
        
        # Conversion rate (purchases)
        purchased_feedback = feedback_df[feedback_df['purchased_games'].notna()]
        conversion_rate = len(purchased_feedback) / recommendation_count if recommendation_count > 0 else 0
        
        # Diversity and coverage (simplified)
        unique_games = set()
        for game_ids_str in recommendations_df['game_ids']:
            game_ids = json.loads(game_ids_str)
            unique_games.update(game_ids)
        
        diversity_score = len(unique_games) / max(recommendation_count * 5, 1)  # Assume 5 recs per request
        coverage_score = len(unique_games) / 10000  # Assume 10k total games
        
        # Confidence distribution
        confidence_scores = recommendations_df['confidence_score']
        confidence_dist = {
            'high': (confidence_scores >= 0.8).sum() / len(confidence_scores),
            'medium': ((confidence_scores >= 0.5) & (confidence_scores < 0.8)).sum() / len(confidence_scores),
            'low': (confidence_scores < 0.5).sum() / len(confidence_scores)
        }
        
        return ModelMetrics(
            model_version=recommendations_df['model_version'].iloc[-1],
            timestamp=datetime.now(),
            response_time_ms=avg_response_time,
            recommendation_count=recommendation_count,
            user_satisfaction_score=user_satisfaction,
            click_through_rate=ctr,
            conversion_rate=conversion_rate,
            diversity_score=diversity_score,
            coverage_score=coverage_score,
            novelty_score=0.5,  # Would need more complex calculation
            confidence_distribution=confidence_dist,
            genre_distribution={},  # Would extract from recommendations
            price_distribution={}   # Would extract from recommendations
        )
    
    def get_performance_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance summary for the last N hours"""
        
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        with self.tracker._get_connection() as conn:
            # Get metrics
            metrics_df = pd.read_sql_query('''
                SELECT * FROM model_performance 
                WHERE timestamp >= ? AND timestamp <= ?
            ''', conn, params=(start_time, end_time))
            
            # Get recommendations
            recs_df = pd.read_sql_query('''
                SELECT * FROM recommendations 
                WHERE timestamp >= ? AND timestamp <= ?
            ''', conn, params=(start_time, end_time))
            
            # Get feedback
            feedback_df = pd.read_sql_query('''
                SELECT uf.*, r.timestamp as rec_timestamp
                FROM user_feedback uf
                JOIN recommendations r ON uf.recommendation_id = r.id
                WHERE r.timestamp >= ? AND r.timestamp <= ?
            ''', conn, params=(start_time, end_time))
        
        # Calculate summary statistics
        total_requests = len(recs_df)
        avg_response_time = recs_df['response_time_ms'].mean() if total_requests > 0 else 0
        avg_confidence = recs_df['confidence_score'].mean() if total_requests > 0 else 0
        
        # User engagement
        total_feedback = len(feedback_df)
        engagement_rate = total_feedback / total_requests if total_requests > 0 else 0
        
        # System performance
        cpu_metrics = metrics_df[metrics_df['metric_name'] == 'cpu_usage']
        avg_cpu = cpu_metrics['metric_value'].mean() if len(cpu_metrics) > 0 else 0
        
        memory_metrics = metrics_df[metrics_df['metric_name'] == 'memory_usage']
        avg_memory = memory_metrics['metric_value'].mean() if len(memory_metrics) > 0 else 0
        
        return {
            'time_period': f'Last {hours} hours',
            'model_performance': {
                'total_requests': total_requests,
                'avg_response_time_ms': avg_response_time,
                'avg_confidence_score': avg_confidence,
                'user_engagement_rate': engagement_rate
            },
            'system_performance': {
                'avg_cpu_usage': avg_cpu,
                'avg_memory_usage': avg_memory
            },
            'user_satisfaction': {
                'total_feedback_events': total_feedback,
                'feedback_types': feedback_df['feedback_type'].value_counts().to_dict() if total_feedback > 0 else {}
            }
        }

class AlertSystem:
    """Alert system for performance issues"""
    
    def __init__(self, monitor: PerformanceMonitor):
        self.monitor = monitor
        self.alert_thresholds = {
            'response_time_ms': 1000,
            'cpu_usage_percent': 80,
            'memory_usage_percent': 85,
            'confidence_score': 0.3,
            'user_satisfaction': 0.4
        }
        self.alerts_history = deque(maxlen=100)
    
    def check_alerts(self):
        """Check for alert conditions"""
        alerts = []
        
        # Check recent system metrics
        if self.monitor.metrics_history['system']:
            latest_system = self.monitor.metrics_history['system'][-1]
            
            if latest_system.cpu_usage_percent > self.alert_thresholds['cpu_usage_percent']:
                alerts.append({
                    'type': 'system',
                    'severity': 'warning',
                    'message': f'High CPU usage: {latest_system.cpu_usage_percent:.1f}%',
                    'timestamp': datetime.now()
                })
            
            if latest_system.memory_usage_percent > self.alert_thresholds['memory_usage_percent']:
                alerts.append({
                    'type': 'system', 
                    'severity': 'warning',
                    'message': f'High memory usage: {latest_system.memory_usage_percent:.1f}%',
                    'timestamp': datetime.now()
                })
        
        # Check model metrics
        if self.monitor.metrics_history['model']:
            latest_model = self.monitor.metrics_history['model'][-1]
            
            if latest_model.response_time_ms > self.alert_thresholds['response_time_ms']:
                alerts.append({
                    'type': 'model',
                    'severity': 'warning',
                    'message': f'Slow response time: {latest_model.response_time_ms:.1f}ms',
                    'timestamp': datetime.now()
                })
            
            if latest_model.user_satisfaction_score < self.alert_thresholds['user_satisfaction']:
                alerts.append({
                    'type': 'model',
                    'severity': 'critical',
                    'message': f'Low user satisfaction: {latest_model.user_satisfaction_score:.2f}',
                    'timestamp': datetime.now()
                })
        
        # Store alerts
        for alert in alerts:
            self.alerts_history.append(alert)
            logger.warning(f"ALERT: {alert['message']}")
        
        return alerts
    
    def get_recent_alerts(self, hours: int = 24) -> List[Dict]:
        """Get alerts from the last N hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        return [
            alert for alert in self.alerts_history 
            if alert['timestamp'] > cutoff_time
        ]

class MLOpsReportGenerator:
    """Generate comprehensive MLOps reports"""
    
    def __init__(self, tracker: RecommendationTracker, monitor: PerformanceMonitor):
        self.tracker = tracker
        self.monitor = monitor
    
    def generate_daily_report(self) -> Dict[str, Any]:
        """Generate daily performance report"""
        
        # Get performance summary
        performance = self.monitor.get_performance_summary(hours=24)
        
        # Calculate model health score
        health_score = self._calculate_health_score(performance)
        
        # Get top queries and games
        top_queries, top_games = self._get_popular_content()
        
        return {
            'report_date': datetime.now().isoformat(),
            'report_type': 'daily',
            'model_health_score': health_score,
            'performance_summary': performance,
            'popular_content': {
                'top_queries': top_queries,
                'recommended_games': top_games
            },
            'recommendations': self._generate_recommendations(performance),
            'next_review': (datetime.now() + timedelta(days=1)).isoformat()
        }
    
    def _calculate_health_score(self, performance: Dict) -> Dict[str, float]:
        """Calculate overall model health score"""
        
        model_perf = performance.get('model_performance', {})
        system_perf = performance.get('system_performance', {})
        
        # Response time score (0-1, where 1 is < 200ms, 0 is > 2000ms)
        response_time = model_perf.get('avg_response_time_ms', 1000)
        response_score = max(0, min(1, (2000 - response_time) / 1800))
        
        # Confidence score (directly usable)
        confidence_score = model_perf.get('avg_confidence_score', 0.5)
        
        # System performance score
        cpu_usage = system_perf.get('avg_cpu_usage', 50)
        system_score = max(0, min(1, (90 - cpu_usage) / 90))
        
        # Engagement score
        engagement_rate = model_perf.get('user_engagement_rate', 0.1)
        engagement_score = min(1, engagement_rate * 5)  # Scale 20% engagement = 1.0
        
        # Overall health (weighted average)
        overall_health = (
            response_score * 0.3 +
            confidence_score * 0.3 +
            system_score * 0.2 +
            engagement_score * 0.2
        )
        
        return {
            'overall_health': overall_health,
            'response_time_score': response_score,
            'confidence_score': confidence_score,
            'system_performance_score': system_score,
            'user_engagement_score': engagement_score
        }
    
    def _get_popular_content(self) -> Tuple[List[Dict], List[Dict]]:
        """Get popular queries and recommended games"""
        
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=24)
        
        with self.tracker._get_connection() as conn:
            # Top queries
            queries_df = pd.read_sql_query('''
                SELECT query, COUNT(*) as count, AVG(confidence_score) as avg_confidence
                FROM recommendations 
                WHERE timestamp >= ? AND timestamp <= ?
                GROUP BY query
                ORDER BY count DESC
                LIMIT 10
            ''', conn, params=(start_time, end_time))
            
            top_queries = queries_df.to_dict('records')
            
            # Top recommended games (would need game metadata)
            games_df = pd.read_sql_query('''
                SELECT game_ids, COUNT(*) as recommendation_count
                FROM recommendations 
                WHERE timestamp >= ? AND timestamp <= ?
                GROUP BY game_ids
                ORDER BY recommendation_count DESC
                LIMIT 10
            ''', conn, params=(start_time, end_time))
            
            top_games = games_df.to_dict('records')
        
        return top_queries, top_games
    
    def _generate_recommendations(self, performance: Dict) -> List[str]:
        """Generate improvement recommendations"""
        
        recommendations = []
        model_perf = performance.get('model_performance', {})
        
        avg_response_time = model_perf.get('avg_response_time_ms', 0)
        if avg_response_time > 800:
            recommendations.append("Consider optimizing response time through caching or model compression")
        
        avg_confidence = model_perf.get('avg_confidence_score', 1)
        if avg_confidence < 0.6:
            recommendations.append("Review and retrain model with more diverse training data")
        
        engagement_rate = model_perf.get('user_engagement_rate', 1)
        if engagement_rate < 0.15:
            recommendations.append("Analyze user feedback patterns to improve recommendation relevance")
        
        if not recommendations:
            recommendations.append("System performing well - continue monitoring")
        
        return recommendations
    
    def export_report(self, report: Dict, filepath: str):
        """Export report to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"MLOps report exported to {filepath}")

# Usage example and testing
def test_mlops_system():
    """Test MLOps monitoring system"""
    logger.info("Testing MLOps monitoring system...")
    
    # Initialize components
    tracker = RecommendationTracker("test_recommendations.db")
    monitor = PerformanceMonitor(tracker)
    alert_system = AlertSystem(monitor)
    reporter = MLOpsReportGenerator(tracker, monitor)
    
    # Simulate some recommendations
    rec_id = tracker.log_recommendation(
        query="RPG games under $20",
        user_id="user123",
        game_ids=[1, 2, 3],
        response_time_ms=245.5,
        confidence_score=0.85,
        model_version="GameMatch-v2.1"
    )
    
    # Simulate user feedback
    tracker.log_user_feedback(
        recommendation_id=rec_id,
        feedback_type="positive",
        rating=4,
        clicked_games=[1, 3],
        time_spent_seconds=120
    )
    
    # Test performance summary
    summary = monitor.get_performance_summary(hours=1)
    print(f"Performance summary: {summary}")
    
    # Test alerts
    alerts = alert_system.check_alerts()
    print(f"Current alerts: {len(alerts)}")
    
    # Generate report
    report = reporter.generate_daily_report()
    print(f"Health score: {report['model_health_score']['overall_health']:.2f}")
    
    # Export report
    reporter.export_report(report, "test_mlops_report.json")
    
    print("✅ MLOps system test completed successfully")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_mlops_system()