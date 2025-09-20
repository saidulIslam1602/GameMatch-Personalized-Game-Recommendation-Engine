"""
A/B Testing and Experimental Evaluation Framework for GameMatch
Comprehensive system for evaluating and improving recommendation models
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import uuid
import logging
from pathlib import Path
import scipy.stats as stats
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, ndcg_score
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class ExperimentStatus(Enum):
    """Status of A/B test experiments"""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ARCHIVED = "archived"

class ExperimentType(Enum):
    """Types of experiments"""
    MODEL_COMPARISON = "model_comparison"
    PROMPT_STRATEGY = "prompt_strategy"
    RAG_CONFIGURATION = "rag_configuration"
    RECOMMENDATION_COUNT = "recommendation_count"
    PERSONALIZATION_LEVEL = "personalization_level"
    UI_INTERACTION = "ui_interaction"

class MetricType(Enum):
    """Types of evaluation metrics"""
    CLICK_THROUGH_RATE = "ctr"
    USER_SATISFACTION = "satisfaction"
    RECOMMENDATION_PRECISION = "precision"
    RECOMMENDATION_RECALL = "recall"
    RESPONSE_TIME = "response_time"
    USER_ENGAGEMENT = "engagement"
    CONVERSION_RATE = "conversion"

@dataclass
class ExperimentConfiguration:
    """Configuration for A/B test experiments"""
    experiment_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    experiment_type: ExperimentType = ExperimentType.MODEL_COMPARISON
    status: ExperimentStatus = ExperimentStatus.DRAFT
    
    # Experiment parameters
    variants: Dict[str, Dict] = field(default_factory=dict)  # variant_name -> configuration
    traffic_split: Dict[str, float] = field(default_factory=dict)  # variant_name -> percentage
    
    # Success criteria
    primary_metric: MetricType = MetricType.CLICK_THROUGH_RATE
    secondary_metrics: List[MetricType] = field(default_factory=list)
    minimum_sample_size: int = 1000
    confidence_level: float = 0.95
    minimum_effect_size: float = 0.05  # 5% minimum improvement
    
    # Duration
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    max_duration_days: int = 30
    
    # Targeting
    target_users: Optional[Dict] = None  # User segmentation criteria
    exclusion_criteria: Optional[Dict] = None

@dataclass
class ExperimentResult:
    """Results from A/B test experiment"""
    experiment_id: str
    variant_name: str
    metric_type: MetricType
    sample_size: int
    metric_value: float
    confidence_interval: Tuple[float, float]
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class UserInteraction:
    """Individual user interaction for experiment tracking"""
    user_id: str
    experiment_id: str
    variant: str
    session_id: str
    timestamp: datetime
    
    # Recommendation data
    query: str
    recommendations: List[Dict]
    
    # User actions
    clicked_games: List[int] = field(default_factory=list)
    liked_games: List[int] = field(default_factory=list)
    purchased_games: List[int] = field(default_factory=list)
    time_spent_seconds: Optional[int] = None
    satisfaction_rating: Optional[int] = None
    
    # System metrics
    response_time_ms: float = 0.0
    error_occurred: bool = False

class ExperimentalEvaluationFramework:
    """Comprehensive A/B testing and evaluation framework"""
    
    def __init__(self, database_tracker=None, output_dir="results/experiments"):
        self.db_tracker = database_tracker
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory storage for experiments (in production, use database)
        self.experiments: Dict[str, ExperimentConfiguration] = {}
        self.interactions: List[UserInteraction] = []
        self.results_cache: Dict[str, List[ExperimentResult]] = defaultdict(list)
        
        # Statistical testing configuration
        self.alpha = 0.05  # significance level
        self.power = 0.8   # statistical power
        
    def create_experiment(self, config: ExperimentConfiguration) -> str:
        """Create a new A/B test experiment"""
        
        # Validate configuration
        self._validate_experiment_config(config)
        
        # Generate unique ID if not provided
        if not config.experiment_id:
            config.experiment_id = str(uuid.uuid4())
        
        # Set start date if not provided
        if not config.start_date:
            config.start_date = datetime.now()
        
        # Calculate end date if not provided
        if not config.end_date:
            config.end_date = config.start_date + timedelta(days=config.max_duration_days)
        
        # Store experiment
        self.experiments[config.experiment_id] = config
        
        logger.info(f"✅ Created experiment '{config.name}' (ID: {config.experiment_id})")
        
        return config.experiment_id
    
    def _validate_experiment_config(self, config: ExperimentConfiguration):
        """Validate experiment configuration"""
        
        if not config.name:
            raise ValueError("Experiment name is required")
        
        if not config.variants:
            raise ValueError("At least one variant is required")
        
        if not config.traffic_split:
            raise ValueError("Traffic split configuration is required")
        
        # Validate traffic split sums to 1.0
        total_traffic = sum(config.traffic_split.values())
        if abs(total_traffic - 1.0) > 0.01:
            raise ValueError(f"Traffic split must sum to 1.0, got {total_traffic}")
        
        # Ensure all variants have traffic allocation
        for variant_name in config.variants.keys():
            if variant_name not in config.traffic_split:
                raise ValueError(f"Variant '{variant_name}' missing traffic allocation")
    
    def start_experiment(self, experiment_id: str) -> bool:
        """Start an experiment"""
        
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment = self.experiments[experiment_id]
        
        if experiment.status != ExperimentStatus.DRAFT:
            raise ValueError(f"Experiment must be in DRAFT status to start")
        
        # Pre-flight checks
        sample_size_ok = self._check_sample_size_feasibility(experiment)
        if not sample_size_ok:
            logger.warning(f"⚠️ Experiment may not reach minimum sample size within duration")
        
        experiment.status = ExperimentStatus.RUNNING
        experiment.start_date = datetime.now()
        
        logger.info(f"🚀 Started experiment '{experiment.name}'")
        
        return True
    
    def assign_user_to_variant(self, experiment_id: str, user_id: str) -> str:
        """Assign user to experiment variant using consistent hashing"""
        
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment = self.experiments[experiment_id]
        
        if experiment.status != ExperimentStatus.RUNNING:
            return "control"  # Default to control if experiment not running
        
        # Check if user meets targeting criteria
        if not self._user_meets_criteria(user_id, experiment):
            return "control"
        
        # Consistent hash-based assignment
        hash_value = hash(f"{user_id}_{experiment_id}") % 10000
        cumulative_weight = 0
        
        for variant_name, weight in experiment.traffic_split.items():
            cumulative_weight += weight * 10000
            if hash_value < cumulative_weight:
                return variant_name
        
        return list(experiment.variants.keys())[0]  # Fallback
    
    def record_interaction(self, interaction: UserInteraction):
        """Record user interaction for experiment"""
        
        self.interactions.append(interaction)
        
        # Also log to database if available
        if self.db_tracker:
            try:
                self.db_tracker.log_experiment_interaction(
                    experiment_id=interaction.experiment_id,
                    user_id=interaction.user_id,
                    variant=interaction.variant,
                    query=interaction.query,
                    recommendations=interaction.recommendations,
                    actions={
                        "clicked_games": interaction.clicked_games,
                        "liked_games": interaction.liked_games,
                        "purchased_games": interaction.purchased_games,
                        "time_spent": interaction.time_spent_seconds,
                        "satisfaction": interaction.satisfaction_rating
                    },
                    response_time_ms=interaction.response_time_ms
                )
            except Exception as e:
                logger.error(f"Failed to log interaction to database: {e}")
    
    def calculate_experiment_results(self, experiment_id: str) -> Dict[str, Any]:
        """Calculate comprehensive experiment results"""
        
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment = self.experiments[experiment_id]
        experiment_interactions = [
            i for i in self.interactions 
            if i.experiment_id == experiment_id
        ]
        
        if not experiment_interactions:
            return {"error": "No interactions recorded for this experiment"}
        
        # Group interactions by variant
        variant_interactions = defaultdict(list)
        for interaction in experiment_interactions:
            variant_interactions[interaction.variant].append(interaction)
        
        # Calculate metrics for each variant
        variant_results = {}
        
        for variant_name, interactions in variant_interactions.items():
            metrics = self._calculate_variant_metrics(interactions)
            variant_results[variant_name] = metrics
        
        # Statistical significance testing
        significance_tests = self._run_significance_tests(variant_results, experiment)
        
        # Overall experiment summary
        summary = {
            "experiment_id": experiment_id,
            "experiment_name": experiment.name,
            "status": experiment.status.value,
            "start_date": experiment.start_date.isoformat() if experiment.start_date else None,
            "duration_days": (datetime.now() - experiment.start_date).days if experiment.start_date else 0,
            "total_interactions": len(experiment_interactions),
            "variants": variant_results,
            "significance_tests": significance_tests,
            "recommendations": self._generate_recommendations(variant_results, significance_tests, experiment)
        }
        
        return summary
    
    def _calculate_variant_metrics(self, interactions: List[UserInteraction]) -> Dict[str, float]:
        """Calculate metrics for a variant"""
        
        if not interactions:
            return {}
        
        metrics = {}
        
        # Basic counts
        total_interactions = len(interactions)
        total_recommendations = sum(len(i.recommendations) for i in interactions)
        
        # Click-through rate
        clicked_interactions = sum(1 for i in interactions if i.clicked_games)
        metrics["click_through_rate"] = clicked_interactions / total_interactions if total_interactions > 0 else 0
        
        # Average satisfaction
        satisfaction_scores = [i.satisfaction_rating for i in interactions if i.satisfaction_rating is not None]
        metrics["avg_satisfaction"] = np.mean(satisfaction_scores) if satisfaction_scores else 0
        
        # Engagement metrics
        metrics["avg_time_spent"] = np.mean([i.time_spent_seconds or 0 for i in interactions])
        metrics["avg_response_time"] = np.mean([i.response_time_ms for i in interactions])
        
        # Conversion rates
        purchase_interactions = sum(1 for i in interactions if i.purchased_games)
        metrics["conversion_rate"] = purchase_interactions / total_interactions if total_interactions > 0 else 0
        
        # Like rate
        like_interactions = sum(1 for i in interactions if i.liked_games)
        metrics["like_rate"] = like_interactions / total_interactions if total_interactions > 0 else 0
        
        # Error rate
        error_interactions = sum(1 for i in interactions if i.error_occurred)
        metrics["error_rate"] = error_interactions / total_interactions if total_interactions > 0 else 0
        
        # Sample size
        metrics["sample_size"] = total_interactions
        
        return metrics
    
    def _run_significance_tests(self, variant_results: Dict, experiment: ExperimentConfiguration) -> Dict[str, Any]:
        """Run statistical significance tests"""
        
        if len(variant_results) < 2:
            return {"error": "Need at least 2 variants for significance testing"}
        
        primary_metric = experiment.primary_metric.value
        variants = list(variant_results.keys())
        control_variant = variants[0]  # Assume first variant is control
        
        tests = {}
        
        for variant_name in variants[1:]:  # Test against control
            if (primary_metric in variant_results[control_variant] and 
                primary_metric in variant_results[variant_name]):
                
                control_value = variant_results[control_variant][primary_metric]
                test_value = variant_results[variant_name][primary_metric]
                
                control_n = variant_results[control_variant]["sample_size"]
                test_n = variant_results[variant_name]["sample_size"]
                
                # Two-proportion z-test for rates (CTR, conversion, etc.)
                if primary_metric in ["click_through_rate", "conversion_rate", "like_rate"]:
                    z_stat, p_value = self._two_proportion_z_test(
                        control_value, control_n, test_value, test_n
                    )
                else:
                    # Two-sample t-test for continuous metrics
                    # Note: This is simplified - in practice you'd need actual samples
                    z_stat, p_value = stats.ttest_ind_from_stats(
                        control_value, np.sqrt(control_value * (1-control_value) / control_n), control_n,
                        test_value, np.sqrt(test_value * (1-test_value) / test_n), test_n
                    )
                
                effect_size = (test_value - control_value) / control_value if control_value > 0 else 0
                is_significant = p_value < self.alpha
                
                tests[f"{control_variant}_vs_{variant_name}"] = {
                    "metric": primary_metric,
                    "control_value": control_value,
                    "test_value": test_value,
                    "effect_size": effect_size,
                    "p_value": p_value,
                    "z_statistic": z_stat,
                    "is_significant": is_significant,
                    "confidence_level": experiment.confidence_level,
                    "interpretation": self._interpret_test_result(
                        effect_size, p_value, is_significant, experiment.minimum_effect_size
                    )
                }
        
        return tests
    
    def _two_proportion_z_test(self, p1: float, n1: int, p2: float, n2: int) -> Tuple[float, float]:
        """Two-proportion z-test"""
        
        # Convert rates to counts
        x1 = int(p1 * n1)
        x2 = int(p2 * n2)
        
        # Pooled proportion
        p_pool = (x1 + x2) / (n1 + n2)
        
        # Standard error
        se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
        
        if se == 0:
            return 0, 1
        
        # Z-statistic
        z = (p1 - p2) / se
        
        # P-value (two-tailed)
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        
        return z, p_value
    
    def _interpret_test_result(self, effect_size: float, p_value: float, is_significant: bool, min_effect: float) -> str:
        """Interpret statistical test results"""
        
        if not is_significant:
            return "No statistically significant difference detected"
        
        if abs(effect_size) < min_effect:
            return f"Statistically significant but effect size ({effect_size:.3f}) below minimum threshold ({min_effect})"
        
        direction = "increase" if effect_size > 0 else "decrease"
        magnitude = "large" if abs(effect_size) > 0.2 else "moderate" if abs(effect_size) > 0.1 else "small"
        
        return f"Statistically significant {magnitude} {direction} ({effect_size:.3f}) detected"
    
    def _generate_recommendations(self, variant_results: Dict, significance_tests: Dict, experiment: ExperimentConfiguration) -> List[str]:
        """Generate actionable recommendations based on results"""
        
        recommendations = []
        
        if not significance_tests or "error" in significance_tests:
            recommendations.append("Insufficient data for statistical analysis. Continue experiment.")
            return recommendations
        
        # Find best performing variant
        primary_metric = experiment.primary_metric.value
        best_variant = max(
            variant_results.keys(),
            key=lambda v: variant_results[v].get(primary_metric, 0)
        )
        
        # Check if best variant is significantly better
        significant_improvements = [
            test for test_name, test in significance_tests.items()
            if test["is_significant"] and test["effect_size"] > experiment.minimum_effect_size
        ]
        
        if significant_improvements:
            recommendations.append(f"🎉 Implement variant '{best_variant}' - shows significant improvement")
            
            for test in significant_improvements:
                improvement = test["effect_size"] * 100
                recommendations.append(f"   • {improvement:.1f}% improvement in {test['metric']}")
        
        else:
            recommendations.append("No variant shows significant improvement over control")
            
            # Check if experiment should continue
            max_sample_size = max(v["sample_size"] for v in variant_results.values())
            if max_sample_size < experiment.minimum_sample_size:
                recommendations.append(f"Continue experiment - need {experiment.minimum_sample_size - max_sample_size} more samples")
        
        # Performance warnings
        for variant_name, metrics in variant_results.items():
            if metrics.get("error_rate", 0) > 0.05:
                recommendations.append(f"⚠️ High error rate ({metrics['error_rate']:.2%}) in variant '{variant_name}'")
            
            if metrics.get("avg_response_time", 0) > 1000:  # > 1 second
                recommendations.append(f"⚠️ Slow response time ({metrics['avg_response_time']:.0f}ms) in variant '{variant_name}'")
        
        return recommendations
    
    def _user_meets_criteria(self, user_id: str, experiment: ExperimentConfiguration) -> bool:
        """Check if user meets experiment targeting criteria"""
        
        # Simplified implementation - in practice, you'd check user attributes
        if experiment.target_users:
            # Check targeting criteria
            pass
        
        if experiment.exclusion_criteria:
            # Check exclusion criteria
            pass
        
        return True  # Default: all users eligible
    
    def _check_sample_size_feasibility(self, experiment: ExperimentConfiguration) -> bool:
        """Check if minimum sample size is achievable within experiment duration"""
        
        # Simplified check - in practice, estimate based on traffic patterns
        estimated_daily_users = 100  # This would be data-driven
        max_users = estimated_daily_users * experiment.max_duration_days
        
        return max_users >= experiment.minimum_sample_size
    
    def generate_experiment_report(self, experiment_id: str, save_to_file: bool = True) -> Dict:
        """Generate comprehensive experiment report"""
        
        results = self.calculate_experiment_results(experiment_id)
        
        if save_to_file:
            report_file = self.output_dir / f"experiment_report_{experiment_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"📊 Experiment report saved to {report_file}")
        
        return results
    
    def create_visualization_dashboard(self, experiment_id: str):
        """Create visualization dashboard for experiment"""
        
        results = self.calculate_experiment_results(experiment_id)
        
        if "error" in results:
            logger.error(f"Cannot create dashboard: {results['error']}")
            return
        
        # Create dashboard plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f"Experiment Dashboard: {results['experiment_name']}", fontsize=16)
        
        # Plot 1: Primary metric comparison
        variants = list(results["variants"].keys())
        primary_metric = results["significance_tests"][list(results["significance_tests"].keys())[0]]["metric"]
        primary_values = [results["variants"][v][primary_metric] for v in variants]
        
        axes[0, 0].bar(variants, primary_values)
        axes[0, 0].set_title(f'Primary Metric: {primary_metric}')
        axes[0, 0].set_ylabel('Value')
        
        # Plot 2: Sample sizes
        sample_sizes = [results["variants"][v]["sample_size"] for v in variants]
        axes[0, 1].bar(variants, sample_sizes)
        axes[0, 1].set_title('Sample Sizes')
        axes[0, 1].set_ylabel('Number of Users')
        
        # Plot 3: Secondary metrics heatmap
        metrics = ["click_through_rate", "avg_satisfaction", "conversion_rate", "avg_response_time"]
        metric_data = []
        for variant in variants:
            variant_metrics = []
            for metric in metrics:
                value = results["variants"][variant].get(metric, 0)
                variant_metrics.append(value)
            metric_data.append(variant_metrics)
        
        sns.heatmap(metric_data, annot=True, xticklabels=metrics, yticklabels=variants, ax=axes[1, 0])
        axes[1, 0].set_title('Metrics Heatmap')
        
        # Plot 4: Statistical significance
        significance_data = []
        test_names = []
        for test_name, test_result in results["significance_tests"].items():
            if isinstance(test_result, dict) and "p_value" in test_result:
                significance_data.append(test_result["p_value"])
                test_names.append(test_name.replace("_vs_", " vs "))
        
        if significance_data:
            axes[1, 1].bar(test_names, significance_data)
            axes[1, 1].axhline(y=0.05, color='r', linestyle='--', label='α = 0.05')
            axes[1, 1].set_title('P-values for Significance Tests')
            axes[1, 1].set_ylabel('P-value')
            axes[1, 1].legend()
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save dashboard
        dashboard_file = self.output_dir / f"experiment_dashboard_{experiment_id}.png"
        plt.savefig(dashboard_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📈 Dashboard saved to {dashboard_file}")

# Predefined experiment templates
class ExperimentTemplates:
    """Common experiment templates for GameMatch"""
    
    @staticmethod
    def model_comparison_template() -> ExperimentConfiguration:
        """Template for comparing different models"""
        
        return ExperimentConfiguration(
            name="Model Comparison: Fine-tuned vs Base GPT",
            description="Compare fine-tuned GameMatch model against base GPT-3.5",
            experiment_type=ExperimentType.MODEL_COMPARISON,
            variants={
                "control_base_gpt": {
                    "model": "gpt-3.5-turbo",
                    "temperature": 0.7,
                    "max_tokens": 300
                },
                "treatment_finetuned": {
                    "model": "ft:gpt-3.5-turbo-1106:personal::8lLwBrjg",
                    "temperature": 0.7,
                    "max_tokens": 300
                }
            },
            traffic_split={
                "control_base_gpt": 0.5,
                "treatment_finetuned": 0.5
            },
            primary_metric=MetricType.CLICK_THROUGH_RATE,
            secondary_metrics=[MetricType.USER_SATISFACTION, MetricType.RESPONSE_TIME],
            minimum_sample_size=1000,
            minimum_effect_size=0.05
        )
    
    @staticmethod
    def prompt_strategy_template() -> ExperimentConfiguration:
        """Template for testing different prompting strategies"""
        
        return ExperimentConfiguration(
            name="Prompt Strategy: Few-shot vs Chain-of-thought",
            description="Compare different prompting strategies for recommendation quality",
            experiment_type=ExperimentType.PROMPT_STRATEGY,
            variants={
                "control_basic": {
                    "prompt_strategy": "zero_shot",
                    "include_examples": False
                },
                "treatment_few_shot": {
                    "prompt_strategy": "few_shot",
                    "num_examples": 3
                },
                "treatment_cot": {
                    "prompt_strategy": "chain_of_thought",
                    "reasoning_steps": True
                }
            },
            traffic_split={
                "control_basic": 0.34,
                "treatment_few_shot": 0.33,
                "treatment_cot": 0.33
            },
            primary_metric=MetricType.USER_SATISFACTION,
            secondary_metrics=[MetricType.CLICK_THROUGH_RATE, MetricType.RESPONSE_TIME]
        )
    
    @staticmethod
    def rag_configuration_template() -> ExperimentConfiguration:
        """Template for testing RAG system configurations"""
        
        return ExperimentConfiguration(
            name="RAG Configuration: Retrieval Strategy Comparison",
            description="Compare different RAG retrieval strategies",
            experiment_type=ExperimentType.RAG_CONFIGURATION,
            variants={
                "control_semantic": {
                    "retrieval_strategy": "semantic_similarity",
                    "top_k": 10
                },
                "treatment_hybrid": {
                    "retrieval_strategy": "hybrid_search",
                    "top_k": 10
                },
                "treatment_collaborative": {
                    "retrieval_strategy": "collaborative_filter",
                    "top_k": 10
                }
            },
            traffic_split={
                "control_semantic": 0.34,
                "treatment_hybrid": 0.33,
                "treatment_collaborative": 0.33
            },
            primary_metric=MetricType.RECOMMENDATION_PRECISION,
            secondary_metrics=[MetricType.CLICK_THROUGH_RATE, MetricType.USER_SATISFACTION]
        )

# Example usage and testing
def test_evaluation_framework():
    """Test the experimental evaluation framework"""
    
    print("🧪 Testing Experimental Evaluation Framework")
    print("=" * 50)
    
    # Initialize framework
    framework = ExperimentalEvaluationFramework()
    
    # Create test experiment
    config = ExperimentTemplates.model_comparison_template()
    experiment_id = framework.create_experiment(config)
    
    print(f"✅ Created experiment: {experiment_id}")
    
    # Start experiment
    framework.start_experiment(experiment_id)
    print("✅ Started experiment")
    
    # Simulate user interactions
    test_users = ["user_001", "user_002", "user_003", "user_004", "user_005"]
    
    for i, user_id in enumerate(test_users):
        variant = framework.assign_user_to_variant(experiment_id, user_id)
        
        interaction = UserInteraction(
            user_id=user_id,
            experiment_id=experiment_id,
            variant=variant,
            session_id=f"session_{i}",
            timestamp=datetime.now(),
            query="games like The Witcher 3",
            recommendations=[{"game_id": 123, "title": "Dragon Age"}],
            clicked_games=[123] if i % 2 == 0 else [],
            satisfaction_rating=4 if i % 2 == 0 else 3,
            response_time_ms=250.0 + i * 50
        )
        
        framework.record_interaction(interaction)
    
    print(f"✅ Recorded {len(test_users)} test interactions")
    
    # Calculate results
    results = framework.calculate_experiment_results(experiment_id)
    
    print("\n📊 Experiment Results:")
    print(f"   • Total interactions: {results['total_interactions']}")
    print(f"   • Variants tested: {list(results['variants'].keys())}")
    
    # Generate report
    report = framework.generate_experiment_report(experiment_id)
    print(f"✅ Generated experiment report")
    
    print("\n🎯 A/B Testing Framework Ready!")

if __name__ == "__main__":
    test_evaluation_framework()