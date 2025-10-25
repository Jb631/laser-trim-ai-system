#!/usr/bin/env python3
"""
Example script demonstrating database performance optimization features.

This script shows how to use the performance optimizer to:
- Monitor query performance
- Implement caching strategies
- Optimize batch operations
- Create indexes based on usage patterns
- Use prepared statements
"""

import time
import json
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from laser_trim_analyzer.database.manager import DatabaseManager
from laser_trim_analyzer.core.models import (
    AnalysisResult, FileMetadata, TrackData, SigmaAnalysis,
    AnalysisStatus, SystemType
)


def create_sample_analysis(index: int) -> AnalysisResult:
    """Create a sample analysis result for testing."""
    metadata = FileMetadata(
        filename=f"sample_{index}.xls",
        file_path=Path(f"/test/sample_{index}.xls"),
        file_date=datetime.now() - timedelta(days=index),
        model=f"MODEL_{index % 5}",
        serial=f"SERIAL_{index:05d}",
        system=SystemType.SYSTEM_A if index % 2 == 0 else SystemType.SYSTEM_B,
        has_multi_tracks=index % 2 == 0
    )
    
    tracks = {}
    num_tracks = 2 if metadata.has_multi_tracks else 1
    
    for i in range(num_tracks):
        track_id = f"TRK{i+1}" if metadata.has_multi_tracks else "default"
        tracks[track_id] = TrackData(
            track_id=track_id,
            status=AnalysisStatus.PASS if (index + i) % 3 != 0 else AnalysisStatus.FAIL,
            travel_length=10.0 + i,
            sigma_analysis=SigmaAnalysis(
                sigma_gradient=0.5 + (index * 0.01),
                sigma_threshold=1.0,
                sigma_pass=(0.5 + (index * 0.01)) < 1.0,
                gradient_margin=0.5 - (index * 0.01)
            )
        )
    
    return AnalysisResult(
        metadata=metadata,
        tracks=tracks,
        overall_status=AnalysisStatus.PASS if index % 4 != 0 else AnalysisStatus.FAIL,
        processing_time=0.5 + (index * 0.1)
    )


def demonstrate_performance_features():
    """Demonstrate various performance optimization features."""
    
    print("=== Database Performance Optimization Demo ===\n")
    
    # Initialize database manager with performance optimization
    db_manager = DatabaseManager(
        database_url=None,  # Use default SQLite
        enable_performance_optimization=True,
        cache_size=500,
        cache_ttl=300,
        echo=False  # Set to True to see SQL queries
    )
    
    # Initialize database schema
    print("1. Initializing database...")
    db_manager.init_db(drop_existing=True)
    
    # Prepare common queries for better performance
    print("\n2. Preparing common queries...")
    db_manager.prepare_common_queries()
    
    # Insert sample data
    print("\n3. Inserting sample data...")
    start_time = time.time()
    
    # Create batch of analyses
    analyses = [create_sample_analysis(i) for i in range(100)]
    
    # Save using batch optimization
    saved_ids = db_manager.batch_save_analyses(analyses, batch_size=20)
    
    insert_time = time.time() - start_time
    print(f"   Inserted {len(saved_ids)} records in {insert_time:.2f} seconds")
    print(f"   Average: {insert_time/len(saved_ids)*1000:.2f} ms per record")
    
    # Test query performance with caching
    print("\n4. Testing query performance with caching...")
    
    # First query - cache miss
    start_time = time.time()
    stats1 = db_manager.get_model_statistics("MODEL_0")
    query1_time = time.time() - start_time
    print(f"   First query (cache miss): {query1_time*1000:.2f} ms")
    
    # Second query - cache hit
    start_time = time.time()
    stats2 = db_manager.get_model_statistics("MODEL_0")
    query2_time = time.time() - start_time
    print(f"   Second query (cache hit): {query2_time*1000:.2f} ms")
    print(f"   Speed improvement: {(query1_time/query2_time):.1f}x faster")
    
    # Test historical data queries
    print("\n5. Testing historical data queries...")
    
    # Query with multiple filters
    start_time = time.time()
    historical = db_manager.get_historical_data(
        model="MODEL_%",  # Wildcard search
        days_back=30,
        status="Pass",
        limit=10
    )
    historical_time = time.time() - start_time
    print(f"   Historical query: {historical_time*1000:.2f} ms for {len(historical)} records")
    
    # Get performance report
    print("\n6. Performance Report:")
    report = db_manager.get_performance_report()
    
    print(f"   - Profiling enabled: {report['profiling']['enabled']}")
    print(f"   - Total queries profiled: {report['profiling']['total_queries']}")
    print(f"   - Slow queries detected: {report['profiling']['slow_queries']}")
    print(f"   - Cache hit rate: {report['caching']['hit_rate']}%")
    print(f"   - Cache size: {report['caching']['cache_size']}/{report['caching']['max_size']}")
    
    # Show connection pool status
    pool_status = report['connection_pool']
    print(f"\n   Connection Pool:")
    print(f"   - Pool size: {pool_status['size']}")
    print(f"   - Active connections: {pool_status['checked_out']}")
    print(f"   - Average wait time: {pool_status['avg_wait_time']} seconds")
    
    # Get slow queries
    print("\n7. Analyzing slow queries...")
    slow_queries = db_manager.get_slow_queries(threshold=0.05)  # 50ms threshold
    
    if slow_queries:
        print(f"   Found {len(slow_queries)} slow queries:")
        for query in slow_queries[:3]:  # Show top 3
            print(f"   - Query: {query['query_text'][:60]}...")
            print(f"     Avg time: {query['avg_time']*1000:.2f} ms")
            print(f"     Executions: {query['execution_count']}")
    else:
        print("   No slow queries detected")
    
    # Test index optimization
    print("\n8. Index optimization analysis...")
    
    # Create some slow queries by querying without indexes
    for i in range(10):
        # These queries might be slow without proper indexes
        db_manager.get_historical_data(
            risk_category="High",
            limit=5
        )
    
    # Get optimization suggestions
    suggestions = db_manager.performance_optimizer._generate_suggestions()
    
    if suggestions:
        print(f"   Found {len(suggestions)} optimization suggestions:")
        for suggestion in suggestions:
            print(f"   - Type: {suggestion['type']}")
            print(f"     Severity: {suggestion['severity']}")
            print(f"     Message: {suggestion['message']}")
            print(f"     Action: {suggestion['action']}")
    
    # Create suggested indexes
    print("\n9. Creating performance indexes...")
    created_indexes = db_manager.create_suggested_indexes()
    
    if created_indexes:
        print(f"   Created {len(created_indexes)} indexes:")
        for index in created_indexes:
            print(f"   - {index['name']} on {index['table']}({', '.join(index['columns'])})")
    else:
        print("   No indexes needed or all suggested indexes already exist")
    
    # Run automatic optimization
    print("\n10. Running automatic performance optimization...")
    optimization_results = db_manager.optimize_performance()
    
    print(f"   Optimization completed at: {optimization_results['timestamp']}")
    print(f"   Applied {len(optimization_results['optimizations'])} optimizations")
    
    # Demonstrate cache invalidation
    print("\n11. Cache management...")
    
    # Clear specific cache entries
    db_manager.clear_query_cache("model_stats:MODEL_0")
    print("   Cleared cache for MODEL_0 statistics")
    
    # Query again to test cache miss
    start_time = time.time()
    stats3 = db_manager.get_model_statistics("MODEL_0")
    query3_time = time.time() - start_time
    print(f"   Query after cache clear: {query3_time*1000:.2f} ms")
    
    # Export performance profiles
    print("\n12. Exporting performance data...")
    
    if db_manager.performance_optimizer:
        export_path = Path("performance_profiles.json")
        db_manager.performance_optimizer.export_profiles(str(export_path))
        print(f"   Exported profiles to: {export_path}")
        
        # Show sample of exported data
        with open(export_path) as f:
            profile_data = json.load(f)
        
        print(f"   Total profiles exported: {len(profile_data['profiles'])}")
        
        # Clean up
        export_path.unlink()
    
    # Final performance summary
    print("\n=== Performance Summary ===")
    final_report = db_manager.get_performance_report()
    
    print(f"Cache hit rate: {final_report['caching']['hit_rate']}%")
    print(f"Total queries: {final_report['profiling']['total_queries']}")
    print(f"Slow queries: {final_report['profiling']['slow_queries']}")
    
    # Show top queries by frequency
    if final_report['profiling']['top_queries']:
        print("\nMost frequent queries:")
        for query in final_report['profiling']['top_queries'][:3]:
            print(f"- {query['query_text'][:50]}... ({query['execution_count']} times)")
    
    print("\n✓ Performance optimization demo completed!")


def demonstrate_advanced_features():
    """Demonstrate advanced performance features."""
    
    print("\n\n=== Advanced Performance Features ===\n")
    
    # Initialize database manager
    db_manager = DatabaseManager(
        enable_performance_optimization=True,
        cache_size=100,
        cache_ttl=60
    )
    
    # Use prepared statements for complex queries
    print("1. Using prepared statements...")
    
    if db_manager.performance_optimizer:
        prepared = db_manager.performance_optimizer.prepared_statements
        
        # Prepare a complex query
        complex_query = """
        SELECT 
            a.model,
            COUNT(DISTINCT a.id) as total_analyses,
            COUNT(DISTINCT t.id) as total_tracks,
            AVG(t.sigma_gradient) as avg_sigma,
            SUM(CASE WHEN t.risk_category = 'High' THEN 1 ELSE 0 END) as high_risk_count
        FROM analysis_results a
        JOIN track_results t ON a.id = t.analysis_id
        WHERE a.timestamp >= :start_date
        GROUP BY a.model
        ORDER BY high_risk_count DESC
        """
        
        prepared.prepare('model_risk_summary', complex_query)
        
        # Execute prepared statement
        start_date = datetime.now() - timedelta(days=30)
        result = prepared.execute('model_risk_summary', {'start_date': start_date})
        
        print("   Prepared statement executed successfully")
        rows = result.fetchall()
        print(f"   Retrieved {len(rows)} model summaries")
    
    # Demonstrate batch query optimization
    print("\n2. Batch query optimization...")
    
    if db_manager.performance_optimizer:
        batch_optimizer = db_manager.performance_optimizer.batch_optimizer
        
        # Get IDs for batch processing
        with db_manager.get_session() as session:
            analysis_ids = [row[0] for row in session.query(DBAnalysisResult.id).limit(50).all()]
        
        # Define batch query function
        def get_analyses_by_ids(ids):
            with db_manager.get_session() as session:
                return session.query(DBAnalysisResult).filter(
                    DBAnalysisResult.id.in_(ids)
                ).all()
        
        # Execute in batches
        start_time = time.time()
        results = batch_optimizer.batch_query(
            get_analyses_by_ids,
            analysis_ids,
            batch_size=10
        )
        batch_time = time.time() - start_time
        
        print(f"   Batch query completed in {batch_time*1000:.2f} ms")
        print(f"   Retrieved {len(results)} records in batches")
    
    # Connection pool optimization
    print("\n3. Connection pool monitoring...")
    
    if db_manager.performance_optimizer:
        pool_optimizer = db_manager.performance_optimizer.pool_optimizer
        
        # Get current pool status
        status = pool_optimizer.get_pool_status()
        print(f"   Current pool status:")
        print(f"   - Size: {status['size']}")
        print(f"   - Checked out: {status['checked_out']}")
        print(f"   - Overflow: {status['overflow']}")
        
        # Get optimization recommendations
        recommendations = pool_optimizer.optimize_pool_size(target_concurrency=15)
        
        if recommendations['recommendations']:
            print("\n   Pool optimization recommendations:")
            for rec in recommendations['recommendations']:
                print(f"   - {rec['parameter']}: {rec['current']} → {rec['recommended']}")
                print(f"     Reason: {rec['reason']}")
    
    print("\n✓ Advanced features demo completed!")


if __name__ == "__main__":
    # Run demonstrations
    demonstrate_performance_features()
    demonstrate_advanced_features()
    
    # Show example of integrating with existing code
    print("\n\n=== Integration Example ===")
    print("""
    # In your existing code, simply enable performance optimization:
    
    db_manager = DatabaseManager(
        database_url=config.database.url,
        enable_performance_optimization=True,
        cache_size=1000,
        cache_ttl=300
    )
    
    # Use cached queries with decorator:
    @cached_query('analysis_by_serial', ttl=600)
    def get_analysis_by_serial(self, serial: str):
        with self.get_session() as session:
            return session.query(AnalysisResult).filter(
                AnalysisResult.serial == serial
            ).first()
    
    # Monitor performance:
    report = db_manager.get_performance_report()
    print(f"Cache hit rate: {report['caching']['hit_rate']}%")
    
    # Optimize when needed:
    if report['profiling']['slow_queries'] > 10:
        db_manager.optimize_performance()
    """)