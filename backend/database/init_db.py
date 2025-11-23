"""
Database Initialization Script
Creates the SQLite database and all required tables
"""

import sqlite3
import os
from pathlib import Path

# Get database path from environment or use default
DATABASE_PATH = os.getenv('DATABASE_PATH', 'disaster_data.db')

def init_database():
    """Initialize the SQLite database with schema"""
    
    # Get the directory of this script
    script_dir = Path(__file__).parent
    schema_file = script_dir / 'schema.sql'
    
    # Check if schema file exists
    if not schema_file.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_file}")
    
    # Read schema SQL
    with open(schema_file, 'r') as f:
        schema_sql = f.read()
    
    # Connect to database (creates file if it doesn't exist)
    print(f"Initializing database at: {DATABASE_PATH}")
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    try:
        # Execute schema SQL
        cursor.executescript(schema_sql)
        conn.commit()
        print("✓ Database schema created successfully")
        
        # Verify tables were created
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        print(f"\n✓ Created {len(tables)} tables:")
        for table in tables:
            print(f"  - {table[0]}")
        
        # Get database file size
        db_size = os.path.getsize(DATABASE_PATH)
        print(f"\n✓ Database file size: {db_size} bytes")
        print(f"✓ Database location: {os.path.abspath(DATABASE_PATH)}")
        
        return True
        
    except sqlite3.Error as e:
        print(f"✗ Error initializing database: {e}")
        return False
        
    finally:
        conn.close()

def verify_database():
    """Verify database structure"""
    
    if not os.path.exists(DATABASE_PATH):
        print(f"✗ Database file not found: {DATABASE_PATH}")
        return False
    
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    try:
        # Check each expected table
        expected_tables = [
            'disasters_historical',
            'weather_historical',
            'features_training',
            'predictions_log',
            'users',
            'alerts_history'
        ]
        
        print("\nVerifying database structure...")
        all_exist = True
        
        for table_name in expected_tables:
            cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}';")
            result = cursor.fetchone()
            
            if result:
                # Get column count
                cursor.execute(f"PRAGMA table_info({table_name});")
                columns = cursor.fetchall()
                print(f"  ✓ {table_name}: {len(columns)} columns")
            else:
                print(f"  ✗ {table_name}: NOT FOUND")
                all_exist = False
        
        return all_exist
        
    except sqlite3.Error as e:
        print(f"✗ Error verifying database: {e}")
        return False
        
    finally:
        conn.close()

if __name__ == "__main__":
    print("=" * 60)
    print("Disaster Early Warning System - Database Initialization")
    print("=" * 60)
    print()
    
    # Initialize database
    success = init_database()
    
    if success:
        print()
        # Verify database
        verify_success = verify_database()
        
        if verify_success:
            print()
            print("=" * 60)
            print("✓ Database initialization completed successfully!")
            print("=" * 60)
        else:
            print()
            print("=" * 60)
            print("✗ Database verification failed!")
            print("=" * 60)
    else:
        print()
        print("=" * 60)
        print("✗ Database initialization failed!")
        print("=" * 60)
