import os
import urllib.parse
from dotenv import load_dotenv

load_dotenv()

basedir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

class Config:
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    @staticmethod
    def get_database_uri():
        db_type = os.getenv("DB_TYPE", "local").lower()
        
        if db_type == "local":
            return 'sqlite:///' + os.path.join(basedir, 'simulation.db')
        
        elif db_type == "azuresql":
            required_vars = ["SQL_SERVER", "SQL_DATABASE", "SQL_USER", "SQL_PASSWORD"]
            missing_vars = [var for var in required_vars if not os.getenv(var)]
            
            if missing_vars:
                raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
            
            params = urllib.parse.quote_plus(
                f"Driver={{{os.getenv('SQL_DRIVER', 'ODBC Driver 18 for SQL Server')}}};"
                f"Server={os.getenv('SQL_SERVER')};"
                f"Database={os.getenv('SQL_DATABASE')};"
                f"Uid={os.getenv('SQL_USER')};"
                f"Pwd={os.getenv('SQL_PASSWORD')};"
                f"Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;"
            )
            return f"mssql+pyodbc:///?odbc_connect={params}"
        
        else:
            raise ValueError(f"Unsupported DB_TYPE: {db_type}")
    
    SQLALCHEMY_DATABASE_URI = get_database_uri()

class DevelopmentConfig(Config):
    DEBUG = True

class ProductionConfig(Config):
    DEBUG = False