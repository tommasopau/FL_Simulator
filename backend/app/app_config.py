import os
import urllib.parse
from dotenv import load_dotenv

load_dotenv()  # Load variables from .env file

basedir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

# Get credentials from environment variables
SQL_DRIVER = os.getenv("SQL_DRIVER", "ODBC Driver 18 for SQL Server")
SQL_SERVER = os.getenv("SQL_SERVER", "tcp:fl-simulations.database.windows.net,1433")
SQL_DATABASE = os.getenv("SQL_DATABASE", "FL_simulations")
SQL_USER = os.getenv("SQL_USER", "tpaus")
SQL_PASSWORD = os.getenv("SQL_PASSWORD", "Tommaso2001")

params = urllib.parse.quote_plus(
    f"Driver={{{SQL_DRIVER}}};Server={SQL_SERVER};Database={SQL_DATABASE};Uid={SQL_USER};Pwd={SQL_PASSWORD};Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;"
)

class Config:
    SQLALCHEMY_DATABASE_URI = "mssql+pyodbc:///?odbc_connect=%s" % params
    #SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(basedir, 'simulation.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False

# You can also define additional configuration classes for other environments
class DevelopmentConfig(Config):
    DEBUG = True

class ProductionConfig(Config):
    DEBUG = False