from sqlalchemy import create_engine
from sqlalchemy import event
from sqlalchemy.orm import sessionmaker, Session
from schema import Base
from dotenv import load_dotenv
import os

load_dotenv()

os.makedirs("./db", exist_ok=True)

sqlite_path =  "./db/sqlite.db"
database_url = f"sqlite:///{sqlite_path}"


engine = create_engine(
    database_url,
    connect_args={"check_same_thread": False} 
)

@event.listens_for(engine, "connect")
def enable_sqlite_wal(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.close()


Base.metadata.create_all(bind=engine)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False
)

def get_db_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()