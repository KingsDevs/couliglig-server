from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from schema import Base
from dotenv import load_dotenv
import os

load_dotenv()


sqlite_path = os.getenv("SQLITE_DB_PATH", "./sqlite.db")

database_url = f"sqlite:///{sqlite_path}"


engine = create_engine(
    database_url,
    connect_args={"check_same_thread": False} 
)


Base.metadata.create_all(bind=engine)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False
)



def get_db_session() -> Session:
    """
    Establishes a connection to the SQLite database
    using environment variable SQLITE_DB_PATH.

    Returns:
        Session: A SQLAlchemy session connected to SQLite.
    """
    session = SessionLocal()
    return session