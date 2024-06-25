from sqlalchemy import JSON, create_engine, Column, Integer, String, Float, DateTime, ForeignKey, Boolean
from sqlalchemy.dialects.postgresql.base import UUID
from sqlalchemy.ext.declarative import as_declarative, declared_attr
from sqlalchemy.orm import declarative_base, relationship
from re import sub
from datetime import datetime
import uuid

Base = declarative_base()

# Function to convert CamelCase to snake_case
def snake_case(s):
    """
    e.g. "SnakeCase" -> "snake_case"
    e.g. "Snake-Case" -> "snake_case"
    e.g. "SNAKECase" -> "snake_case"
    e.g. "snakeCase" -> "snake_case"
    e.g. "SnakeCASE" -> "snake_case"
    """
    return "_".join(
        sub(
            "([A-Z][a-z]+)", r" \1", sub("([A-Z]+)", r" \1", s.replace("-", " "))
        ).split()
    ).lower()


# Base class for ORM models with common columns
@as_declarative()
class Base:
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime(timezone=True), default=datetime.now)
    updated_at = Column(
        DateTime(timezone=True), default=datetime.now, onupdate=datetime.now
    )
    is_active = Column(Boolean, default=True)
    deleted_at = Column(DateTime(timezone=True), default=None)

    __name__: str

    # Generate __tablename__ automatically from class name
    @declared_attr
    def __tablename__(cls) -> str:
        return snake_case(cls.__name__)

class Vehicle(Base):
    __tablename__ = 'vehicles'
    vehicle_type = Column(String)
    count = Column(Integer)
    timestamp = Column(DateTime)
    
    camera_id = Column(
        UUID(as_uuid=True), ForeignKey("cameras.id", ondelete="CASCADE"), nullable=False
    )
    
    camera = relationship("Camera", back_populates="vehicles")

    
class Camera(Base):
    __tablename__ = 'cameras'
    name = Column(String)
    latitude = Column(Float)
    longitude = Column(Float)
    area = Column(JSON)
    count_roi = Column(JSON)
    stream_url = Column(String)
    
    vehicles = relationship("Vehicle", back_populates="camera")
    traffic_density = relationship("TrafficDensity", back_populates="camera")

class TrafficDensity(Base):
    __tablename__ = 'traffic_density'
    timestamp = Column(DateTime)
    density_level = Column(Float)
    vehicle_count = Column(Integer)

    camera_id = Column(
        UUID(as_uuid=True), ForeignKey("cameras.id", ondelete="CASCADE"), nullable=False
    )
    
    camera = relationship("Camera", back_populates="traffic_density")
