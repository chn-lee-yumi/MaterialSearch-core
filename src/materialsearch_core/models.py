"""
定义数据库模型
"""
import os

from sqlalchemy import BINARY, Column, DateTime, Integer, String, ForeignKey
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from materialsearch_core.config import SQLALCHEMY_DATABASE_URL

# 数据库目录不存在的时候自动创建目录。TODO：如果是mysql之类的数据库，这里的代码估计是不兼容的
folder_path = os.path.dirname(SQLALCHEMY_DATABASE_URL.replace("sqlite:///", ""))
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# 本地扫描数据库
BaseModel = declarative_base()
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False}
)
DatabaseSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def create_tables():
    """
    创建数据库表
    """
    BaseModel.metadata.create_all(bind=engine)


class Image(BaseModel):
    """图片表"""
    __tablename__ = "image"
    id = Column(Integer, primary_key=True, index=True)
    path = Column(String(4096), index=True)  # 文件路径
    modify_time = Column(DateTime, index=True)  # 文件修改时间
    checksum = Column(String(40), index=True)  # 文件SHA1


class ImageFeatures(BaseModel):
    """图片特征表"""
    __tablename__ = "image_features"
    id = Column(Integer, primary_key=True, index=True)
    checksum = Column(String(40), ForeignKey("image.checksum", ondelete="CASCADE"), index=True)  # 文件SHA1
    features = Column(BINARY)  # 文件预处理后的二进制数据


class Video(BaseModel):
    """视频表"""
    __tablename__ = "video"
    id = Column(Integer, primary_key=True, index=True)
    path = Column(String(4096), index=True)  # 文件路径
    modify_time = Column(DateTime, index=True)  # 文件修改时间
    checksum = Column(String(40), index=True)  # 文件SHA1


class VideoFeatures(BaseModel):
    """视频特征表"""
    __tablename__ = "video_features"
    id = Column(Integer, primary_key=True, index=True)
    checksum = Column(String(40), ForeignKey("video.checksum", ondelete="CASCADE"), index=True)  # 文件SHA1
    frame_time = Column(Integer)  # 这一帧所在的时间
    features = Column(BINARY)  # 文件预处理后的二进制数据
