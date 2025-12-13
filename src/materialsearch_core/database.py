"""
数据库操作相关函数
"""
import datetime
import logging

from sqlalchemy import asc
from sqlalchemy.orm import Session

from materialsearch_core.models import Image, ImageFeatures, Video, VideoFeatures

logger = logging.getLogger(__name__)


def get_image_features_by_id(session: Session, image_id: int):
    """
    返回id对应的图片feature
    """
    features = session.query(ImageFeatures.features).join(Image, ImageFeatures.checksum == Image.checksum).filter(Image.id == image_id).first()
    if not features:
        logger.warning("用数据库的图来进行搜索，但id在数据库中不存在")
        return None
    return features[0]


def get_image_path_by_id(session: Session, id: int):
    """
    返回id对应的图片路径
    """
    path = session.query(Image.path).filter_by(id=id).first()
    if not path:
        return None
    return path[0]


def get_image_count(session: Session):
    """获取图片总数"""
    return session.query(Image).count()


def get_image_modify_time_and_hash(session: Session, path: str) -> tuple[datetime.datetime, str]:
    """
    获取图片的修改时间和hash
    :param session: Session, 数据库 session
    :param path: str, 图片路径
    :return: (datetime, str), 修改时间和hash
    """
    record = session.query(Image.modify_time, Image.checksum).filter_by(path=path).first()
    if not record:
        return None, None
    return record[0], record[1]


def get_video_modify_time_and_hash(session: Session, path: str) -> tuple[datetime.datetime, str]:
    """
    获取视频的修改时间和hash
    :param session: Session, 数据库 session
    :param path: str, 视频路径
    :return: (datetime, str), 修改时间和hash
    """
    record = session.query(Video.modify_time, Video.checksum).filter_by(path=path).first()
    if not record:
        return None, None
    return record[0], record[1]


def delete_image(session: Session, path: str):
    """
    判断图片是否修改，若修改则删除原来的记录
    :param session: Session, 数据库 session
    :param path: str, 图片路径
    :return: bool, 若文件未修改返回 True
    """
    record = session.query(Image).filter_by(path=path).first()
    session.delete(record)  # 删除图片记录
    session.commit()


def delete_video(session: Session, path: str):
    """
    判断视频是否修改，若修改则删除
    :param session: Session, 数据库 session
    :param path: str, 视频路径
    """
    record = session.query(Video).filter_by(path=path).first()
    session.delete(record)
    session.commit()
    return False


def get_video_paths(session: Session, filter_path: str = None, start_time: int = None, end_time: int = None):
    """获取所有视频的路径，支持通过路径和修改时间筛选"""
    query = session.query(Video.path).distinct()
    if filter_path:
        query = query.filter(Video.path.like("%" + filter_path + "%"))
    if start_time:
        query = query.filter(Video.modify_time >= datetime.datetime.fromtimestamp(start_time))
    if end_time:
        query = query.filter(Video.modify_time <= datetime.datetime.fromtimestamp(end_time))
    for (path,) in query:
        yield path


def get_frame_times_features_by_path(session: Session, path: str):
    """获取路径对应视频的features"""
    query = session.query(VideoFeatures.frame_time, VideoFeatures.features).join(Video, VideoFeatures.checksum == Video.checksum).filter(Video.path == path).order_by(VideoFeatures.frame_time).all()
    frame_times, features = zip(*query)
    return frame_times, features


def get_video_count(session: Session):
    """获取视频总数"""
    return session.query(Video.path).distinct().count()


def get_video_frame_count(session: Session):
    """获取视频帧总数"""
    return session.query(VideoFeatures).count()


def delete_video_by_path(session: Session, path: str):
    """删除路径对应的视频数据"""
    session.query(Video).filter_by(path=path).delete()
    session.commit()


def add_image(session: Session, path: str, modify_time: datetime.datetime, checksum: str, features: bytes):
    """添加图片到数据库"""
    logger.info(f"新增文件：{path}")
    image = Image(path=path, modify_time=modify_time, checksum=checksum)
    session.add(image)
    if session.query(ImageFeatures).filter_by(checksum=checksum).first():
        logger.warning(f"文件checksum重复，特征已存在，跳过添加特征：{path}")
        session.commit()
        return
    image_feature = ImageFeatures(checksum=checksum, features=features)
    session.add(image_feature)
    session.commit()


def check_duplicate_image(session: Session, path: str, modify_time: datetime.datetime, checksum: str, auto_add: bool = True) -> bool:
    """尝试添加重复图片到数据库，如果存在相同checksum，则添加image并返回true（不添加特征），否则返回false"""
    if session.query(Image).filter_by(checksum=checksum).first():
        if auto_add:
            logger.debug(f"新增重复文件：{path}")
            image = Image(path=path, modify_time=modify_time, checksum=checksum)
            session.add(image)
            session.commit()
        return True
    return False


def add_video(session: Session, path: str, modify_time: datetime.datetime, checksum: str, frame_time_features_generator):
    """
    将处理后的视频数据入库
    :param session: Session, 数据库session
    :param path: str, 视频路径
    :param modify_time: datetime, 文件修改时间
    :param checksum: str, 文件hash
    :param frame_time_features_generator: 返回(帧序列号,特征)元组的迭代器
    """
    # 使用 bulk_save_objects 一次性提交，因此处理至一半中断不会导致下次扫描时跳过
    logger.info(f"新增文件：{path}")
    video = Video(path=path, modify_time=modify_time, checksum=checksum)
    session.add(video)
    if session.query(VideoFeatures).filter_by(checksum=checksum).first():
        logger.warning(f"文件checksum重复，特征已存在，跳过添加特征：{path}")
        session.commit()
        return
    feature_rows = (
        VideoFeatures(checksum=checksum, frame_time=frame_time, features=features)
        for frame_time, features in frame_time_features_generator
    )
    session.bulk_save_objects(feature_rows)
    session.commit()


def check_duplicate_video(session: Session, path: str, modify_time: datetime.datetime, checksum: str, auto_add: bool = True) -> bool:
    """尝试添加重复视频到数据库，如果存在相同checksum，则添加video并返回true（不添加特征），否则返回false"""
    if session.query(Video).filter_by(checksum=checksum).first():
        if auto_add:
            logger.debug(f"新增重复文件：{path}")
            video = Video(path=path, modify_time=modify_time, checksum=checksum)
            session.add(video)
            session.commit()
        return True
    return False


def delete_record_if_not_exist(session: Session, assets: set):
    """
    删除不存在于 assets 集合中的图片 / 视频的数据库记录
    """
    for file in session.query(Image):
        if file.path not in assets:
            logger.info(f"文件已删除：{file.path}")
            session.delete(file)
    for path in session.query(Video.path).distinct():
        path = path[0]
        if path not in assets:
            logger.info(f"文件已删除：{path}")
            session.query(Video).filter_by(path=path).delete()
    session.commit()


def is_video_exist(session: Session, path: str):
    """判断视频是否存在"""
    video = session.query(Video).filter_by(path=path).first()
    if video:
        return True
    return False


def get_image_id_path_features(session: Session) -> tuple[list[int], list[str], list[bytes]]:
    """
    获取全部图片的 id, 路径, 特征，返回三个列表
    """
    rows = session.query(Image.id, Image.path, ImageFeatures.features).join(ImageFeatures, Image.checksum == ImageFeatures.checksum).all()
    if not rows:
        return [], [], []
    id_list, path_list, features_list = zip(*rows)
    return id_list, path_list, features_list


def get_image_id_path_features_filter_by_path_time(session: Session, path: str, start_time: int, end_time: int) -> tuple[
    list[int], list[str], list[bytes]]:
    """
    根据路径和时间，筛选出对应图片的 id, 路径, 特征，返回三个列表
    """
    query = session.query(Image.id, Image.path, ImageFeatures.features, Image.modify_time).join(ImageFeatures, Image.checksum == ImageFeatures.checksum)
    if start_time:
        query = query.filter(Image.modify_time >= datetime.datetime.fromtimestamp(start_time))
    if end_time:
        query = query.filter(Image.modify_time <= datetime.datetime.fromtimestamp(end_time))
    if path:
        query = query.filter(Image.path.like("%" + path + "%"))
    rows = query.all()
    if not rows:
        return [], [], []
    id_list, path_list, features_list, _ = zip(*rows)
    return list(id_list), list(path_list), list(features_list)


def search_image_by_path(session: Session, path: str):
    """
    根据路径搜索图片
    :return: (图片id, 图片路径) 元组列表
    """
    return (
        session.query(Image.id, Image.path)
        .filter(Image.path.like("%" + path + "%"))
        .order_by(asc(Image.path))
        .all()
    )


def search_video_by_path(session: Session, path: str):
    """
    根据路径搜索视频
    """
    return (
        session.query(Video.path)
        .distinct()
        .filter(Video.path.like("%" + path + "%"))
        .order_by(asc(Video.path))
        .all()
    )
