"""
预处理图片和视频，建立索引，加快搜索速度
"""
import logging
import traceback

import cv2
import numpy as np
import requests
import torch.cuda
from PIL import Image
from tqdm import trange
from transformers import AutoModelForZeroShotImageClassification, AutoProcessor

from materialsearch_core.config import *

logger = logging.getLogger(__name__)

logger.info("Loading model...")
model = AutoModelForZeroShotImageClassification.from_pretrained(MODEL_NAME).to(DEVICE)
processor = AutoProcessor.from_pretrained(MODEL_NAME)
logger.info("Model loaded.")


def get_image_feature(images):
    """
    :param images: 图片列表
    :return: feature
    """
    if images is None or len(images) == 0:
        return None
    features = None
    try:
        inputs = processor(images=images, return_tensors="pt")["pixel_values"].to(DEVICE)
        features = model.get_image_features(inputs)
        normalized_features = features / torch.norm(features, dim=1, keepdim=True)  # 归一化，方便后续计算余弦相似度
        features = normalized_features.detach().cpu().numpy()
    except Exception as e:
        logger.exception("处理图片报错：type=%s error=%s" % (type(images), repr(e)))
        traceback.print_stack()
        if isinstance(images, list) and images:
            print("images[0]:", images[0])
        else:
            print("images:", images)
        if features is not None:
            print("feature.shape:", features.shape)
            print("feature:", features)
        # 如果报错内容包含 not enough GPU video memory，就打印额外的日志
        if "not enough GPU video memory" in repr(e) and MODEL_NAME != "OFA-Sys/chinese-clip-vit-base-patch16":
            logger.error("显存不足，请使用小模型（OFA-Sys/chinese-clip-vit-base-patch16）！！！")
    return features


def get_image_data(path: str, ignore_small_images: bool = True):
    """
    获取图片像素数据，如果出错返回 None
    :param path: string, 图片路径
    :param ignore_small_images: bool, 是否忽略尺寸过小的图片
    :return: <class 'numpy.nparray'>, 图片数据，如果出错返回 None
    """
    try:
        image = Image.open(path)
        if ignore_small_images:
            width, height = image.size
            if width < IMAGE_MIN_WIDTH or height < IMAGE_MIN_HEIGHT:
                return None
                # processor 中也会这样预处理 Image
        # 在这里提前转为 np.array 避免到时候抛出异常
        image = image.convert('RGB')
        image = np.array(image)
        return image
    except Exception as e:
        logger.exception("打开图片报错：path=%s error=%s" % (path, repr(e)))
        traceback.print_stack()
        return None


def process_image(path, ignore_small_images=True):
    """
    处理图片，返回图片特征
    :param path: string, 图片路径
    :param ignore_small_images: bool, 是否忽略尺寸过小的图片
    :return: <class 'numpy.nparray'>, 图片特征
    """
    image = get_image_data(path, ignore_small_images)
    if image is None:
        return None
    feature = get_image_feature(image)
    return feature


def process_images(path_list, ignore_small_images=True):
    """
    处理图片，返回图片特征
    :param path_list: string, 图片路径列表
    :param ignore_small_images: bool, 是否忽略尺寸过小的图片
    :return: <class 'numpy.nparray'>, 图片特征
    """
    images = []
    for path in path_list.copy():
        image = get_image_data(path, ignore_small_images)
        if image is None:
            path_list.remove(path)
            continue
        images.append(image)
    if not images:
        return None, None
    feature = get_image_feature(images)
    if torch.cuda.is_available() and LOW_CUDA_MEM:
        torch.cuda.empty_cache()
    return path_list, feature


def process_web_image(url):
    """
    处理网络图片，返回图片特征
    :param url: string, 图片URL
    :return: <class 'numpy.nparray'>, 图片特征
    """
    try:
        image = Image.open(requests.get(url, stream=True).raw)
    except Exception as e:
        logger.warning("获取图片报错：%s %s" % (url, repr(e)))
        return None
    feature = get_image_feature(image)
    return feature


def get_frames(video: cv2.VideoCapture):
    """ 
    获取视频的帧数据
    :return: (list[int], list[array]) (帧编号列表, 帧像素数据列表) 元组
    """
    frame_rate = round(video.get(cv2.CAP_PROP_FPS))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.debug(f"fps: {frame_rate} total: {total_frames}")
    ids, frames = [], []
    for current_frame in trange(
            0, total_frames, FRAME_INTERVAL * frame_rate, desc="当前进度", unit="frame"
    ):
        # 在 FRAME_INTERVAL 为 2（默认值），frame_rate 为 24
        # 即 FRAME_INTERVAL * frame_rate == 48 时测试
        # 直接设置当前帧的运行效率低于使用 grab 跳帧
        # 如果需要跳的帧足够多，也许直接设置效率更高
        # video.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = video.read()
        if not ret:
            break
        ids.append(current_frame // frame_rate)
        frames.append(frame)
        if len(frames) == SCAN_PROCESS_BATCH_SIZE:
            yield ids, frames
            ids = []
            frames = []
        for _ in range(FRAME_INTERVAL * frame_rate - 1):
            video.grab()  # 跳帧
    yield ids, frames


def process_video(path):
    """
    处理视频并返回处理完成的数据
    返回一个生成器，每调用一次则返回视频下一个帧的数据
    :param path: string, 视频路径
    :return: [int, <class 'numpy.nparray'>], [当前是第几帧（被采集的才算），图片特征]
    """
    logger.info(f"处理视频中：{path}")
    video = None
    try:
        video = cv2.VideoCapture(path)
        for ids, frames in get_frames(video):
            if not frames:
                continue
            features = get_image_feature(frames)
            if features is None:
                logger.warning("features is None in process_video")
                continue
            for id, feature in zip(ids, features):
                yield id, feature
    except Exception as e:
        logger.exception("处理视频报错：path=%s error=%s" % (path, repr(e)))
        traceback.print_stack()
        if video is not None:
            frame_rate = round(video.get(cv2.CAP_PROP_FPS))
            total_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
            print(f"fps: {frame_rate} total: {total_frames}")
            video.release()
        return
    finally:
        if torch.cuda.is_available() and LOW_CUDA_MEM:
            torch.cuda.empty_cache()


def process_text(input_text):
    """
    预处理文字，返回文字特征
    :param input_text: string, 被处理的字符串
    :return: <class 'numpy.nparray'>,  文字特征
    """
    feature = None
    if not input_text:
        return None
    try:
        text = processor(text=input_text, return_tensors="pt", padding=True)["input_ids"].to(DEVICE)
        feature = model.get_text_features(text)
        normalize_feature = feature / torch.norm(feature, dim=1, keepdim=True)  # 归一化，方便后续计算余弦相似度
        feature = normalize_feature.detach().cpu().numpy()
    except Exception as e:
        logger.exception("处理文字报错：text=%s error=%s" % (input_text, repr(e)))
        traceback.print_stack()
        if feature is not None:
            print("feature.shape:", feature.shape)
            print("feature:", feature)
    return feature


def match_text_and_image(text_feature, image_feature):
    """
    匹配文字和图片，返回余弦相似度
    :param text_feature: <class 'numpy.nparray'>, 文字特征
    :param image_feature: <class 'numpy.nparray'>, 图片特征
    :return: <class 'numpy.nparray'>, 文字和图片的余弦相似度，shape=(1, 1)
    """
    score = image_feature @ text_feature.T
    return score


# 以下为FAISS测试代码
# def cache_faiss_index(func):
#     """
#     装饰器：缓存FAISS索引
#     """
#     cache = {}
#     stats = {'hits': 0, 'misses': 0}
#
#     @wraps(func)
#     def wrapper(features, *args, **kwargs):
#         # 生成缓存键（使用特征的shape和部分数据）
#         key = f"{features.shape}_{features[0][0]}_{features[-1][-1]}_{features.mean()}"
#
#         if key in cache:
#             stats['hits'] += 1
#             print("FAISS index cache hit")
#             return cache[key]
#
#         stats['misses'] += 1
#         result = func(features, *args, **kwargs)
#         cache[key] = result
#
#         # 限制缓存大小
#         if len(cache) > 10:
#             # 移除第一个键
#             first_key = next(iter(cache))
#             del cache[first_key]
#
#         return result
#
#     wrapper.cache_info = lambda: stats
#     wrapper.clear_cache = lambda: cache.clear()
#
#     return wrapper
#
#
# @cache_faiss_index
# def get_faiss_index(image_features):
#     image_features_f32 = image_features.astype(np.float32)
#     d = image_features_f32.shape[1]
#     index = faiss.IndexFlatIP(d)
#     index.add(image_features_f32)
#     return index

# @cache_faiss_index
# def get_faiss_index_pq(image_features):
#     image_features_f32 = image_features.astype(np.float32)
#     d = image_features_f32.shape[1]
#
#     # 设置PQ参数
#     m = 8  # 子空间数量（通常设置为d的约1/4）
#     nbits = 8  # 每个子空间的比特数（精度越低速度越快）
#
#     quantizer = faiss.IndexFlatIP(d)
#     index = faiss.IndexIVFPQ(quantizer, d, 100, m, nbits)  # 100个聚类中心
#
#     # 需要训练
#     index.train(image_features_f32)
#     index.add(image_features_f32)
#
#     return index
#
#
# @cache_faiss_index
# def get_faiss_index_ivf(image_features):
#     image_features_f32 = image_features.astype(np.float32)
#     d = image_features_f32.shape[1]
#     nlist = min(100, image_features_f32.shape[0] // 40)  # 聚类数量
#
#     quantizer = faiss.IndexFlatIP(d)
#     index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
#
#     # 需要训练
#     index.train(image_features_f32)
#     index.add(image_features_f32)
#
#     # 搜索时控制搜索的聚类数量（值越小越快，精度越低）
#     index.nprobe = 5  # 默认搜索5个最近的聚类
#     return index
#
#
# @cache_faiss_index
# def get_faiss_index_hnsw(image_features):
#     image_features_f32 = image_features.astype(np.float32)
#     d = image_features_f32.shape[1]
#
#     M = 16  # 每个节点的连接数（越大越准越慢）
#     index = faiss.IndexHNSWFlat(d, M, faiss.METRIC_INNER_PRODUCT)
#     index.hnsw.efConstruction = 40  # 构建时的邻居数量
#     index.hnsw.efSearch = 16  # 搜索时的邻居数量（越小越快）
#
#     index.add(image_features_f32)
#     return index


def match_batch(
        positive_feature,
        negative_feature,
        image_features,
        positive_threshold,
        negative_threshold,
):
    """
    匹配image_feature列表并返回余弦相似度
    :param positive_feature: <class 'numpy.ndarray'>, 正向提示词特征，shape=(1, m)
    :param negative_feature: <class 'numpy.ndarray'>, 反向提示词特征，shape=(1, m)
    :param image_features: <class 'numpy.ndarray'>, 图片特征，shape=(n, m)
    :param positive_threshold: int/float, 正向提示分数阈值，高于此分数才显示
    :param negative_threshold: int/float, 反向提示分数阈值，低于此分数才显示
    :return: <class 'numpy.nparray'>, 提示词和每个图片余弦相似度列表，shape=(n, )，如果小于正向提示分数阈值或大于反向提示分数阈值则会置0
    TODO: FAISS（试了一下，两万多张图没有任何效果。不知道数十万效果如何。暂时无法测试。） 记得Mac需要安装1.7.0版本，否则会 segmentation fault
    """
    # 把feature都reshape成(1, m)的形状，方便矩阵运算，兼容不同版本造成的shape不一致的问题
    if positive_feature is not None:
        positive_feature = np.asarray(positive_feature).reshape(1, -1)
    if negative_feature is not None:
        negative_feature = np.asarray(negative_feature).reshape(1, -1)

    # FAISS 实现
    # n_vectors = len(image_features)
    # index = get_faiss_index(image_features)
    # # 计算score
    # scores = np.ones(n_vectors, dtype=np.float32)
    # if positive_feature is not None:
    #     positive_feature_f32 = np.asarray(positive_feature, dtype=np.float32).reshape(1, -1)
    #     positive_scores, _ = index.search(positive_feature_f32, n_vectors)
    #     scores = positive_scores.flatten()
    # # 根据阈值进行过滤
    # pos_thresh = positive_threshold / 100
    # scores = np.where(scores < pos_thresh, 0, scores)
    # if negative_feature is not None:
    #     negative_feature_f32 = np.asarray(negative_feature, dtype=np.float32).reshape(1, -1)
    #     negative_scores, _ = index.search(negative_feature_f32, n_vectors)
    #     negative_scores = negative_scores.flatten()
    #     neg_thresh = negative_threshold / 100
    #     scores = np.where(negative_scores > neg_thresh, 0, scores)
    # return scores

    # 计算score
    if positive_feature is None:  # 没有正向feature就把分数全部设成1
        positive_scores = np.ones((len(image_features), 1))
    else:
        positive_scores = image_features @ positive_feature.T
    if negative_feature is not None:
        negative_scores = image_features @ negative_feature.T
    # 根据阈值进行过滤
    scores = np.where(positive_scores < positive_threshold / 100, 0, positive_scores)
    if negative_feature is not None:
        scores = np.where(negative_scores > negative_threshold / 100, 0, scores)
    return scores.reshape(-1)
