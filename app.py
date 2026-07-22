#!/usr/bin/env python3
"""数据集处理Demo - 支持PDF全文检索、视频逐帧像素查看与混合检索(关键词+FAISS向量)"""

import os
import sys
import json
import io
import re
import hashlib
import pickle
import time
import traceback
import warnings
import urllib.parse
from pathlib import Path
from collections import defaultdict

from flask import Flask, render_template, request, jsonify, send_file, url_for
import numpy as np
from PIL import Image
import jieba

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get('DATA_DIR', BASE_DIR / 'data'))
CACHE_DIR = BASE_DIR / 'cache'
CACHE_DIR.mkdir(parents=True, exist_ok=True)

VIDEO_EXTENSIONS = {'.mp4', '.wmv', '.avi', '.mov', '.mkv', '.webm', '.flv', '.m4v', '.mpg', '.mpeg'}
PDF_EXTENSIONS = {'.pdf'}
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.tif', '.webp'}
TEXT_EXTENSIONS = {'.txt', '.md', '.csv', '.json', '.xml', '.yaml', '.yml', '.log'}
ARCHIVE_EXTENSIONS = {'.zip', '.rar', '.7z', '.tar', '.gz'}
SKIP_NAMES = {'desktop.ini', 'thumbs.db', '.ds_store', '.gitkeep'}

# ============================================================================
# 可选依赖
# ============================================================================
fitz = None
cv2 = None
faiss = None
torch = None
transformers = None

try:
    import fitz as _fitz
    fitz = _fitz
except ImportError:
    print('[WARN] PyMuPDF 未安装, PDF文本提取不可用')

try:
    import cv2 as _cv2
    cv2 = _cv2
except ImportError:
    print('[WARN] opencv-python 未安装, 视频处理不可用')

try:
    import faiss as _faiss
    faiss = _faiss
except ImportError:
    print('[WARN] faiss 未安装, 向量检索不可用')

# 国内环境使用hf-mirror下载模型（需在transformers import前设置）
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')

try:
    import torch as _torch
    torch = _torch
except ImportError:
    print('[WARN] torch 未安装, 向量检索不可用')

try:
    import transformers as _transformers
    transformers = _transformers
except ImportError:
    print('[WARN] transformers 未安装, 向量检索不可用')

with warnings.catch_warnings():
    warnings.simplefilter('ignore')

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

# ============================================================================
# 全局状态
# ============================================================================
search_index = {'docs': [], 'inverted': {}, 'files': {}, 'ready': False}
video_meta_cache = {}
_dataset_tree = None
_dataset_flat_files = []

# FAISS 向量检索
EMBED_DIM = 512  # Chinese-CLIP base
_faiss_index = None          # faiss.Index
_faiss_metadata = []         # [(doc_id, type, extra), ...]
_faiss_ready = False
_embed_model = None
_embed_processor = None
_embed_device = None

# 用于向量->关键词搜索的doc_id映射
_vector_doc_map = {}         # faiss_entry_id -> {search_idx, type}

# ============================================================================
# 工具函数
# ============================================================================

def normalize_path(p):
    return str(Path(p)).replace('\\', '/')


def classify_file(suffix):
    s = suffix.lower()
    if s in PDF_EXTENSIONS:
        return 'pdf'
    if s in VIDEO_EXTENSIONS:
        return 'video'
    if s in IMAGE_EXTENSIONS:
        return 'image'
    if s in TEXT_EXTENSIONS:
        return 'text'
    if s in ARCHIVE_EXTENSIONS:
        return 'archive'
    return 'other'


# ============================================================================
# Embedding 模型（Chinese-CLIP，图文多模态）
# ============================================================================

def init_embedding_model():
    """懒加载Chinese-CLIP模型"""
    global _embed_model, _embed_processor, _embed_device
    if _embed_model is not None:
        return True
    if torch is None or transformers is None:
        print('[WARN] torch/transformers 未安装, 无法加载embedding模型')
        return False
    try:
        model_name = 'OFA-Sys/chinese-clip-vit-base-patch16'
        print(f'[INFO] 加载 Embedding 模型: {model_name} ...')
        _embed_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if _embed_device == 'cuda':
            print('[INFO] 使用 GPU 加速')
        else:
            print('[INFO] 使用 CPU')

        _embed_processor = transformers.ChineseCLIPProcessor.from_pretrained(model_name, trust_remote_code=True)
        _embed_model = transformers.ChineseCLIPModel.from_pretrained(model_name, trust_remote_code=True)
        _embed_model = _embed_model.to(_embed_device)
        _embed_model.eval()
        print(f'[INFO] Embedding 模型加载完成 (device={_embed_device})')
        return True
    except Exception as e:
        print(f'[ERROR] 加载 Embedding 模型失败: {e}')
        traceback.print_exc()
        return False


@torch.no_grad()
def embed_text(text):
    """文本 -> 向量（手动mean pooling兼容新版transformers）"""
    if _embed_model is None:
        if not init_embedding_model():
            return None
    try:
        inputs = _embed_processor(text=[text], return_tensors='pt', padding=True, max_length=128, truncation=True)
        inputs = {k: v.to(_embed_device) for k, v in inputs.items()}
        # Chinese-CLIP的text_model没有pooler_output，需手动mean pooling
        outputs = _embed_model.text_model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        mask = inputs['attention_mask'].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
        pooled = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1)
        features = _embed_model.text_projection(pooled)
        vec = features.cpu().numpy().flatten().astype('float32')
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec
    except Exception as e:
        print(f'[ERROR] embed_text 失败: {e}')
        traceback.print_exc()
        return None


@torch.no_grad()
def embed_image(pil_image):
    """PIL图像 -> 向量"""
    if _embed_model is None:
        if not init_embedding_model():
            return None
    try:
        inputs = _embed_processor(images=pil_image, return_tensors='pt')
        inputs = {k: v.to(_embed_device) for k, v in inputs.items()}
        features = _embed_model.get_image_features(**inputs)
        vec = features.cpu().numpy().flatten().astype('float32')
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec
    except Exception as e:
        print(f'[ERROR] embed_image 失败: {e}')
        return None


# ============================================================================
# FAISS 向量索引
# ============================================================================

def build_vector_index(force=False):
    """构建 FAISS 向量索引（图文多模态），覆盖所有PDF页面和视频关键帧"""
    global _faiss_index, _faiss_metadata, _faiss_ready, _vector_doc_map

    cache_vec = CACHE_DIR / 'faiss_index.bin'
    cache_meta = CACHE_DIR / 'faiss_meta.pkl'
    cache_hash = CACHE_DIR / 'faiss_hash.txt'

    if faiss is None or torch is None or transformers is None:
        print('[WARN] FAISS/torch/transformers 缺失, 跳过向量索引')
        return

    # 计算哈希
    hasher = hashlib.md5()
    for f in _dataset_flat_files:
        hasher.update(f['name'].encode('utf-8'))
        hasher.update(str(f['size']).encode('utf-8'))
    current_hash = hasher.hexdigest()

    # 尝试加载缓存
    if not force and cache_vec.exists() and cache_meta.exists() and cache_hash.exists():
        try:
            cached_h = open(cache_hash, 'r').read().strip()
            if cached_h == current_hash:
                _faiss_index = faiss.read_index(str(cache_vec))
                with open(cache_meta, 'rb') as f:
                    _faiss_metadata = pickle.load(f)
                _faiss_ready = True
                print(f'[INFO] 从缓存加载 FAISS 索引 ({_faiss_index.ntotal} 个向量)')
                return
        except Exception:
            pass

    # 加载embedding模型
    if not init_embedding_model():
        print('[ERROR] 无法加载 embedding 模型, 跳过向量索引')
        return

    print('[INFO] 构建 FAISS 向量索引...')
    dim = EMBED_DIM
    _faiss_index = faiss.IndexFlatIP(dim)  # 内积 = 余弦相似度(已归一化)
    _faiss_metadata = []
    _vector_doc_map = {}
    total = 0

    # 1) 索引所有PDF页面（文字型用文本编码，扫描型渲染成图片用图像编码，真正多模态）
    for fi, finfo in enumerate(_dataset_flat_files):
        if finfo['category'] != 'pdf':
            continue
        filepath = DATA_DIR / finfo['path']
        rel_path = finfo['path']
        pages = extract_pdf_text(str(filepath))

        for page_num, text in pages:
            if not text or len(text.strip()) < 5:
                continue
            # 检测是否为扫描版（文字提取是乱码）
            if is_garbled_text(text):
                # 扫描版：渲染页面为图片 → 用CLIP图像编码器
                buf = render_pdf_page_as_image(str(filepath), page_num, scale=1.0)
                if buf:
                    img = Image.open(buf)
                    vec = embed_image(img)
                else:
                    vec = None
                text_preview = '(扫描页面，已通过图像编码索引)'
                is_scanned = True
            else:
                # 文字版：直接用CLIP文本编码器
                vec = embed_text(text[:1024])
                text_preview = text[:300]
                is_scanned = False

            if vec is None:
                continue
            vec = vec.reshape(1, -1).astype('float32')
            _faiss_index.add(vec)
            entry_id = _faiss_index.ntotal - 1
            _faiss_metadata.append({
                'type': 'pdf',
                'file': rel_path,
                'filename': finfo['name'],
                'page': page_num,
                'text_preview': text_preview,
                'is_scanned': is_scanned,
                'score': 0.0,
            })
            _vector_doc_map[entry_id] = {'idx': fi, 'type': 'pdf', 'page': page_num}
            total += 1
            if total % 200 == 0:
                print(f'  [VEC] 已索引 {total} 个PDF页面...')

    # 2) 索引视频关键帧
    for fi, finfo in enumerate(_dataset_flat_files):
        if finfo['category'] != 'video':
            continue
        filepath = DATA_DIR / finfo['path']
        rel_path = finfo['path']
        info = video_meta_cache.get(rel_path)
        if not info:
            continue
        total_frames = info['total_frames']
        fps = info['fps']

        # 采样：最多80帧（覆盖整个视频）
        max_samples = 80
        if total_frames <= max_samples:
            sample_frames = list(range(total_frames))
        else:
            step = max(1, total_frames // max_samples)
            sample_frames = list(range(0, total_frames, step))[:max_samples]

        for frame_num in sample_frames:
            img = extract_frame(str(filepath), frame_num)
            if img is None:
                continue
            vec = embed_image(img)
            if vec is None:
                continue
            vec = vec.reshape(1, -1).astype('float32')
            _faiss_index.add(vec)
            entry_id = _faiss_index.ntotal - 1
            timestamp = round(frame_num / fps, 2) if fps > 0 else 0
            _faiss_metadata.append({
                'type': 'video',
                'file': rel_path,
                'filename': finfo['name'],
                'frame': frame_num,
                'timestamp': timestamp,
                'total_frames': total_frames,
                'fps': fps,
                'score': 0.0,
            })
            _vector_doc_map[entry_id] = {'idx': fi, 'type': 'video', 'frame': frame_num}
            total += 1
            print(f'  [VEC] 索引视频帧: {finfo["name"]} frame={frame_num}')

    _faiss_ready = True
    print(f'[INFO] FAISS 索引完成: {total} 个向量 (dim={dim})')

    # 保存缓存
    try:
        faiss.write_index(_faiss_index, str(cache_vec))
        with open(cache_meta, 'wb') as f:
            pickle.dump(_faiss_metadata, f, pickle.HIGHEST_PROTOCOL)
        with open(cache_hash, 'w') as f:
            f.write(current_hash)
        print(f'[INFO] FAISS 索引已缓存')
    except Exception as e:
        print(f'[WARN] FAISS 缓存写入失败: {e}')


def vector_search(query, k=100):
    """向量检索：query -> embedding -> FAISS 最近邻搜索"""
    if not _faiss_ready or _faiss_index is None or _faiss_index.ntotal == 0:
        return [], '向量索引不可用'

    query_vec = embed_text(query)
    if query_vec is None:
        return [], '文本嵌入失败'

    query_vec = query_vec.reshape(1, -1).astype('float32')
    similarities, indices = _faiss_index.search(query_vec, min(k, _faiss_index.ntotal))

    results = []
    for i in range(len(indices[0])):
        idx = int(indices[0][i])
        sim = float(similarities[0][i])
        if idx < 0 or idx >= len(_faiss_metadata):
            continue
        meta = dict(_faiss_metadata[idx])
        meta['score'] = round(sim, 4)
        results.append(meta)

    return results, None


# ============================================================================
# 数据集扫描
# ============================================================================

def scan_dataset(data_dir=None):
    if data_dir is None:
        data_dir = DATA_DIR
    root = Path(data_dir)
    if not root.exists():
        return None, []

    flat = []

    def walk(path, depth=0):
        if depth > 10:
            return None
        name = path.name or path.root or str(path)
        node = {
            'name': name,
            'path': normalize_path(path.relative_to(root)) if path != root else '',
            'type': 'dir', 'depth': depth, 'children': [],
            'file_count': 0, 'total_size': 0, 'categories': defaultdict(int),
        }
        try:
            entries = sorted(path.iterdir(), key=lambda e: (e.is_file(), e.name.lower()))
        except (PermissionError, OSError):
            return node
        for entry in entries:
            if entry.name.lower() in SKIP_NAMES:
                continue
            if entry.is_file():
                cat = classify_file(entry.suffix)
                size = entry.stat().st_size
                fi = {
                    'name': entry.name, 'path': normalize_path(entry.relative_to(root)),
                    'size': size, 'size_mb': round(size / (1024 * 1024), 2),
                    'category': cat, 'suffix': entry.suffix.lower(),
                    'parent_dir': normalize_path(path.relative_to(root)) if path != root else '',
                }
                if cat == 'pdf' and fitz:
                    fi['pages'] = get_pdf_page_count(str(entry))
                if cat == 'video':
                    cached = video_meta_cache.get(fi['path'])
                    if cached:
                        fi['video_info'] = cached
                node['children'].append({'name': entry.name, 'path': fi['path'], 'type': 'file', 'category': cat, **fi})
                node['file_count'] += 1
                node['total_size'] += size
                node['categories'][cat] += 1
                flat.append(fi)
            elif entry.is_dir():
                child = walk(entry, depth + 1)
                if child:
                    node['children'].append(child)
                    node['file_count'] += child['file_count']
                    node['total_size'] += child['total_size']
                    for k, v in child['categories'].items():
                        node['categories'][k] += v
        node['categories'] = dict(node['categories'])
        node['total_size_mb'] = round(node['total_size'] / (1024 * 1024), 2)
        return node

    tree = walk(root)
    return tree, flat


# ============================================================================
# PDF 处理
# ============================================================================

def extract_pdf_text(filepath):
    if fitz is None:
        return []
    results = []
    try:
        doc = fitz.open(filepath)
        for i, page in enumerate(doc):
            try:
                text = page.get_text('text')
                if text and text.strip():
                    results.append((i + 1, text.strip()))
            except Exception:
                continue
        doc.close()
    except Exception as e:
        print(f'[ERROR] 读取PDF失败 {filepath}: {e}')
    return results


def get_pdf_page_count(filepath):
    if fitz is None:
        return 0
    try:
        doc = fitz.open(filepath)
        count = len(doc)
        doc.close()
        return count
    except Exception:
        return 0


# ============================================================================
# 视频处理
# ============================================================================

def get_video_info(filepath):
    if cv2 is None:
        return None
    try:
        cap = cv2.VideoCapture(str(filepath))
        if not cap.isOpened():
            cap.release()
            return None
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0
        cap.release()
        return {'fps': round(fps, 2), 'total_frames': total_frames,
                'duration': round(duration, 2), 'width': width, 'height': height}
    except Exception as e:
        print(f'[ERROR] 视频信息读取失败 {filepath}: {e}')
        return None


def extract_frame(filepath, frame_num):
    if cv2 is None:
        return None
    try:
        cap = cv2.VideoCapture(str(filepath))
        if not cap.isOpened():
            cap.release()
            return None
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_num < 0 or frame_num >= total:
            cap.release()
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return None
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    except Exception as e:
        print(f'[ERROR] 提取帧失败 {filepath} frame={frame_num}: {e}')
        return None


# ============================================================================
# 关键词搜索引擎
# ============================================================================

def tokenize(text):
    words = jieba.lcut(text.lower())
    result = []
    for w in words:
        w = w.strip()
        if not w:
            continue
        if len(w) <= 1 and re.match(r'^[\s\W_]+$', w):
            continue
        # 单字中文保留（已由前面的条件过滤了纯标点）
        result.append(w)
    return list(set(result))


def build_index(force=False):
    global search_index
    cache_file = CACHE_DIR / 'search_index.pkl'
    cache_hash_file = CACHE_DIR / 'search_index.hash'

    pdf_files = [f for f in _dataset_flat_files if f['category'] == 'pdf']
    if not pdf_files:
        search_index['ready'] = True
        print('[INFO] 未找到PDF文件, 跳过关键词索引')
        return

    cur_hash = hashlib.md5()
    for f in pdf_files:
        cur_hash.update(f['name'].encode('utf-8'))
        cur_hash.update(str(f['size']).encode('utf-8'))
    cur_hash = cur_hash.hexdigest()

    if not force and cache_file.exists() and cache_hash_file.exists():
        try:
            if open(cache_hash_file, 'r').read().strip() == cur_hash:
                with open(cache_file, 'rb') as f:
                    search_index = pickle.load(f)
                print(f'[INFO] 从缓存加载关键词索引 ({len(search_index["docs"])} 个文档)')
                search_index['ready'] = True
                return
        except Exception:
            pass

    print(f'[INFO] 构建关键词索引 ({len(pdf_files)} 个PDF)...')
    docs, inverted, finfo = [], defaultdict(list), {}
    for fi in pdf_files:
        rel = fi['path']
        pages = extract_pdf_text(str(DATA_DIR / rel))
        finfo[rel] = {'pages': len(pages), 'size': fi['size'], 'size_mb': fi['size_mb'], 'parent_dir': fi['parent_dir']}
        for pn, text in pages:
            did = len(docs)
            docs.append({'id': did, 'file': rel, 'filename': fi['name'], 'page': pn, 'text': text, 'text_preview': text[:200]})
            for t in tokenize(text):
                inverted[t].append(did)

    search_index = {'docs': docs, 'inverted': dict(inverted), 'files': finfo, 'ready': True}
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(search_index, f, pickle.HIGHEST_PROTOCOL)
        with open(cache_hash_file, 'w') as f:
            f.write(cur_hash)
    except Exception:
        pass
    print(f'[INFO] 关键词索引完成: {len(docs)} 文档, {len(inverted)} 词条')


def keyword_search(query, limit=50):
    """纯关键词搜索"""
    if not search_index['ready']:
        return [], '关键词索引未就绪'
    if not query or not query.strip():
        return [], '请输入搜索关键词'
    terms = tokenize(query)
    if not terms:
        return [], '未能解析出有效搜索词'
    scores = defaultdict(float)
    for t in terms:
        for did in search_index['inverted'].get(t, []):
            scores[did] += 1
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:limit]
    results = []
    for did, sc in sorted_docs:
        doc = search_index['docs'][did].copy()
        doc['score'] = sc
        doc['mode'] = '关键词匹配'
        doc['snippet'] = generate_snippet(doc['text'], terms)
        doc.pop('text', None)
        results.append(doc)
    return results, None


def generate_snippet(text, query_terms, max_len=300):
    best = len(text)
    lower = text.lower()
    for t in query_terms:
        pos = lower.find(t)
        if pos != -1 and pos < best:
            best = pos
    if best == len(text):
        snippet = text[:max_len]
    else:
        start = max(0, best - 80)
        snippet = text[start:start + max_len]
        if start > 0:
            snippet = '...' + snippet
        if start + max_len < len(text):
            snippet = snippet + '...'
    for t in query_terms:
        snippet = re.sub(re.escape(t), f'<<<{t}>>>', snippet, flags=re.IGNORECASE)
    snippet = snippet.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    snippet = snippet.replace('&lt;&lt;&lt;', '<mark>').replace('&gt;&gt;&gt;', '</mark>')
    return snippet


# ============================================================================
# 混合检索（向量 + 关键词）
# ============================================================================

def hybrid_search(query, limit=50, alpha=0.7):
    """混合检索：向量语义搜索 + 关键词精确匹配（仅限PDF文本）"""
    vec_results, vec_err = vector_search(query, k=_faiss_index.ntotal if _faiss_index else limit * 2)
    kw_results, kw_err = keyword_search(query, limit=limit * 2)

    if not vec_results and not kw_results:
        return [], vec_err or kw_err or '无结果'

    query_terms = tokenize(query)

    # 仅保留PDF文本结果
    vec_text = [vr for vr in vec_results if vr.get('type') == 'pdf']
    kw_text = kw_results

    if not vec_text:
        return _format_text_results(kw_text, query_terms), None
    if not kw_text:
        for vt in vec_text:
            vt['mode'] = '语义匹配'
            vt['kw_score'] = 0
        return _format_text_results(vec_text, query_terms), None

    # 归一化关键词分数
    max_kw = max((r.get('score', 0) for r in kw_text), default=1)
    kw_map = {}
    for r in kw_text:
        kw_map[(r['file'], r['page'])] = r.get('score', 0) / max(max_kw, 1)

    merged = {}
    for vr in vec_text:
        key = (vr['file'], vr['page'])
        kw_sc = kw_map.get(key, 0)
        combined = alpha * vr['score'] + (1 - alpha) * kw_sc
        vr['mode'] = '混合匹配' if kw_sc > 0 else '语义匹配'
        merged[key] = {**vr, 'combined': combined, 'kw_score': kw_sc}
        if key in kw_map:
            del kw_map[key]

    for r in kw_text:
        key = (r['file'], r['page'])
        if key not in merged:
            r['mode'] = '关键词匹配'
            r['combined'] = (1 - alpha) * (r.get('score', 0) / max(max_kw, 1))
            r['kw_score'] = r.get('score', 0) / max(max_kw, 1)
            r['vec_score'] = 0
            merged[key] = r

    sorted_items = sorted(merged.values(), key=lambda x: x['combined'], reverse=True)[:limit]
    return _format_text_results(sorted_items, query_terms), None


def _format_text_results(items, query_terms):
    """格式化PDF文本结果，生成带关键词高亮的上下文片段"""
    results = []
    for item in items:
        text = item.get('text_preview', item.get('text', ''))
        if item.get('is_scanned'):
            snippet = '(扫描页面，已通过图像语义索引)'
        else:
            snippet = generate_snippet(text, query_terms) if text else ''
        # 推断匹配模式：优先使用显式设置的mode，否则根据kw_score推断
        mode = item.get('mode', '')
        if not mode:
            kw_sc = item.get('kw_score', 0)
            vec_sc = item.get('score', 0)
            if kw_sc > 0 and vec_sc > 0:
                mode = '混合匹配'
            elif kw_sc > 0:
                mode = '关键词匹配'
            else:
                mode = '语义匹配'
        results.append({
            'type': 'pdf',
            'file': item.get('file', ''),
            'filename': item.get('filename', ''),
            'page': item.get('page', 1),
            'score': round(item.get('combined', item.get('score', 0)), 4),
            'vec_score': round(item.get('score', 0), 4),
            'kw_score': round(item.get('kw_score', 0), 4),
            'mode': mode,
            'snippet': snippet,
        })
    return results


def _merge_text_video_results(text_results, video_results, limit):
    """合并文本和视频结果，确保视频可见"""
    if not text_results and not video_results:
        return [], '无结果'

    # 文本结果按 combined 排序
    text_sorted = sorted(text_results, key=lambda x: x.get('combined', x.get('score', 0)), reverse=True)

    # 视频结果分数归一化到 [0, 1] 范围（相对于自身最大值）
    if video_results:
        max_video = max(vr.get('score', 0) for vr in video_results)
        if max_video > 0:
            for vr in video_results:
                vr['score_original'] = vr['score']
                vr['score'] = vr['score'] / max_video  # 归一化
        video_sorted = sorted(video_results, key=lambda x: x.get('score', 0), reverse=True)
    else:
        video_sorted = []

    merged = []
    # 取 top-N 文本 + top-M 视频交叉排列
    text_per_page = max(3, limit // 4)
    video_per_page = max(1, limit // 6)

    ti, vi = 0, 0
    while len(merged) < limit and (ti < len(text_sorted) or vi < len(video_sorted)):
        # 先放文本
        for _ in range(text_per_page):
            if ti < len(text_sorted) and len(merged) < limit:
                merged.append(text_sorted[ti])
                ti += 1
        # 再放视频
        for _ in range(video_per_page):
            if vi < len(video_sorted) and len(merged) < limit:
                video_sorted[vi]['vec_score'] = video_sorted[vi].pop('score', 0)
                if 'score_original' in video_sorted[vi]:
                    video_sorted[vi]['score'] = video_sorted[vi].pop('score_original')
                video_sorted[vi]['combined'] = video_sorted[vi].get('score', 0)
                merged.append(video_sorted[vi])
                vi += 1

    # 格式化为统一输出
    results = []
    for item in merged:
        if item.get('type') == 'pdf':
            results.append({
                'type': 'pdf',
                'file': item.get('file', ''),
                'filename': item.get('filename', ''),
                'page': item.get('page', 1),
                'score': round(item.get('combined', item.get('score', 0)), 4),
                'vec_score': round(item.get('score', 0), 4),
                'kw_score': round(item.get('kw_score', 0), 4),
                'mode': item.get('mode', ''),
                'snippet': item.get('snippet', generate_snippet(item.get('text_preview', ''), [])),
            })
        elif item.get('type') == 'video':
            results.append({
                'type': 'video',
                'file': item.get('file', ''),
                'filename': item.get('filename', ''),
                'frame': item.get('frame', 0),
                'timestamp': item.get('timestamp', 0),
                'score': round(item.get('combined', item.get('score', 0)), 4),
                'vec_score': round(item.get('score', 0), 4),
                'kw_score': 0,
                'mode': item.get('mode', '语义匹配（视频帧）'),
            })

    return results[:limit], None


# ============================================================================
# 路由
# ============================================================================

@app.route('/favicon.ico')
def favicon():
    return '', 204

@app.route('/')
def index():
    return render_template('index.html')


# ============================================================================
# API：数据集结构
# ============================================================================

@app.route('/api/structure')
def api_structure():
    global _dataset_tree, _dataset_flat_files
    _dataset_tree, _dataset_flat_files = scan_dataset()
    if _dataset_tree is None:
        return jsonify({'error': '数据目录不存在', 'tree': None, 'files': []})
    return jsonify({'tree': _dataset_tree, 'total_files': len(_dataset_flat_files),
                    'total_size_mb': _dataset_tree.get('total_size_mb', 0) or 0})


@app.route('/api/stats')
def api_stats():
    global _dataset_tree, _dataset_flat_files
    force = request.args.get('force', '0') == '1'
    if _dataset_tree is None or force:
        _dataset_tree, _dataset_flat_files = scan_dataset()
    if _dataset_tree is None:
        return jsonify({'error': '数据目录不存在', 'categories': [], 'total_files': 0})

    cat_summary = defaultdict(lambda: {'count': 0, 'total_size': 0, 'total_pages': 0})
    for f in _dataset_flat_files:
        c = f['category']
        cat_summary[c]['count'] += 1
        cat_summary[c]['total_size'] += f['size']
        cat_summary[c]['total_pages'] += f.get('pages', 0)

    return jsonify({
        'tree': _summarize_tree(_dataset_tree),
        'categories': {k: dict(v) for k, v in cat_summary.items()},
        'total_files': len(_dataset_flat_files),
        'pdf_count': cat_summary.get('pdf', {}).get('count', 0),
        'video_count': cat_summary.get('video', {}).get('count', 0),
        'total_pages': sum(f.get('pages', 0) for f in _dataset_flat_files),
        'index_ready': search_index['ready'],
        'index_docs': len(search_index['docs']),
        'faiss_ready': _faiss_ready,
        'faiss_vectors': _faiss_index.ntotal if _faiss_ready and _faiss_index is not None else 0,
        'pyMuPDF_available': fitz is not None,
        'opencv_available': cv2 is not None,
        'vector_search_available': _faiss_ready and faiss is not None,
    })


def _summarize_tree(node):
    summary = {
        'name': node['name'], 'path': node['path'], 'type': 'dir',
        'file_count': node.get('file_count', 0),
        'total_size_mb': node.get('total_size_mb', 0),
        'categories': node.get('categories', {}),
        'subdirs': [], 'files': [],
    }
    for child in node.get('children', []):
        if child.get('type') == 'dir':
            summary['subdirs'].append(_summarize_tree(child))
        else:
            info = {'name': child['name'], 'path': child['path'], 'category': child['category'],
                    'size_mb': child.get('size_mb', 0), 'pages': child.get('pages'),
                    'video_info': child.get('video_info')}
            summary['files'].append(info)
    return summary


# ============================================================================
# API：搜索（混合检索）
# ============================================================================

@app.route('/api/search')
def api_search():
    q = request.args.get('q', '').strip()
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 20))
    mode = request.args.get('mode', 'hybrid')

    if not q:
        return jsonify({'error': '请输入搜索关键词', 'results': [], 'total': 0})

    if mode == 'vector':
        results, err = vector_search(q, k=per_page * 3)
        if err:
            return jsonify({'error': err, 'results': [], 'total': 0})
        terms = tokenize(q)
        for r in results:
            if r.get('type') == 'pdf':
                r['mode'] = '语义匹配'
                if 'snippet' not in r:
                    text = r.get('text_preview', '')
                    r['snippet'] = generate_snippet(text, terms) if text else ''
    elif mode == 'keyword':
        results, err = keyword_search(q, limit=per_page * 3)
        if err:
            return jsonify({'error': err, 'results': [], 'total': 0})
        for r in results:
            r['mode'] = '关键词匹配'
    else:
        results, err = hybrid_search(q, limit=per_page * 3)

    if err:
        return jsonify({'error': err, 'results': [], 'total': 0})

    total = len(results)
    start = (page - 1) * per_page
    end = start + per_page
    paged = results[start:end]

    return jsonify({
        'query': q, 'results': paged, 'total': total,
        'page': page, 'per_page': per_page,
        'total_pages': max(1, (total + per_page - 1) // per_page),
        'mode': mode,
        'search_modes': {
            'hybrid': _faiss_ready and search_index['ready'],
            'vector': _faiss_ready,
            'keyword': search_index['ready'],
        },
    })


# ============================================================================
# API：PDF页
# ============================================================================

def render_pdf_page_as_image(filepath, page_num, scale=1.5):
    """将PDF页面渲染为PNG图片（用于扫描版PDF展示）"""
    if fitz is None:
        return None
    try:
        doc = fitz.open(filepath)
        if page_num < 1 or page_num > len(doc):
            doc.close()
            return None
        page = doc[page_num - 1]
        mat = fitz.Matrix(scale, scale)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
        doc.close()
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        buf.seek(0)
        return buf
    except Exception as e:
        print(f'[ERROR] 渲染PDF页面失败 {filepath} page={page_num}: {e}')
        return None


def is_garbled_text(text):
    """检测文本是否为乱码（扫描版PDF的无效提取）"""
    if not text or len(text) < 10:
        return False
    meaningful = sum(1 for c in text if '一' <= c <= '鿿' or 'a' <= c.lower() <= 'z' or '0' <= c <= '9')
    blanks = sum(1 for c in text if c in ' \t\n\r\x0c')
    body_len = max(len(text) - blanks, 1)
    return meaningful < 10 or (meaningful / body_len) < 0.3


@app.route('/api/pdf/page')
def api_pdf_page():
    file_rel = request.args.get('file', '')
    page_num = int(request.args.get('page', 1))
    filepath = DATA_DIR / file_rel
    if not filepath.exists() or not filepath.is_file():
        return jsonify({'error': '文件不存在'}), 404

    # 获取PDF总页数（真实页数）
    pdf_total = get_pdf_page_count(str(filepath))
    if page_num < 1 or page_num > pdf_total:
        return jsonify({'error': f'页码超出范围 (1-{pdf_total})'}), 400

    # 按真实页码查找文本
    pages = extract_pdf_text(str(filepath))
    text = ''
    for pn, txt in pages:
        if pn == page_num:
            text = txt
            break

    is_image = is_garbled_text(text)
    show_image = is_image or not text
    image_url = f'/api/pdf/page/image?file={urllib.parse.quote(file_rel)}&page={page_num}' if show_image else None
    return jsonify({
        'file': file_rel,
        'page': page_num,
        'total_pages': pdf_total,
        'text': text if not show_image else '(该页面为扫描/图片型PDF)',
        'is_image': show_image,
        'image_url': image_url,
    })


@app.route('/api/pdf/page/image')
def api_pdf_page_image():
    """返回PDF页面的渲染图（用于扫描版/图片型PDF的展示）"""
    file_rel = request.args.get('file', '')
    page_num = int(request.args.get('page', 1))
    filepath = DATA_DIR / file_rel
    if not filepath.exists() or not filepath.is_file():
        return jsonify({'error': '文件不存在'}), 404
    buf = render_pdf_page_as_image(str(filepath), page_num)
    if buf is None:
        return jsonify({'error': '页面渲染失败'}), 500
    return send_file(buf, mimetype='image/png')


# ============================================================================
# API：视频
# ============================================================================

@app.route('/api/videos')
def api_videos():
    global _dataset_flat_files
    if not _dataset_flat_files:
        _, _dataset_flat_files = scan_dataset()
    videos = []
    for f in _dataset_flat_files:
        if f['category'] != 'video':
            continue
        info = video_meta_cache.get(f['path'])
        if info is None and cv2:
            info = get_video_info(str(DATA_DIR / f['path']))
            if info:
                video_meta_cache[f['path']] = info
        f['video_info'] = info
        videos.append(f)
    videos.sort(key=lambda x: x['path'])
    return jsonify({'videos': videos})


@app.route('/api/video/frame')
def api_video_frame():
    file_rel = request.args.get('file', '')
    frame_num = int(request.args.get('frame', 0))
    filepath = DATA_DIR / file_rel
    if not filepath.exists() or not filepath.is_file():
        return jsonify({'error': '视频文件不存在'}), 404
    info = video_meta_cache.get(file_rel)
    if info is None and cv2:
        info = get_video_info(str(filepath))
        if info:
            video_meta_cache[file_rel] = info
    if info and (frame_num < 0 or frame_num >= info['total_frames']):
        return jsonify({'error': f'帧号超出范围 (0-{info["total_frames"] - 1})'}), 400
    img = extract_frame(str(filepath), frame_num)
    if img is None:
        return jsonify({'error': '无法提取该帧。可能原因：视频编码不兼容、文件损坏。WMV文件建议转MP4后重试。'}), 500
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return send_file(buf, mimetype='image/png')


@app.route('/api/video/frame/pixels')
def api_video_frame_pixels():
    file_rel = request.args.get('file', '')
    frame_num = int(request.args.get('frame', 0))
    x = int(request.args.get('x', 0))
    y = int(request.args.get('y', 0))
    radius = int(request.args.get('radius', 5))
    filepath = DATA_DIR / file_rel
    if not filepath.exists() or not filepath.is_file():
        return jsonify({'error': '视频文件不存在'}), 404
    img = extract_frame(str(filepath), frame_num)
    if img is None:
        return jsonify({'error': '无法提取该帧。可能原因：视频编码不兼容、文件损坏。WMV文件建议转MP4后重试。'}), 500
    arr = np.array(img)
    h, w = arr.shape[:2]
    xs, xe = max(0, x - radius), min(w, x + radius + 1)
    ys, ye = max(0, y - radius), min(h, y + radius + 1)
    region = arr[ys:ye, xs:xe]
    if region.size == 0:
        return jsonify({'error': '坐标超出范围'}), 400
    r_mean, g_mean, b_mean = int(np.mean(region[:, :, 0])), int(np.mean(region[:, :, 1])), int(np.mean(region[:, :, 2]))
    if 0 <= y < h and 0 <= x < w:
        cr, cg, cb = arr[y, x].tolist()
    else:
        cr = cg = cb = None
    return jsonify({
        'x': x, 'y': y, 'image_size': {'width': w, 'height': h},
        'center_rgb': {'r': cr, 'g': cg, 'b': cb},
        'region_avg_rgb': {'r': r_mean, 'g': g_mean, 'b': b_mean},
        'hex': f'#{cr:02x}{cg:02x}{cb:02x}' if cr is not None else None,
    })


# ============================================================================
# API：索引管理
# ============================================================================

@app.route('/api/reindex')
def api_reindex():
    global _dataset_tree, _dataset_flat_files
    _dataset_tree, _dataset_flat_files = scan_dataset()
    errors = []
    try:
        build_index(force=True)
    except Exception as e:
        errors.append(f'关键词索引: {e}')
    try:
        build_vector_index(force=True)
    except Exception as e:
        errors.append(f'向量索引: {e}')
    return jsonify({
        'ok': len(errors) == 0,
        'errors': errors,
        'kw_docs': len(search_index['docs']),
        'faiss_vectors': _faiss_index.ntotal if _faiss_index is not None else 0,
    })


@app.route('/api/index_status')
def api_index_status():
    return jsonify({
        'ready': search_index['ready'],
        'doc_count': len(search_index['docs']),
        'term_count': len(search_index['inverted']),
        'faiss_ready': _faiss_ready,
        'faiss_vectors': _faiss_index.ntotal if _faiss_index is not None else 0,
    })


@app.template_filter('format_size')
def _format_size(sz):
    if sz < 1024:
        return f'{sz} B'
    elif sz < 1024 * 1024:
        return f'{sz / 1024:.1f} KB'
    elif sz < 1024 * 1024 * 1024:
        return f'{sz / (1024 * 1024):.1f} MB'
    return f'{sz / (1024 * 1024 * 1024):.1f} GB'


# ============================================================================
# 启动
# ============================================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='数据集处理Demo - 向量检索版')
    parser.add_argument('--host', default='127.0.0.1', help='绑定地址')
    parser.add_argument('--port', type=int, default=5000, help='端口号')
    parser.add_argument('--data-dir', default=str(DATA_DIR), help='数据集目录')
    parser.add_argument('--no-index', action='store_true', help='启动时不自动构建索引')
    parser.add_argument('--no-vector', action='store_true', help='禁用向量检索')
    parser.add_argument('--debug', action='store_true', help='调试模式')
    args = parser.parse_args()

    if args.data_dir:
        DATA_DIR = Path(args.data_dir)

    print('=' * 60)
    print('  数据集处理Demo (FAISS向量检索版)')
    print('=' * 60)
    print(f'  数据目录: {DATA_DIR}')
    print(f'  缓存目录: {CACHE_DIR}')
    print(f'  PyMuPDF:   {"可用" if fitz else "不可用"}')
    print(f'  OpenCV:    {"可用" if cv2 else "不可用"}')
    print(f'  FAISS:     {"可用" if faiss else "不可用"}')
    print(f'  PyTorch:   {"可用" if torch else "不可用"}')
    print(f'  Transformers: {"可用" if transformers else "不可用"}')
    print(f'  Jieba:     可用')
    print('=' * 60)

    # 扫描
    _dataset_tree, _dataset_flat_files = scan_dataset()
    if _dataset_tree:
        total = len(_dataset_flat_files)
        cats = _dataset_tree.get('categories', {})
        print(f'  数据集: {total} 个文件')
        for cat, count in sorted(cats.items()):
            label = {'pdf': '[PDF]', 'video': '[VID]', 'image': '[IMG]', 'text': '[TXT]', 'archive': '[ZIP]', 'other': '[OTH]'}
            print(f'    {label.get(cat, "[?]")} {cat}: {count} 个')
    else:
        print(f'  [WARN] 数据目录不存在或为空: {DATA_DIR}')

    # 视频元数据预热
    if cv2:
        for f in _dataset_flat_files:
            if f['category'] == 'video':
                fp = str(DATA_DIR / f['path'])
                info = get_video_info(fp)
                if info:
                    video_meta_cache[f['path']] = info
                    print(f'  [VIDEO] {f["name"]}: {info["total_frames"]}帧, {info["fps"]}fps, {info["width"]}x{info["height"]}')

    # 关键词索引
    if not args.no_index and fitz:
        build_index()
    elif not fitz:
        print('[WARN] PyMuPDF不可用, 跳过关键词索引')

    # FAISS向量索引
    if not args.no_index and not args.no_vector and faiss and torch and transformers:
        build_vector_index()
    elif not args.no_index:
        print('[WARN] FAISS/torch/transformers 缺失, 跳过向量索引 (关键词搜索仍可用)')

    print(f'\n  搜索模式: {"混合检索(向量+关键词)" if _faiss_ready else "关键词检索"}')
    print(f'  向量库:   {_faiss_index.ntotal if _faiss_index else 0} 个向量')
    print(f'  关键词库: {len(search_index["docs"])} 个文档, {len(search_index["inverted"])} 个词条')
    print(f'\n  访问: http://{args.host}:{args.port}')
    print('=' * 60)

    app.run(host=args.host, port=args.port, debug=args.debug)
