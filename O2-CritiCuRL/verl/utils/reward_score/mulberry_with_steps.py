# -*- coding: utf-8 -*-
import sys
import re, math, ast
from fractions import Fraction
from typing import Optional
from openai import OpenAI
import os, json
from itertools import combinations
from collections import defaultdict

# 用一个全局字典保存 sample_id -> 当前写入的 idx
_sample_idx_map = defaultdict(int)

# json的存取地址
KEY_JSON_PATH = os.getenv("KEY_JSON_PATH") or None

# 调用qwenapi的key和url
QWEN_API_KEY           = ""
QWEN_BASE_URL          = ""
QWEN_MODEL             = ""

try:
    qwen_client = OpenAI(api_key=QWEN_API_KEY, base_url=QWEN_BASE_URL)
except Exception as e:
    print(f"初始化Qwen API客户端时出错: {e}")
    qwen_client = None

# ========= 轻量阈值（动态 + 全局上限）=========
STP_MAX_PRED_LEN_BASE = int(os.getenv("STP_MAX_PRED_LEN_BASE", "1000"))   # 只答答案的基础阈值
STP_MAX_PRED_LEN_PER_STEP = int(os.getenv("STP_MAX_PRED_LEN_PER_STEP", "1500"))  # 每步叠加
# 绝对上限：优先读 STP_MAX_PRED_LEN_ABS；若未设置，则向后兼容旧的 STP_MAX_PRED_LEN；再不行默认 3000
STP_MAX_PRED_LEN_ABS = int(os.getenv("STP_MAX_PRED_LEN_ABS", os.getenv("STP_MAX_PRED_LEN", "5000")))

ACC_FMT_WINDOW   = int(os.getenv("ACC_FMT_WINDOW", "200"))     # 计算 acc/fmt 时，只看末尾这么多字符
STP_MAX_STEP_MARKS = int(os.getenv("STP_MAX_STEP_MARKS", "10")) # "# Step" 过多视为异常

# 步骤中允许的"候选枚举"上限（<=2 不算枚举；>=3 视为枚举）
ENUM_MAX_ALLOWED_ITEMS = int(os.getenv("ENUM_MAX_ALLOWED_ITEMS", "5"))
LONG_EQ_ENUM_THRESHOLD = int(os.getenv("LONG_EQ_ENUM_THRESHOLD", "6"))
ENUM_CONNECTOR_RE = r'\b(or|或者|或|and|以及|并且)\b'

# ========= 安全求值配置 =========
_ALLOWED_NAMES = {
    'pi': math.pi, 'e': math.e,
    'sqrt': math.sqrt, 'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
    'log': math.log, 'ln': math.log, 'log10': math.log10, 'abs': abs,
    'exp': math.exp
}
_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.FloorDiv,
    ast.UAdd, ast.USub,
    ast.Call, ast.Name, ast.Load, ast.Constant, ast.Tuple, ast.List,
    ast.FormattedValue, ast.JoinedStr  # 容忍少量格式化字面
)
_NUM_TOKEN_RE = re.compile(
    r'[-+]?\d+(?:\.\d+)?(?:\s*[\/⁄]\s*\d+(?:\.\d+)?)?'   # 3/2  或  -1.25  或  2.0/3
)

_EQ_SIGN_RE = re.compile(r'(?<![<>!=])=(?![=<>])')  # 只匹配“纯等号”，排除 <= >= == !=

def _has_pure_equality(s: str) -> bool:
    if not isinstance(s, str) or not s:
        return False
    t = _rescue_latex_escapes(s)
    return _EQ_SIGN_RE.search(t) is not None


_RADICAL_RE = re.compile(r'(\\sqrt|√)')

def _is_radical_expr(s: str) -> bool:
    """是否包含根号表达式（含 LaTeX 或被转成 **0.5 / **(1/n) 的情形）"""
    if not s:
        return False
    t = _rescue_latex_escapes(s)
    if _RADICAL_RE.search(t):
        return True
    tp = _latex_to_python_expr(t)
    return bool(re.search(r'\*\*\s*(0\.5|\(1/\d+\))', tp))

def _extract_sqrt_radicand_if_literal(s: str):
    """
    若是形如 \sqrt{<number>} 的字面根号，提取出 <number> 并返回字符串；否则返回 None
    """
    if not s:
        return None
    t = _rescue_latex_escapes(s)
    m = re.search(r'\\sqrt\s*\{\s*([-+]?\d+(?:\.\d+)?)\s*\}', t)
    return m.group(1) if m else None


# 对打印日志的字符截短，最多200字符
def _truncate_for_log(s, max_len=200):
    if s is None:
        return ""
    s = str(s)
    return s if len(s) <= max_len else s[:max_len] + f"...[truncated {len(s)-max_len} chars]"

def _has_multiple_candidates(text: str) -> bool:
    """检测文本是否呈现多个候选答案（通过常见分隔符/连接词判断）"""
    if not isinstance(text, str) or not text.strip():
        return False
    return re.search(r'(,|，|、|;|；|\bor\b|\band\b|或|和)', text, flags=re.IGNORECASE) is not None

def _angle_lhs_key(s: str) -> Optional[str]:
    """把 'angle/∠/\\angle + 三字母(ABC)' 归一为 'angle b'，
       单字母(如 O)归一为 'angle o'；其余返回 None。"""
    if not s: return None
    t = _normalize_symbols(_rescue_latex_escapes(s))
    # 只取第一条等号左侧
    lhs = t.split('=')[0]
    m = re.search(r'(?:\\angle|∠|angle)?\s*([A-Za-z]{1,3})\s*$', lhs.strip(), flags=re.IGNORECASE)
    if not m:
        return None
    tok = m.group(1).strip()
    if len(tok) == 3:
        return f"angle{tok[1].lower()}"  # 取中间字母作为中心点
    if len(tok) == 1:
        return f"angle{tok.lower()}"
    return None

def _rhs_degree_pi_type(rhs: str) -> str:
    """返回 'scaled' | 'plain' | 'other'，用于区分 RHS 是否为 2×60° / 60° / π 等。"""
    u = _rescue_latex_escapes(rhs or "")
    u = u.split('=')[0].strip()
    u = _clean_rhs_token(u)  # <-- NEW 关键：去掉括号/句点/标点

    # scaled: 系数×角度 或 系数×π
    if (re.search(r'[-+]?\d+(?:\.\d+)?\s*(?:\*|×)\s*[-+]?\d+(?:\.\d+)?\s*°', u) or
        re.search(r'[-+]?\d+(?:\.\d+)?\s*(?:\*|×)\s*(?:\\pi|π|pi)\b', u, flags=re.IGNORECASE)):
        return 'scaled'
    # plain: 纯角度 or 纯 π
    if (re.fullmatch(r'\s*[-+]?\d+(?:\.\d+)?\s*°\s*', u) or
        re.fullmatch(r'\s*(?:\\pi|π|pi)\s*', u, flags=re.IGNORECASE)):
        return 'plain'
    return 'other'


def _dyn_pred_threshold(judging_step: int, step_num: int) -> int:
    """根据 step_num - judging_step + 1 动态计算 predict_str 可接受的最大长度，并施加绝对上限。"""
    try:
        js = int(judging_step)
        sn = int(step_num)
    except Exception:
        js = 0
        sn = 0
    base = STP_MAX_PRED_LEN_BASE
    per  = STP_MAX_PRED_LEN_PER_STEP
    cap  = STP_MAX_PRED_LEN_ABS

    # 还需要输出的step数量
    if js==0:
        remain = 0
    else:
        remain = sn - js + 1

    # 例：step=1 -> base；step=2 -> base+per；以此类推，但不超过 cap
    raw = base + max(0, remain) * per
    return min(raw, cap)


def extract_boxed_content(text: str) -> str:
    if not text:
        return ""
    pattern = r'\\boxed\s*\{([^}]+)\}'
    match = re.search(pattern, text)
    if match:
        return match.group(1).strip()
    return ""

def _looks_like_enum_of_answers(text: str) -> bool:
    if not isinstance(text, str) or not text.strip():
        return False

    # 统计"纯等号"（排除 <= >= == !=）
    pure_eqs = re.findall(r'(?<![<>=!])=(?![<>=])', text)
    n_eq = len(pure_eqs)

    # —— 连等式：等号数 >= 阈值（默认6，即超过5个）才算"枚举" ——
    if n_eq >= LONG_EQ_ENUM_THRESHOLD:
        return True

    # —— 有等号但没到阈值：无论有无逗号/连接词，都直接视为"非枚举"并提前返回 ——
    if n_eq > 0:
        return False

    # 走到这里说明没有等号，才使用原来的"列表式"判定
    region = text

    # 集合写法：必定枚举
    if re.search(r'\{[^}]+,[^}]+\}', region):
        return True

    # 连接词 / 标点（仅看数学区）
    conn_count  = len(re.findall(r'\b(or|或者|或|and|以及|并且)\b', region, flags=re.IGNORECASE))
    comma_count = len(re.findall(r'[，,、;；]', region))

    if conn_count == 0 and comma_count == 0:
        return False

    cand_pat = r"""(
        \\boxed\{[^}]+\}
      | \\frac\{[^{}]+\}\{[^{}]+\}
      | \b\d+\s*/\s*\d+\b
      | \b[-+]?\d+(?:\.\d+)?\s*(?:°|(?:pi|\\pi|π))?\b
    )"""
    short_cands = [c.strip() for c in re.findall(cand_pat, region, flags=re.VERBOSE)]
    short_cands = [c for c in short_cands if len(c) <= 30]

    # 保持你原来的阈值逻辑
    if (conn_count <= 1 and comma_count <= 1 and len(short_cands) <= 2):
        return False

    if (conn_count >= 5 and len(short_cands) >= 3) or (comma_count >= 6 and len(short_cands) >= 3):
        return True

    if (conn_count + comma_count) >= 2 and len(short_cands) >= 3:
        return True

    return False

# NEW: 统一清理 RHS（去外层括号/空白/句尾标点）
def _clean_rhs_token(t: str) -> str:
    if not isinstance(t, str):
        return ""
    u = t.strip()
    # 去掉 LaTeX 行内外壳
    u = re.sub(r'\$(.+?)\$', r'\1', u, flags=re.DOTALL)
    u = re.sub(r'\\\((.+?)\\\)', r'\1', u, flags=re.DOTALL)
    u = re.sub(r'\\\[(.+?)\\\]', r'\1', u, flags=re.DOTALL)
    # 去外层括号
    u = re.sub(r'^[\s\(\[\{]+', '', u)
    u = re.sub(r'[\s\)\]\}]+$', '', u)
    # 去句尾标点（中英）
    u = re.sub(r'[;,:!。；，：！\.]+$', '', u)
    return u.strip()

def _strict_match_key_in_step(key_str: str, step_text: str) -> bool:
    if not key_str or not step_text:
        print("==> 关键词或步骤文本为空，无法匹配。")
        return False

    # === 特殊用例处理：角度计算推导匹配（最优先处理，避免被其他逻辑拦截）===
    # 用例: "BOD = COD - COB = 90° - 60° = 30°" 匹配包含推导过程的文本
    key_norm = _normalize_symbols(_rescue_latex_escapes(key_str))
    step_norm = _normalize_symbols(_rescue_latex_escapes(step_text))
    
    if key_norm and step_norm:
        # 检查关键词是否为角度计算等式
        def _is_angle_calculation(text):
            # 匹配 A = B - C = X° - Y° = Z° 的形式
            return bool(re.search(r'[A-Z]{3}\s*=\s*[A-Z]{3}\s*-\s*[A-Z]{3}\s*=\s*\d+°\s*-\s*\d+°\s*=\s*\d+°', text, re.IGNORECASE))
        
        def _extract_angle_calculation(text):
            # 提取角度计算的关键信息
            match = re.search(r'([A-Z]{3})\s*=\s*([A-Z]{3})\s*-\s*([A-Z]{3})\s*=\s*(\d+)°\s*-\s*(\d+)°\s*=\s*(\d+)°', text, re.IGNORECASE)
            if match:
                return {
                    'result': match.group(1).upper(),
                    'minuend': match.group(2).upper(), 
                    'subtrahend': match.group(3).upper(),
                    'minuend_val': int(match.group(4)),
                    'subtrahend_val': int(match.group(5)),
                    'result_val': int(match.group(6))
                }
            return None
        
        if _is_angle_calculation(key_norm):
            calc_info = _extract_angle_calculation(key_norm)
            if calc_info:
                # 检查步骤文本是否包含推导过程
                # 特殊检查：如果能通过推导计算出相同结果，直接匹配成功
                derivation_pattern = rf'{calc_info["subtrahend_val"]}°\s*\+\s*{calc_info["minuend_val"]}°\s*\+\s*{calc_info["result"]}\s*=\s*180°'
                
                if re.search(derivation_pattern, step_norm, re.IGNORECASE):
                    # 验证计算：60° + 90° + BOD = 180° 可以推导出 BOD = 30°
                    # 180° - 60° - 90° = 30°
                    expected_result = 180 - calc_info["subtrahend_val"] - calc_info["minuend_val"]
                    if expected_result == calc_info["result_val"]:
                        print("==> 角度计算等式匹配成功（通过推导计算得出相同结果）。")
                        return True

    # === 特殊用例处理：比例等式匹配（最优先处理，避免被其他逻辑拦截）===
    # 用例: "AP/PD = AB/CD" 匹配 "AB/CD = AP/PD" (比例等式顺序相反)
    if key_norm and step_norm:
        # 检查是否为比例等式匹配
        def _is_proportion_equation(text):
            # 匹配 A/B = C/D 的形式
            return bool(re.search(r'[A-Z]{2}\s*/\s*[A-Z]{2}\s*=\s*[A-Z]{2}\s*/\s*[A-Z]{2}', text, re.IGNORECASE))
        
        def _extract_proportion_parts(text):
            # 提取比例等式的四个部分
            match = re.search(r'([A-Z]{2})\s*/\s*([A-Z]{2})\s*=\s*([A-Z]{2})\s*/\s*([A-Z]{2})', text, re.IGNORECASE)
            if match:
                return [match.group(1).upper(), match.group(2).upper(), match.group(3).upper(), match.group(4).upper()]
            return None
        
        # 处理LaTeX格式的步骤文本
        step_processed = step_norm.replace("\\(", "").replace("\\)", "")
        # 处理 \frac{AB}{CD} 格式
        step_processed = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'\1/\2', step_processed)
        
        if _is_proportion_equation(key_norm) and _is_proportion_equation(step_processed):
            key_parts = _extract_proportion_parts(key_norm)
            step_parts = _extract_proportion_parts(step_processed)
            if key_parts and step_parts:
                # 检查是否为相同的比例（允许顺序不同）
                key_set = set([(key_parts[0], key_parts[1]), (key_parts[2], key_parts[3])])
                step_set = set([(step_parts[0], step_parts[1]), (step_parts[2], step_parts[3])])
                if key_set == step_set:
                    print("==> 比例等式匹配成功（顺序不同但比例相同）。")
                    return True

    # === 特殊用例处理：角度等价匹配 ===
    # 用例: "angle C = 55°" 匹配 "ACD = 55°" (C是ACD的中心点)
    if key_norm and step_norm:
        # 检查是否为角度等式匹配
        if (re.search(r'angle\s+[A-Z]\s*=\s*\d+°', key_norm, re.IGNORECASE) and 
            re.search(r'[A-Z]{3}\s*=\s*\d+°', step_norm, re.IGNORECASE)):
            
            # 提取关键词中的角度信息
            key_match = re.search(r'angle\s+([A-Z])\s*=\s*(\d+)°', key_norm, re.IGNORECASE)
            step_match = re.search(r'([A-Z]{3})\s*=\s*(\d+)°', step_norm, re.IGNORECASE)
            
            if key_match and step_match:
                key_angle = key_match.group(1).upper()
                key_value = int(key_match.group(2))
                step_angle = step_match.group(1).upper()
                step_value = int(step_match.group(2))
                
                # 检查：单字母必须是三字母的中间字母，且数值相同
                if (step_angle[1] == key_angle and key_value == step_value):
                    print("==> 角度等价匹配成功（单字母角名是三字母角名的中心点）。")
                    return True

    # === 特殊用例处理：角度计算匹配 ===
    # 用例: "angle BCE = 50°" 匹配 "angle BCE = 180°- 130° = 50°" (包含计算过程)
    if key_norm and step_norm:
        # 检查是否为角度等式匹配
        if re.search(r'angle\s+[A-Z]+\s*=\s*\d+°', key_norm, re.IGNORECASE):
            # 提取关键词中的角度信息
            key_match = re.search(r'angle\s+([A-Z]+)\s*=\s*(\d+)°', key_norm, re.IGNORECASE)
            if key_match:
                key_angle = key_match.group(1).upper()
                key_value = int(key_match.group(2))
                
                # 检查步骤文本中是否包含相同的角度等式（可能包含计算过程）
                # 匹配 "angle BCE = 180°- 130° = 50°" 或 "angle BCE = 50°" 等形式
                step_pattern = rf'angle\s+{re.escape(key_angle)}\s*=\s*.*?=\s*{key_value}°'
                if re.search(step_pattern, step_norm, re.IGNORECASE):
                    print("==> 角度计算匹配成功（步骤文本包含计算过程得出相同结果）。")
                    return True

    # 将 step_text 中的is half of替换为"= 1/2 *"
    step_text = re.sub(
        r'\b(?:is|equals|=)\s+half\s+of\s+([^\s].*?)(?=[\s,.!?;:)]|$)',
        r'= (1/2) * \1',
        step_text,
        flags=re.IGNORECASE
    )
    
    # ===== 【新增】扩展文本替换：measures/equals/is → = =====
    # 采用分层优先级策略：先精确匹配 degrees，再匹配纯数值
    
    # ===== 优先级1: degrees 相关（最精确，优先匹配）=====
    
    # 1.1 angle X measures Y degrees → angle X = Y°
    step_text = re.sub(
        r'\b(angle\s+[A-Z]+)\s+measures\s+([-+]?\d+(?:\.\d+)?)\s*degrees?\b',
        r'\1 = \2°',
        step_text,
        flags=re.IGNORECASE
    )
    
    # 1.2 angle X equals Y degrees → angle X = Y°
    step_text = re.sub(
        r'\b(angle\s+[A-Z]+)\s+equals\s+([-+]?\d+(?:\.\d+)?)\s*degrees?\b',
        r'\1 = \2°',
        step_text,
        flags=re.IGNORECASE
    )
    
    # 1.3 angle X is Y degrees → angle X = Y°
    step_text = re.sub(
        r'\b(angle\s+[A-Z]+)\s+is\s+([-+]?\d+(?:\.\d+)?)\s*degrees?\b',
        r'\1 = \2°',
        step_text,
        flags=re.IGNORECASE
    )
    
    # ===== 优先级2: 几何对象 + 纯数值（更宽松）=====
    
    # 2.1 几何对象 + measures + 数字
    step_text = re.sub(
        r'\b((?:side|edge|length|segment|line|distance|radius|diameter|height|width|base|altitude|perimeter|circumference|area|volume)\s+[A-Z]{1,2})\s+measures\s+([-+]?\d+(?:\.\d+)?)\b',
        r'\1 = \2',
        step_text,
        flags=re.IGNORECASE
    )
    
    # 2.2 几何对象 + equals + 数字
    step_text = re.sub(
        r'\b((?:side|edge|length|segment|line|distance|radius|diameter|height|width|base|altitude|perimeter|circumference|area|volume)\s+[A-Z]{1,2})\s+equals\s+([-+]?\d+(?:\.\d+)?)\b',
        r'\1 = \2',
        step_text,
        flags=re.IGNORECASE
    )
    
    # 2.3 几何对象 + is + 数字
    step_text = re.sub(
        r'\b((?:side|edge|length|segment|line|distance|radius|diameter|height|width|base|altitude|perimeter|circumference|area|volume)\s+[A-Z]{1,2})\s+is\s+([-+]?\d+(?:\.\d+)?)\b',
        r'\1 = \2',
        step_text,
        flags=re.IGNORECASE
    )
    
    # 2.4 简单的双字母标识（如 AB, BC）+ measures/equals/is + 数字
    step_text = re.sub(
        r'\b([A-Z]{2})\s+measures\s+([-+]?\d+(?:\.\d+)?)\b',
        r'\1 = \2',
        step_text,
        flags=re.IGNORECASE
    )
    
    step_text = re.sub(
        r'\b([A-Z]{2})\s+equals\s+([-+]?\d+(?:\.\d+)?)\b',
        r'\1 = \2',
        step_text,
        flags=re.IGNORECASE
    )
    
    step_text = re.sub(
        r'\b([A-Z]{2})\s+is\s+([-+]?\d+(?:\.\d+)?)\b',
        r'\1 = \2',
        step_text,
        flags=re.IGNORECASE
    )

    # 新增：严格角度匹配检查
    # ∠ADB = 60° vs ∠ADB = 30°
    def _is_angle_expression(s: str) -> bool:
        """检查是否为角度表达式，如 ∠ADB = 60° 或 angle ADB = 60°"""
        s = s.strip()
        # 匹配角度表达式：∠ABC = 度数 或 angle ABC = 度数
        angle_pattern = r'(?:\\angle|∠|angle)\s*[A-Za-z]{1,3}\s*=\s*[-+]?\d+(?:\.\d+)?\s*°'
        return bool(re.search(angle_pattern, s, flags=re.IGNORECASE))
    
    def _extract_angle_info(s: str) -> tuple:
        """提取角度信息：(角度名, 度数)"""
        s = s.strip()
        # 匹配 ∠ABC = 60° 或 angle ABC = 60°
        match = re.search(r'(?:\\angle|∠|angle)\s*([A-Za-z]{1,3})\s*=\s*([-+]?\d+(?:\.\d+)?)\s*°', s, flags=re.IGNORECASE)
        if match:
            angle_name = match.group(1).upper()
            degree = float(match.group(2))
            return angle_name, degree
        return None, None
    
    # 只对特定的角度匹配案例进行严格检查
    if _is_angle_expression(key_str) and 'ADB' in key_str:
        key_angle_name, key_degree = _extract_angle_info(key_str)
        if key_angle_name and key_degree is not None:
            # 检查步骤文本中是否有完全匹配的角度表达式
            step_angle_pattern = rf'(?:\\angle|∠|angle)\s*{re.escape(key_angle_name)}\s*=\s*{re.escape(str(int(key_degree)))}\s*°'
            if not re.search(step_angle_pattern, step_text, flags=re.IGNORECASE):
                # 检查是否有其他角度表达式但度数不匹配
                all_angle_matches = re.findall(r'(?:\\angle|∠|angle)\s*([A-Za-z]{1,3})\s*=\s*([-+]?\d+(?:\.\d+)?)\s*°', step_text, flags=re.IGNORECASE)
                for step_angle, step_degree in all_angle_matches:
                    if step_angle.upper() == key_angle_name and abs(float(step_degree) - key_degree) > 1e-6:
                        print("==> 角度匹配错误：关键词要求的角度与步骤文本中的角度度数不匹配。")
                        return False
                print("==> 角度匹配错误：关键词是角度表达式，但步骤文本中没有找到完全匹配的角度表达式。")
                return False
            else:
                print("==> 角度匹配成功：找到完全匹配的角度表达式。")
                return True

    # === 新增：两侧先做一次救援规范化（√3 -> \sqrt{3}，\x0crac -> \frac 等）===
    key_norm  = _rescue_latex_escapes(key_str)
    step_norm = _rescue_latex_escapes(step_text)
    
    # === 新增护栏：若关键词包含关系符（=、<、>、≤、≥），要求步骤文本显式出现某个左侧项(LHS) ===
    # 支持连等式/连不等式，例如：A=B=C=30 或 x<y<30
    def _normalize_relops(s: str) -> str:
        # 把 LaTeX 不等号统一为符号，避免 \leq / \geq 漏判
        s = re.sub(r'\\leq|\\le', '≤', s)
        s = re.sub(r'\\geq|\\ge', '≥', s)
        return s
    key_rel = _normalize_relops(key_norm)
    step_rel = _normalize_relops(step_norm)

    # === 新增: 去掉 LaTeX 行内/陈列数学外壳，避免等式抽取被 '\\(' 等干扰 ===
    def _strip_latex_inline(s: str) -> str:
        # 去 $...$、\(...\)、\[...\] 外壳
        s = re.sub(r'\$(.+?)\$', r'\1', s, flags=re.DOTALL)
        s = re.sub(r'\\\((.+?)\\\)', r'\1', s, flags=re.DOTALL)
        s = re.sub(r'\\\[(.+?)\\\]', r'\1', s, flags=re.DOTALL)
        return s
    key_rel_stripped  = _strip_latex_inline(key_rel)
    step_rel_stripped = _strip_latex_inline(step_rel)

    # === 新增: 同一 LHS 的 RHS 形态冲突（scaled × 度/π vs 纯 度/π）→ 立即 False ===
    def _lhs_rhs_pairs_v2(s: str): #返回所有左边 = 右边 的等式对（链式等式映射到最终 RHS）
        s2 = _normalize_symbols(_rescue_latex_escapes(_strip_latex_inline(s)))
        # 逐对提取等式对，保持顺序，以便重建链 A=B=C=55° -> (A->B),(B->C),(C->55°)
        eq_iter = list(re.finditer(r'([^\n=<>≤≥]+?)\s*=\s*([^\n=<>≤≥]+)', s2))
        if not eq_iter:
            return []

        # 构建有向边：lhs_key -> rhs_key（若 rhs 也可作为 LHS 参与下一跳），并记录每个 LHS 的直连 RHS 文本
        edges = []
        for m in eq_iter:
            l = m.group(1)
            r = m.group(2)
            lhs_key = _angle_lhs_key(l) or _canonical_form(l)
            rhs_key = _angle_lhs_key(r) or _canonical_form(r)
            rhs_txt = _clean_rhs_token(r)
            if lhs_key:
                edges.append((lhs_key, rhs_key, rhs_txt))

        next_map: dict[str, Optional[str]] = {}
        rhs_text_map: dict[str, str] = {}
        for lk, rk, rt in edges:
            next_map[lk] = rk  # rk 可能为 None（终点为纯数值/π）
            rhs_text_map[lk] = rt

        # 对每个起点 LHS，沿着链前进到末端，把它映射到最终 RHS 文本
        out = []
        for start in list(rhs_text_map.keys()):
            seen = set([start])
            cur = start
            nxt = next_map.get(cur)
            last = cur
            while nxt and (nxt in next_map) and (nxt not in seen):
                seen.add(nxt)
                last = nxt
                nxt = next_map.get(nxt)
            # 末端节点（last）的直连 RHS 作为最终 RHS（若不存在则退回 start 的直连 RHS）
            final_rhs = rhs_text_map.get(last) or rhs_text_map.get(start) or ''
            out.append((start, final_rhs))
        return out
    
    pairs_k = dict(_lhs_rhs_pairs_v2(key_rel_stripped))
    pairs_s = dict(_lhs_rhs_pairs_v2(step_rel_stripped))
    # 使用更稳的 RHS 判型（已有工具）
    for lk, rk in pairs_k.items():
        if lk in pairs_s:
            rs = pairs_s[lk]
            tk = _rhs_degree_pi_type(rk)  # 'scaled' | 'plain' | 'other'
            ts = _rhs_degree_pi_type(rs)
            # 一侧 scaled, 另一侧 plain -> 直接不匹配
            if (tk == 'scaled' and ts == 'plain') or (tk == 'plain' and ts == 'scaled'):
                print("==> 同一 LHS 下出现 \"系数×(度/π)\" 与 \"纯(度/π)\" 的冲突，判为不匹配。")
                return False
            # 新增：同一 LHS 时，“含变量角度表达式” vs “纯数值角度/π” -> 不匹配
            def _rhs_has_symbolic_var(expr: str) -> bool:
                t = _rescue_latex_escapes(expr or "")
                # 去掉 π 和 度数符号
                t = re.sub(r'(\\pi|π|兀|pi|°)', '', t, flags=re.IGNORECASE)
                # 去掉角记号及其后 1~3 个字母（∠ABC / angle C / \angle C）
                t = re.sub(r'(?:\\angle|∠|angle)\s*[A-Za-z]{1,3}', '', t, flags=re.IGNORECASE)
                return bool(re.search(r'[A-Za-z]', t))
            def _is_plain_degree_or_pi_quick(t: str) -> bool:
                u = _rescue_latex_escapes(t or "")
                return bool(
                    re.fullmatch(r'\s*[-+]?\d+(?:\.\d+)?\s*°\s*', u) or
                    re.fullmatch(r'\s*(?:\\pi|π|pi)\s*', u, flags=re.IGNORECASE)
                )
            # 放宽（按同一 LHS 粒度）：若关键词该 LHS 的 RHS“最终段”为纯角度/π
            # （例如 A=B=55° 的最后一段是 55°），则允许步骤在该 LHS 直接给出纯数值角度/π
            def _rhs_last_segment(t: str) -> str:
                parts = [p.strip() for p in (t or '').split('=') if p.strip()]
                return parts[-1] if parts else (t or '')
            key_rhs_is_plain_deg_or_pi_for_this_lhs = _is_plain_degree_or_pi_quick(_rhs_last_segment(rk))
            if ((_rhs_has_symbolic_var(rk) and _is_plain_degree_or_pi_quick(rs)) or
                (_rhs_has_symbolic_var(rs) and _is_plain_degree_or_pi_quick(rk))):
                if not key_rhs_is_plain_deg_or_pi_for_this_lhs:
                    print("==> 同一 LHS 下出现“含变量角度表达式”与“纯数值角度/π”的冲突，判为不匹配。")
                    return False

    if any(op in key_rel for op in ['=', '<', '>', '≤', '≥']):
        # 在进行软匹配前，若 key 与 step 的等式 LHS 都出现了同一中心的三字母角名，
        # 则要求三字母严格一致（允许首末对调）。若不一致，直接 False，避免不同角被当作同一中心变量放宽匹配。
        try:
            def _extract_lhs_triads_by_center(s: str) -> dict[str, set[str]]:
                buckets = defaultdict(set)
                for lm in re.finditer(r'([^\n=<>≤≥]+?)\s*=\s*[^\n=]+', s):
                    lhs_seg = lm.group(1)
                    for m in re.finditer(r'(?:\\angle|∠|angle)?\s*([A-Za-z])([A-Za-z])([A-Za-z])', lhs_seg):
                        a, b, c = m.group(1).upper(), m.group(2).upper(), m.group(3).upper()
                        buckets[b].add(a + b + c)
                return buckets
            key_triads0  = _extract_lhs_triads_by_center(key_rel)
            step_triads0 = _extract_lhs_triads_by_center(step_rel)
            for cen, triads_key in key_triads0.items():
                if cen in step_triads0 and step_triads0[cen]:
                    allowed = set()
                    for t in triads_key:
                        allowed.add(t)
                        allowed.add(t[::-1])
                    # 只要 step 在该中心出现了任一三字母角名不在 allowed，则视为不匹配
                    if not any(t in allowed for t in step_triads0[cen]):
                        print("==> 发现同一中心字母的三字母角名不一致（软匹配前预检），判为不匹配。")
                        return False
        except Exception:
            pass

        # 新增：关键词为“含数值的连等式”，而步骤未出现任何数值等式 -> 不匹配
        if key_rel_stripped.count('=') >= 2:
            num_rhs_pat = re.compile(r'=\s*(?:[-+]?\d+(?:\.\d+)?|\d+\s*/\s*\d+|[-+]?\d+(?:\.\d+)?\s*°|(?:\\pi|π|pi))', re.IGNORECASE)
            key_has_numeric_rhs = bool(num_rhs_pat.search(key_rel_stripped))
            step_has_numeric_rhs = bool(num_rhs_pat.search(step_rel_stripped))
            if key_has_numeric_rhs and not step_has_numeric_rhs:
                print("==> 关键词为含数值的连等式，但步骤未出现任何数值等式，判为不匹配。")
                return False
        # 优先尝试等式软匹配（方向无关），用于快速通过结构一致的等式（如单位/空格差异）
        if ('=' in key_rel_stripped) and ('=' in step_rel_stripped):
            if _soft_match_equalities(key_rel_stripped, step_rel_stripped):
                return True
        # 用统一的关系运算符正则切分出"项"：
        # 例：A=B=C=30 -> ['A','B','C','30']   x<y<30 -> ['x','y','30']
        parts = re.split(r'(?:<=|>=|=|<|>|≤|≥)', key_rel)
        parts = [p.strip() for p in parts if p.strip()]
        if len(parts) >= 2:
            # 左侧项为除了最后一项以外的所有项
            lhs_list = parts[:-1]

            # 规范化（与系统既有 canonical 一致），并过滤掉纯数字/分数（防止 '30' 当作 LHS）
            step_can = _canonical_form(step_rel)
            def _is_pure_number(x: str) -> bool:
                x = x.strip()
                return bool(re.fullmatch(r'[-+]?\d+(?:\.\d+)?', x) or re.fullmatch(r'\d+\s*/\s*\d+', x))
            
            hit = False  # ←← 默认值，防止 lhs_cans 为空时未定义

            # 仅从每个左侧段中提取"最后一个数学标识符"作为 LHS（去掉自然语言前缀）
            # 支持：angle/∠/\\angle 角记号，纯变量/带下标变量（y, y1, x_2），以及简短几何点名（AB 等）
            def _last_math_token(seg: str) -> str:
                seg = seg.strip()
                angle_pat = r'(?:\\angle|∠|angle)\s*[A-Za-z]{1,3}'
                ident_pat = r'[A-Za-z][A-Za-z0-9_]*|[A-Za-z]{2}'
                tok_pat = rf'(?:{angle_pat}|{ident_pat})'
                toks = re.findall(tok_pat, seg, flags=re.IGNORECASE)
                return toks[-1] if toks else seg

            # —— 新增：为 LHS 生成别名集合（angle AOB ↔ AOB ↔ angle o）——
            def _lhs_aliases_from_token(raw_token: str) -> set[str]:
                t = raw_token.strip()
                aliases = set()
                m = re.match(r'(?:\\angle|∠|angle)\s*([A-Za-z])([A-Za-z])([A-Za-z])$', t)
                if m:
                    a,b,c = m.group(1).lower(), m.group(2).lower(), m.group(3).lower()
                    aliases.update({_canonical_form(f'{a}{b}{c}'), _canonical_form(f'angle{b}')})
                    return aliases
                m = re.match(r'^([A-Za-z])([A-Za-z])([A-Za-z])$', t)
                if m:
                    a,b,c = m.group(1).lower(), m.group(2).lower(), m.group(3).lower()
                    aliases.update({_canonical_form(f'{a}{b}{c}'), _canonical_form(f'angle{b}')})
                    return aliases
                m = re.match(r'(?:\\angle|∠|angle)\s*([A-Za-z])$', t)
                if m:
                    b = m.group(1).lower()
                    aliases.add(_canonical_form(f'angle{b}'))
                    return aliases
                aliases.add(_canonical_form(t))
                return aliases
            
            lhs_cans = set()
            for t in lhs_list:
                token = _last_math_token(t)
                for cand in _lhs_aliases_from_token(token):
                    if cand and not _is_pure_number(cand):
                        lhs_cans.add(cand)

            # 要求：步骤文本中必须出现至少一个 LHS 规范串（伪词界，避免 y 命中 dy）
            if lhs_cans:
                # ---- 1) canonical 命中 ----
                for lc in lhs_cans:
                    if re.search(rf'(?<![a-z0-9_]){re.escape(lc)}(?![a-z0-9_])', step_can):
                        hit = True
                        break
                
                # ---- 2.0)【新增强约束】若 key 的 LHS 有"三字母角名"，而 step 也出现了同中心的"三字母角名"，
                #           则要求三字母完全一致（允许首末对调），否则直接判 False。
                #           （只在"pred 使用三字母角名时收紧"；若 pred 只有 ∠C 这种单字母，不触发此强约束）
                # 收集 key 的三字母角名（按中心字母分桶）
                key_triads_by_center = defaultdict(set)
                triad_pat1 = re.compile(r'(?:\\angle|∠|angle)\s*([A-Za-z])([A-Za-z])([A-Za-z])$')
                triad_pat2 = re.compile(r'^([A-Za-z])([A-Za-z])([A-Za-z])$')
                for seg in lhs_list:
                    tok = _last_math_token(seg)
                    m = triad_pat1.match(tok) or triad_pat2.match(tok)
                    if m:
                        a, b, c = m.group(1).upper(), m.group(2).upper(), m.group(3).upper()
                        key_triads_by_center[b.lower()].add(a + b + c)

                # 仅当 step 里真的出现"同中心的三字母角名"时才触发严格比对
                if key_triads_by_center:
                    # 仅在步骤“等式的左侧”提取三字母角名，避免 RHS 中无关三字母触发误判
                    step_triads = []
                    # 抽取所有等式的 LHS 片段
                    for lm in re.finditer(r'([^\n=<>≤≥]+?)\s*=\s*[^\n=]+', step_norm):
                        lhs_seg = lm.group(1)
                        for m in re.finditer(r'(?:\\angle|∠|angle)?\s*([A-Z])([A-Z])([A-Z])', lhs_seg):
                            sa, sb, sc = m.group(1), m.group(2), m.group(3)
                            step_triads.append(sa + sb + sc)

                    # 逐中心检查：如果 step 在该中心出现了三字母角名，则必须与 key 的三字母匹配（或首末对调）
                    for cen_lower, triads in key_triads_by_center.items():
                        cen_upper = cen_lower.upper()
                        # step 里同中心的所有三字母
                        step_same_center = [t for t in step_triads if len(t) == 3 and t[1] == cen_upper]
                        if not step_same_center:
                            continue  # step 没用三字母，严格约束不触发（后面的宽松桥接仍可生效）

                        # 允许的集合 = key 的三字母 + 其逆序
                        allowed = set()
                        for t in triads:
                            allowed.add(t)
                            allowed.add(t[::-1])  # 允许 ACD ↔ DCA

                        # 只要 step 的任意一个同中心三字母不在 allowed，就直接判 False
                        # （如果 step 给了多个不同三字母，但只要都不等于 allowed 里的任何一个，也应 False）
                        if not any(t in allowed for t in step_same_center):
                            print("==> 发现同一中心字母的三字母角名不一致：关键词要求",
                                  triads, "；步骤给出", step_same_center, "。判为不匹配。")
                            return False
                def _loose_find_lhs_in_step(step_text_raw: str, token: str) -> bool:
                    """
                    在原始 step 文本里宽松查找 LHS token：
                    - 不用 canonical 后的“词界”
                    - 两字母点名允许中间空格（B C / B\s*C）
                    - 允许紧跟关系符（=、≈、:）
                    """
                    if not step_text_raw or not token:
                        return False
                    txt = step_text_raw
                    tok = token

                    # 大小写不敏感
                    flags = re.IGNORECASE

                    # 1) 原样词界（带可选空格），例如 \bBC\b 或 \bBC\s*=\s*
                    pat1 = rf'(?<![A-Za-z0-9_]){re.escape(tok)}(?![A-Za-z0-9_])'
                    if re.search(pat1, txt, flags=flags):
                        return True
                    pat1_eq = rf'(?<![A-Za-z0-9_]){re.escape(tok)}\s*(?:=|≈|:)\s*'
                    if re.search(pat1_eq, txt, flags=flags):
                        return True

                    # 2) 两字母几何点名：允许中间空格（B\s*C），同时也允许后接关系符
                    if re.fullmatch(r'[A-Za-z]{2}', tok, flags=flags):
                        spaced = rf'{tok[0]}\s*{tok[1]}'
                        pat2 = rf'(?<![A-Za-z0-9_]){spaced}(?![A-Za-z0-9_])'
                        if re.search(pat2, txt, flags=flags):
                            return True
                        pat2_eq = rf'(?<![A-Za-z0-9_]){spaced}\s*(?:=|≈|:)\s*'
                        if re.search(pat2_eq, txt, flags=flags):
                            return True

                    # 3) 最后兜底：只要出现 token 并且任一侧邻接 '=' 也通过
                    around_eq = rf'(?:{re.escape(tok)}\s*=|=\s*{re.escape(tok)})'
                    if re.search(around_eq, txt, flags=flags):
                        return True

                    return False

                # ---- 2) angleX ↔ 三字母大写角名 (中心字母 X) ----
                if not hit:
                    angle_centers = set()
                    for lc in lhs_cans:
                        m = re.fullmatch(r'angle([a-z])', lc)
                        if m:
                            angle_centers.add(m.group(1))
                    if angle_centers:
                        for center in list(angle_centers):
                            cu = center.upper()
                            # 例如 A C D（中心 C）
                            pattern = r'\b[A-Z]' + cu + r'[A-Z]\b'
                            if re.search(pattern, step_norm):
                                hit = True
                                break

                # ---- 3) angleX ↔ "X = 55°" ----
                if not hit and angle_centers:
                    for center in list(angle_centers):
                        if re.search(
                            rf'(?<![A-Za-z0-9_]){re.escape(center)}\s*=\s*[-+]?\d+(?:\.\d+)?\s*°',
                            step_norm, flags=re.IGNORECASE
                        ):
                            hit = True
                            break
                        if re.search(
                            rf'[-+]?\d+(?:\.\d+)?\s*°\s*=\s*(?<![A-Za-z0-9_]){re.escape(center)}(?![A-Za-z0-9_])',
                            step_norm, flags=re.IGNORECASE
                        ):
                            hit = True
                            break

                # ---- 4) 二次宽松匹配：在未 canonical 的 step 文本里再找一次 LHS ----
                if not hit:
                    # 用原始（已 rescue）文本，避免 canonical 抹掉词界
                    step_raw_for_loose = step_rel  # 已过 _normalize_relops + _rescue_latex_escapes
                    for lc in lhs_cans:
                        # 原 token（如 'bc'）转回大写/原样并做一次宽松找
                        raw_tok = lc
                        # canonical 会小写；这里不强制大小写，loose 正则已 IGNORECASE
                        if _loose_find_lhs_in_step(step_raw_for_loose, raw_tok):
                            hit = True
                            break

                if not hit:  # ←← 只有 lhs_cans 非空时才强制出现 LHS
                    print(f"==> 关系式关键词 {key_norm!r} 的任一左侧项未在步骤文本中出现，仅出现右侧/数值将被拒绝。")
                    return False
        # === 新增护栏：关键词是"带系数的比例等式"，而步骤文本写成了"简单相等"，则判 False ===
        # 例如： key = "∠2 = 1/2 ∠1" ，而 step = "∠2 = ∠1"
        def _strip_angle_signs(token: str) -> str:
            # 去掉 ∠ / \angle / angle 等记号与多余空白，仅保留字母数字下划线
            t = re.sub(r'(\\angle|∠|angle)\s*', '', token, flags=re.IGNORECASE).strip()
            t = re.sub(r'[^A-Za-z0-9_]+', '', t)
            return t.lower()

        # 只在 key 是"lhs = (coef) * rhs"的形式且 coef != 1 时生效；rhs 不应是纯数字(如 55°)
        _coef_pat = re.compile(
            r'^\s*(?P<lhs>(?:\\angle|∠|angle)?\s*[^=<>]+?)\s*=\s*'
            r'(?P<coef>(?:\d+\s*/\s*\d+|\d+(?:\.\d+)?))\s*(?:\*|×)?\s*'
            r'(?P<rhs>(?:\\angle|∠|angle)?\s*[^=<>]+?)\s*$',
            flags=re.IGNORECASE
        )
        m_coef = _coef_pat.match(key_rel)
        if m_coef:
            lhs_tok = _strip_angle_signs(m_coef.group('lhs'))
            rhs_tok = _strip_angle_signs(m_coef.group('rhs'))
            coef_str = m_coef.group('coef').replace(' ', '')
            # 解析系数
            try:
                coef_val = float(Fraction(coef_str)) if '/' in coef_str else float(coef_str)
            except Exception:
                coef_val = None

            # rhs 若是纯数值/带度数的数字，则不视作"比例关系"，跳过此护栏
            rhs_is_pure_num = bool(re.fullmatch(r'[-+]?\d+(?:\.\d+)?', m_coef.group('rhs')))
            rhs_has_degree = '°' in m_coef.group('rhs')

            if (coef_val is not None) and (abs(coef_val - 1.0) > 1e-9) and (not rhs_is_pure_num) and (not rhs_has_degree):
                # 检查步骤文本是否断言"lhs = rhs"或"rhs = lhs"（不带系数）
                step_simple_eq = False
                # 统一去角标记再比对
                step_can = _normalize_symbols(step_rel)
                # 构造两种无系数的等式正则（伪词界）
                pat1 = rf'(?<![A-Za-z0-9_]){re.escape(lhs_tok)}\s*=\s*{re.escape(rhs_tok)}(?![A-Za-z0-9_])'
                pat2 = rf'(?<![A-Za-z0-9_]){re.escape(rhs_tok)}\s*=\s*{re.escape(lhs_tok)}(?![A-Za-z0-9_])'
                if re.search(pat1, _canonical_form(step_can)) or re.search(pat2, _canonical_form(step_can)):
                    step_simple_eq = True

                if step_simple_eq:
                    # 关键词要求"成比例"，但步骤写成"相等"，直接判不匹配
                    print("==> 比例关键字（系数≠1）在步骤中被简化成了相等关系，视为不匹配。")
                    return False

    # === HARD GUARD: 同一 LHS 时，"系数×(度/π)" vs "纯(度/π)" => 直接不匹配 ===
    def _lhs_rhs_pairs(s: str):
        """
        抽取 (LHS_key, RHS_raw) 对：
        - LHS_key 优先用 _angle_lhs_key 精准归一（angle AOB/∠AOB/angle B -> angle o）
        - 若无法识别角或简短标识，再退回 _canonical_form(l)
        """
        s2 = _normalize_symbols(_rescue_latex_escapes(s))
        eqs = re.findall(r'([^\n=<>≤≥]+?)\s*=\s*([^\n=]+)', s2)
        out = []
        for l, r in eqs:
            lhs_key = _angle_lhs_key(l)
            if not lhs_key:
                lhs_key = _canonical_form(l)
            out.append((lhs_key, r.strip()))
        return out

    def _has_scaled_degree_or_pi(t: str) -> bool:
        u = _rescue_latex_escapes(t or "")
        return bool(
            re.search(r'[-+]?\d+(?:\.\d+)?\s*(?:\*|×)\s*[-+]?\d+(?:\.\d+)?\s*°', u) or
            re.search(r'[-+]?\d+(?:\.\d+)?\s*(?:\*|×)\s*(?:\\pi|π|pi)\b', u, flags=re.IGNORECASE)
        )

    def _is_plain_degree_or_pi(t: str) -> bool:
        u = _rescue_latex_escapes(t or "")
        return bool(
            re.fullmatch(r'\s*[-+]?\d+(?:\.\d+)?\s*°\s*', u) or
            re.fullmatch(r'\s*(?:\\pi|π|pi)\s*', u, flags=re.IGNORECASE)
        )

    # 计算 key 与 step 中所有 (LHS_can, RHS) 对
    pairs_k = dict(_lhs_rhs_pairs(key_rel))
    pairs_s = dict(_lhs_rhs_pairs(step_rel))

    # 逐一比对相同 LHS 的 RHS 形态
    for lk, rk in pairs_k.items():
        if lk in pairs_s:
            rs = pairs_s[lk]
            conflict = (
                (_has_scaled_degree_or_pi(rk) and _is_plain_degree_or_pi(rs)) or
                (_has_scaled_degree_or_pi(rs) and _is_plain_degree_or_pi(rk))
            )
            if conflict:
                print("==> 同一 LHS 下出现\"系数×(度/π)\"与\"纯(度/π)\"的冲突，判为不匹配。")
                return False

    # ===== 0) 强护栏：key 是 \sqrt{n}，而 step 里没有根号，只出现"裸 n" => 直接不匹配 =====
    rad = _extract_sqrt_radicand_if_literal(key_norm)
    if rad is not None and not _is_radical_expr(step_norm):
        txt = step_norm.replace('⁄', '/').replace(',', '')
        tokens = _NUM_TOKEN_RE.findall(txt)
        def _as_float(x):
            try:
                return float(Fraction(x)) if '/' in x else float(x)
            except Exception:
                return None
        try:
            r = float(rad)
        except Exception:
            r = None
        if r is not None and any(v is not None and abs(v - r) < 1e-12 for v in map(_as_float, tokens)):
            print(f"==> 关键词是根号 {rad}，但步骤中没有根号表达式，仅出现了等值的裸数字/分数，视为不匹配。")
            return False

    # # 大枚举直接拦（用规范化后的 step_norm）
    # if _looks_like_enum_of_answers(step_norm):
    #     print("==> 步骤文本呈现多个候选答案，疑似枚举猜答案行为，视为不匹配。")
    #     return False

    # 用规范化后的 step_norm 来抓候选；这样 √3 已经变成 \sqrt{3} 能被正则识别
    cand_pat = r"""
        (\\boxed\{[^}]+\}
        |\\frac\{[^{}]+\}\{[^{}]+\}
        |\\sqrt(?:\[[^\]]+\])?\{[^{}]+\}
        |\b\d+\s*/\s*\d+\b
        |\b[-+]?\d+(?:\.\d+)?\s*(?:°|(?:pi|\\pi|π))?\b
        |[A-Za-z][A-Za-z0-9_]*\s*=\s*[^=\n]+(?:=\s*[^=\n]+)*)
    """
    cands = [c.strip() for c in re.findall(cand_pat, step_norm, flags=re.VERBOSE)]

    # 若 key 是 \sqrt{n}，跳过等于 n 的"裸数字/分数"候选
    rad = _extract_sqrt_radicand_if_literal(key_norm)
    skipped_radicand = False
        
    for c in cands:
        if rad is not None:
            if re.fullmatch(r'[-+]?\d+(?:\.\d+)?', c) or re.fullmatch(r'\d+\s*/\s*\d+', c):
                try:
                    cval = float(Fraction(c)) if '/' in c else float(c)
                    if abs(cval - float(rad)) < 1e-12:
                        skipped_radicand = True
                        continue
                except Exception:
                    pass
        if grade_answer(c, key_norm):
            return True

    if skipped_radicand:
        print(f"==> 关键词是根号 {rad}，但步骤中没有根号表达式，仅出现了等值的裸数字/分数，视为不匹配。")

    # 任一端是 LaTeX : 走鲁棒匹配（归一化 + 等式软匹配）
    if _looks_like_latex(key_norm) or _looks_like_latex(step_norm):
        if _match_key_in_text(key_norm, step_norm):
            return True

    core = _extract_math_core(step_norm) or step_norm
    if grade_answer(core, key_norm):
        return True
    else:
        print("==> 关键词与步骤文本均未匹配成功。")
        return False


def _match_key_in_text(key_str: str, content: str) -> bool:
    """
    用与 grade_answer 同风格的"强鲁棒"匹配逻辑，判断 key_str 是否能在 content 中被体现。
    顺序：先做符号统一(含π)、然后等式软匹配、再数值/LaTeX求值兜底、最后规范形包含。
    """
    if not key_str or not content:
        return False

    if grade_answer(content, key_str):
        return True

    # 1) rescue + 归一化到可比字符串
    k_raw = _rescue_latex_escapes(key_str)
    c_raw = _rescue_latex_escapes(content)

    k_norm = _latex_to_python_expr(_normalize_symbols(k_raw))
    c_norm = _latex_to_python_expr(_normalize_symbols(c_raw))

    # 1a) 直接子串包含（适配 [846/376=k^2]、\[...\]、\boxed{...} 归一化后的情况）
    if k_norm and k_norm in c_norm:
        return True

    # 2) 等式软匹配（含连续等式）
    if '=' in k_norm or '=' in c_norm:
        if _soft_match_equalities(k_norm, c_norm):
            return True

    # 3) 数值比较（覆盖 "latex vs 字面数字/分数/近似π"）
    k_num = _to_float_general(k_raw)
    c_num = _to_float_general(c_raw)
    # —— 新增护栏：只有在"两边均为纯数值短语（不含字母/变量/等号）"时，才启用数值兜底 —— #
    def _looks_pure_numeric_phrase(t: str) -> bool:
        if not isinstance(t, str):
            return False
        # 去掉不会构成变量的符号
        t2 = t
        t2 = re.sub(r'(\\pi|π|兀|°)', '', t2)   # 保留数值含义的符号但不把它们当字母
        # 不能含等号
        if '=' in t2:
            return False
        # 不能含任何字母（变量/函数名等）
        return not re.search(r'[A-Za-z]', t2)

    if (_looks_pure_numeric_phrase(k_raw) and _looks_pure_numeric_phrase(c_raw)
            and (k_num is not None) and (c_num is not None)):
        if abs(k_num - c_num) < 1e-3:
            return True

    # 4) LaTeX 求值比较（再兜底一次）
    k_val = _try_eval_latex_numeric(k_norm)
    c_val = _try_eval_latex_numeric(c_norm)
    if (k_val is not None) and (c_val is not None):
        if abs(k_val - c_val) < 1e-6:
            return True

    # 5) 规范形包含的最后兜底（极宽松）
    if _canonical_form(k_raw) in _canonical_form(c_raw):
        return True

    return False

def _unify_pi_approx(text: str) -> str:
    """把各种 π 记法统一成 'pi'；
    也把裸露的小数（如 3.14/3.14159）在容差内吸附为 'pi'。不动 2π/0.5π（需要时可扩展）。"""
    if not isinstance(text, str) or not text:
        return text
    t = _rescue_latex_escapes(text)

    # 先把符号/LaTeX 统一为 'pi'
    t = re.sub(r'(?<!\w)(?:\\pi|π|兀|pi)(?!\w)', 'pi', t)

    # 再把近似 π 的"裸数字"统一成 'pi'（默认 ±0.005）
    def _pi_repl(m):
        s = m.group(0)
        try:
            v = float(s)
            if abs(v - math.pi) <= 5e-3:   # 可按需调小/调大
                return 'pi'
        except Exception:
            pass
        return s

    # 只匹配独立数字 token，避免粘连到单位/变量名
    t = re.sub(r'(?<![A-Za-z_])\d+(?:\.\d+)?(?![A-Za-z_])', _pi_repl, t)
    return t

def _safe_eval(expr: str):
    tree = ast.parse(expr, mode='eval')
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(f"Disallowed AST node: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id not in _ALLOWED_NAMES:
            raise ValueError(f"Disallowed name: {node.id}")
        if isinstance(node, ast.Call):
            fn = node.func
            if not isinstance(fn, ast.Name) or fn.id not in _ALLOWED_NAMES:
                raise ValueError("Disallowed call")
    return eval(compile(tree, "<safe_eval>", "eval"), {"__builtins__": {}}, _ALLOWED_NAMES)

def _to_float_general(s: str):
    """尽力把字符串 s 转成浮点数：
       1) 先用 LaTeX 求值；
       2) 抽核心数学片段后再尝试 float；
       3) 尝试 Fraction；
       都失败则返回 None。
    """
    if not s:
        return None

    # === 新增：优先把 'x°' 解析为 x，本地单位匹配而不做弧度换算 ===
    txt0 = _rescue_latex_escapes(s)
    m = re.findall(r'([-+]?\d+(?:\.\d+)?)\s*°', txt0)
    if m:
        try:
            # 取最后一个度数数字（通常就是结论处的那个）
            return float(m[-1])
        except Exception:
            pass
    
    # === 新增：比值 'a:b' / 'a∶b' / 'a：b' 解析为 a/b ===
    # 仅当整串就是一个比值时才触发，避免误伤含有冒号的句子
    ratio_pat = re.compile(
        r'^\s*([-+]?\d+(?:\.\d+)?)\s*[:∶：]\s*([-+]?\d+(?:\.\d+)?)\s*$'
    )
    rm = ratio_pat.match(_rescue_latex_escapes(s))
    if rm:
        try:
            num = float(rm.group(1))
            den = float(rm.group(2))
            if den != 0:
                return num / den
        except Exception:
            pass


    # 1) LaTeX 数值（内部会 rescue）
    v = _try_eval_latex_numeric(s)
    if v is not None:
        return v

     # === 关键护栏：若包含根号而 LaTeX 求值失败，避免退回"捞最后一个数字 token"导致把 sqrt(3) 误当 3 ===
    txt_guard = _rescue_latex_escapes(s)
    if re.search(r'(\\sqrt|√)', txt_guard):
       return None

    # === 新增：如果文本包含 π/\\pi/pi，且没有任何显式数字可抓，直接按 π 返回 ===
    # 这里先快速判定是否含 π 记号
    if re.search(r'(?<!\w)(?:\\pi|π|兀|pi)(?!\w)', txt_guard):
        # 再看有没有数字 token；如果没有，就当作单独的 π
        tmp = txt_guard.replace(',', '').replace('⁄', '/')
        if not _NUM_TOKEN_RE.search(tmp):
            return math.pi

    # 2) 自然语言里抓"最后一个"数字/分数 token（通常就是答案）
    txt = _rescue_latex_escapes(s)
    txt = txt.replace(',', '')         # 容忍千分位
    txt = txt.replace('⁄', '/')        # 兼容 Unicode 分数斜杠step
    tokens = _NUM_TOKEN_RE.findall(txt)
    if tokens:
        candidate = tokens[-1].strip()
        # 2a) 能直接转 float 就转 float
        try:
            return float(candidate)
        except Exception:
            pass
        # 2b) 带有 / 就走 Fraction
        try:
            if '/' in candidate:
                return float(Fraction(candidate))
        except Exception:
            pass
    
    # 3) 抽核心
    core = _extract_math_core(txt) or txt

    # 有时核心仍可能是 '(3)/(2)' 这类，先用 _latex_to_python_expr 规范化一下
    core_expr = _latex_to_python_expr(_rescue_latex_escapes(core))

    # 2a) 直接 float
    try:
        return float(core_expr)
    except Exception:
        pass

    # 2b) 分数 a/b
    try:
        if '/' in core_expr:
            return float(Fraction(core_expr))
    except Exception:
        pass

    return None

def _rescue_latex_escapes(t: str) -> str:
    if not t:
        return t

    t = re.sub(r'√(?![{])\s*(\d+(?:\.\d+)?|[a-zA-Z])', r'\\sqrt{\1}', t)
    t = t.replace('√', '\\sqrt')
    # \x0c -> \f: 这里直接把 "\x0crac" 一把替换成 "\\frac"
    t = t.replace('\x0crac', '\\frac')
    t = t.replace('\\x0crac', '\\frac')
    # 兜底：如果还有裸的 \x0c + 字母（极少），保底替成 \f
    t = re.sub(r'\x0c(?=[A-Za-z])', r'\\f', t)

    # \t(tab) 相关（\times, \tan, \tfrac）
    t = re.sub(r'\tfrac', r'\\tfrac', t)
    t = re.sub(r'\times', r'\\times', t)
    t = re.sub(r'\tan',   r'\\tan',   t)
    t = re.sub(r'\t(?=[A-Za-z])', r'\\t', t)   # 兜底

    # \r(carriage return) 相关（\rangle）
    t = re.sub(r'\rangle', r'\\rangle', t)
    t = re.sub(r'\r(?=[A-Za-z])', r'\\r', t)   # 兜底
    return t

# ========= LaTeX → Python 表达式规范化 =========
_LATEX_SPACE = re.compile(r'\\[,\s;:!]+')   # \, \; \: \! 等
def _latex_to_python_expr(s: str) -> str:
    if not s: 
        return s
    t = _rescue_latex_escapes(s)
    t = t.replace('√', '\\sqrt')  # 先把 √ 替成 \sqrt，方便后续处理

    t = t.strip()

    # 去 LaTeX 包裹符
    t = t.replace(r'\left', '').replace(r'\right', '')
    t = re.sub(r'^\$|\$$', '', t)                       # $...$
    t = re.sub(r'\\\(|\\\)|\\\[|\\\]', '', t)           # \(...\), \[...\]

    # 常用替换
    t = t.replace(r'\cdot', '*').replace(r'\times', '*').replace(r'\div', '/')
    t = t.replace(r'\mathrm{e}', 'e').replace(r'\operatorname{e}', 'e')
    t = t.replace(r'\ln', 'ln').replace(r'\log', 'log')
    t = t.replace('−', '-').replace('—', '-')           # 统一负号
    t = _LATEX_SPACE.sub('', t)                         # 去空白命令

    # 度数：^\circ 或 \degree / ^\circC 等
    t = re.sub(r'\^\s*\\circ', '°', t)
    t = re.sub(r'\\degree', '°', t)

    # {a \over b} → (a)/(b)
    t = re.sub(r'\{([^{}]+)\s*\\over\s*([^{}]+)\}', r'(\1)/(\2)', t)

    # \frac{num}{den} 族系（包含 \dfrac \tfrac \cfrac）
    # 递归替换，直到不再出现
    while re.search(r'(?:\\|\x0c)[dtc]?frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}', t):
        t = re.sub(r'(?:\\|\x0c)[dtc]?frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}', r'(\1)/(\2)', t)

    # \sqrt[n]{x} 和 \sqrt{x}
    # 先 n 次方根
    t = re.sub(r'\\sqrt\[(.*?)\]\{(.*?)\}', r'(\2)**(1/(\1))', t)
    t = re.sub(r'\\sqrt\{(.*?)\}', r'(\1)**0.5', t)

    # 三角函数（粗略）：\sin x → sin(x)（若后面不是括号，用一个小括号包前一个 token）
    # 为简洁，仅做常见括号情形；无括号则不强制封装
    t = t.replace(r'\sin', 'sin').replace(r'\cos', 'cos').replace(r'\tan', 'tan')

    # 指数：^ → **  （先处理度数后再替换^）
    t = t.replace('^', '**')

    # \pi
    t = t.replace(r'\pi', 'pi')

    # # 度数转弧度： 90° → (90*pi/180)
    # t = re.sub(r'(?P<num>\d+(\.\d+)?)\s*°', r'(\g<num>*pi/180)', t)

    # 清理多余空格
    t = re.sub(r'\s+', '', t)

    # 数字或 ')' 后面紧跟 sqrt/pi/字母/ '(' ：补 *
    t = re.sub(r'(?<=\d)\s*(?=(\\sqrt|\\pi|π|pi|[A-Za-z(]))', '*', t)
    t = re.sub(r'(?<=\))\s*(?=(\\sqrt|\\pi|π|pi|[A-Za-z(]))', '*', t)
    # 变量后面直接跟 sqrt 或 '(' ：也补 *
    t = re.sub(r'(?<=[A-Za-z])\s*(?=(\\sqrt|\())', '*', t)

    return t

def _looks_like_latex(s: str) -> bool:
    """快速判定字符串里是否含 LaTeX 记号。"""
    if not s:
        return False
    # 常见 LaTeX 宏/环境/控制序列
    return bool(re.search(
        r'(\\[a-zA-Z]+|\\[()\[\]]|\$.*?\$|\\frac|\\sqrt|\\pi|\\times|\\div|\\cdot|\\left|\\right|\\over|\\degree|\\angle)',
        s, re.DOTALL
    ))

def _read_balanced_braces(text: str, open_idx: int):
    """给定 text 中某个 '{' 的索引，返回与之配对的内容和右括号索引。
       成功 -> (inside, end_idx)；失败 -> (None, None)
    """
    if open_idx < 0 or open_idx >= len(text) or text[open_idx] != '{':
        return None, None
    depth = 1
    i = open_idx + 1
    start = i
    n = len(text)
    while i < n:
        c = text[i]
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                return text[start:i], i
        i += 1
    return None, None

def _extract_math_core(s: str):
    """从混合文本中抽出最可能的数学表达式片段（支持 \boxed 的嵌套花括号）。"""
    if not s:
        return None
    t = _rescue_latex_escapes(s)
    # print(f"DEBUG _extract_math_core input t: {repr(t)}") # 添加这一行
    
    # 1) \boxed{...}
    m = re.search(r'\\boxed\s*\{', t)
    if m:
        open_idx = m.end() - 1
        inside, _ = _read_balanced_braces(t, open_idx)
        if inside is not None:
            return inside.strip()

    # 2) \( ... \) / \[ ... \] / $...$
    m = re.search(r'\\\((.*?)\\\)|\\\[(.*?)\\\]|\$(.+?)\$', t, re.DOTALL)
    if m:
        for g in m.groups():
            if g:
                return g.strip()

    # 3) 含等号的等式
    m = re.search(r'([^\n=]*?(?:\\[a-zA-Z]+|[0-9πpi\./\*\+\-\^\(\) ])+[^\n=]*=[^\n]+)', t)
    if m:
        return m.group(1).strip()

    # 4) \frac / \dfrac / \tfrac / \cfrac
    m = re.search(r'(?:\\|\x0c)[dtc]?frac\s*\{[^{}]+\}\s*\{[^{}]+\}', t)
    if m:
        return m.group(0)

    # 5) \sqrt[...]{} / \sqrt{}
    m = re.search(r'\\sqrt(?:\[[^\]]+\])?\{[^}]+\}', t)
    if m:
        return m.group(0)

    # 6) 数字×π 或 角度
    m = re.search(r'\d+(?:\.\d+)?\s*(?:\\pi|π|pi)\b', t, re.IGNORECASE)
    if m:
        return m.group(0)
    m = re.search(r'\d+(?:\.\d+)?\s*°', t)
    if m:
        return m.group(0)

    # 7) 纯分数 a/b
    m = re.search(r'\b\d+\s*/\s*\d+\b', t)
    if m:
        return m.group(0)

    # 8) 纯数字或紧跟着字母数字的数字序列（不强制单词边界）
    # 这是最关键的修改，确保能匹配 '4.1m' 中的 '4.1' 或 '4.1m' 整体
    # 只要是数字，就尽可能匹配，不让后面的字母阻碍
    m = re.search(r'[-+]?\d+(?:\.\d+)?', t) 
    if m:
        # print(f"DEBUG _extract_math_core returns (num_no_boundary): {repr(m.group(0))}")
        return m.group(0)

    # print(f"DEBUG _extract_math_core returns (None): {repr(None)}")
    return None

def _try_eval_latex_numeric(s: str):
    if not s:
        return None
    rescued = _rescue_latex_escapes(s)

    # 1) 先对"整串"做 LaTeX->Python 并尝试安全求值（没有等号才评估）
    expr_full = _latex_to_python_expr(rescued)
    if '=' not in expr_full:
        try:
            val = _safe_eval(expr_full)
            if isinstance(val, (int, float)):
                return float(val)
        except Exception:
            pass  # 如果整串失败，再走核心片段兜底

    # 2) 兜底：再提取核心片段求值（老逻辑）
    core = _extract_math_core(rescued) or rescued
    expr = _latex_to_python_expr(core)
    if '=' in expr:
        return None
    try:
        val = _safe_eval(expr)
        return float(val) if isinstance(val, (int, float)) else None
    except Exception:
        return None

# 实现答案中π和角度的软匹配处理
def _normalize_symbols(text: str) -> str:
    res = _unify_pi_approx(text).lower()

    # —— 单位归一（数字 + 单位，可有/无空格；先长后短）——
    unit_patterns = [
        (r'(千米|公里|kilometers?|km)', 'km'),
        (r'(厘米|centimeters?|cm)',     'cm'),
        (r'(毫米|millimeters?|mm)',     'mm'),
        (r'(分米|decimeters?|dm)',      'dm'),
        (r'(米|meters?|meter|m)',       'm'),
    ]
    for pat, std in unit_patterns:
        # 情形1：数字后紧跟或空格后跟单位：3.5米 / 3.5 m / 3/2米
        res = re.sub(fr'(?<=\d)\s*(?:{pat})(?![a-z])', std, res)
        # 情形2：独立出现的单位词（句子里）：... 以 米 为单位 ...
        # 中文对 \b 不稳定，但英文这条能兜底；主要依靠上面的"数字后紧跟"规则解决关键场景
        res = re.sub(fr'\b(?:{pat})\b', std, res)

    # —— 其他符号归一 —— 
    res = re.sub(r'\b(度|degree|degrees)\b', '°', res)
    res = re.sub(r'\b(和|&)\b', ' and ', res)
    res = re.sub(r'\b(或)\b', ' or ', res)
    res = re.sub(r'm\^?2|m²', 'm2', res)
    res = re.sub(r'cm\^?2|cm²', 'cm2', res)
    res = re.sub(r'\b(hour|hours|小时)\b', 'h', res)
    res = re.sub(r'\b(minute|minutes|分钟)\b', 'min', res)
    res = re.sub(r'\b(second|seconds|秒)\b', 's', res)
    # 不等号 LaTeX 归一
    res = re.sub(r'\\leq|\\le', '≤', res)
    res = re.sub(r'\\geq|\\ge', '≥', res)

    # π 的写法归一
    res = re.sub(r'\b(3\.14159|π|3\.14|兀)\b', 'pi', res)

    # 角命名规范化（与 _canonical_form 完全对齐）
    # 三字母角名：angle/∠/\angle + ABC -> angle B
    res = re.sub(r'(?:\\angle|∠|angle)\s*([a-z])([a-z])([a-z])', r'angle \2', res, flags=re.IGNORECASE)
    # 单字母角名：angle/∠/\angle + C   -> angle C
    res = re.sub(r'(?:\\angle|∠|angle)\s*([a-z])', r'angle \1', res, flags=re.IGNORECASE)

    return res

def _soft_match_equalities(key_str: str, content_str: str) -> bool:
    """
    统一的等式软匹配引擎。
    检查 key_str 中表达的等式逻辑是否能在 content_str 中找到。
    支持连续等式（阈值匹配）和简单等式（严格匹配）。
    """
    if not key_str or not content_str:
        return False

    # 对被检查的内容进行一次规范化
    canon_content = _canonical_form(content_str)

    # 1. 判断关键词是否为连续等式
    is_chained_equality = len(key_str.split('=')) >= 3

    if is_chained_equality:
        # 情况A: 连续等式，采用阈值匹配
        parts = [p.strip() for p in key_str.split('=')]
        sub_keywords = [f"{a}={b}" for a, b in combinations(parts, 2)]
        
        found_count = 0
        for sub_key in sub_keywords:
            canon_sub_key = _canonical_form(sub_key)
            if canon_sub_key in canon_content:
                found_count += 1
                continue
            
            sub_parts = canon_sub_key.split('=')
            if len(sub_parts) == 2:
                reversed_canon_sub_key = f"{sub_parts[1]}={sub_parts[0]}"
                if reversed_canon_sub_key in canon_content:
                    found_count += 1
        
        threshold = 1 / 3 # 使用与 stp_acc_reward 一致的阈值
        match_ratio = found_count / len(sub_keywords) if sub_keywords else 0
        return match_ratio >= threshold
    else:
        # 情况B: 简单等式或非等式，采用严格匹配
        canon_key = _canonical_form(key_str)
        if canon_key in canon_content:
            return True
        
        key_parts = canon_key.split('=')
        if len(key_parts) == 2:
            reversed_canon_key = f"{key_parts[1]}={key_parts[0]}"
            if reversed_canon_key in canon_content:
                return True
        
        # === 新增兜底：同一 LHS 时比 RHS 的数值（支持 content 是连等式，取 RHS 最后一段） ===
        def _split_lr(s: str):
            parts = [p.strip() for p in s.split('=')]
            return parts[0], '='.join(parts[1:]) if len(parts) >= 2 else None

        lk, rk = _split_lr(key_str)
        lc, rc = _split_lr(content_str)
        if lk and rk and lc and rc and lk.strip().lower() == lc.strip().lower():
            rk_last = rk.split('=')[-1].strip()
            rc_last = rc.split('=')[-1].strip()
            # —— 新增：RHS"系数×角度/π" vs "纯角度/π"的冲突直接判 False —— #
            def _has_scaled_degree_or_pi(rhs: str) -> bool:
                r = rhs.strip()
                return bool(
                    re.search(r'[-+]?\d+(?:\.\d+)?\s*(?:\*|×)\s*[-+]?\d+(?:\.\d+)?\s*°', r) or
                    re.search(r'[-+]?\d+(?:\.\d+)?\s*(?:\*|×)\s*(?:\\pi|π|pi)\b', r, flags=re.IGNORECASE)
                )
            def _is_plain_degree_or_pi(rhs: str) -> bool:
                r = rhs.strip()
                return bool(
                    re.fullmatch(r'[-+]?\d+(?:\.\d+)?\s*°', r) or
                    re.fullmatch(r'[-+]?\d+(?:\.\d+)?\s*(?:\\pi|π|pi)\b', r, flags=re.IGNORECASE)
                )
            if ((_has_scaled_degree_or_pi(rk_last) and _is_plain_degree_or_pi(rc_last)) or
                (_has_scaled_degree_or_pi(rc_last) and _is_plain_degree_or_pi(rk_last))):
                return False
            
            kv = _try_eval_latex_numeric(rk_last)
            cv = _try_eval_latex_numeric(rc_last)
            if (kv is not None) and (cv is not None) and abs(kv - cv) < 1e-6:
                return True

        # === 特殊修复：针对 \frac{5}{2} 与 2.5 的匹配 ===
        # 只在非常特定的情况下启用，避免影响其他案例
        if ('\\frac{5}{2}' in key_str or '\\frac{5}{2}' in content_str) and ('2.5' in key_str or '2.5' in content_str):
            # 检查是否包含相同的变量名（BC 或 b）
            key_vars = re.findall(r'\b([A-Za-z]{1,2})\s*=', key_str)
            content_vars = re.findall(r'\b([A-Za-z]{1,2})\s*=', content_str)
            
            if key_vars and content_vars:
                # 检查是否有相同的变量名
                if any(var in content_vars for var in key_vars):
                    # 检查数值是否匹配
                    key_val = _try_eval_latex_numeric(key_str.split('=')[-1].strip())
                    content_val = _try_eval_latex_numeric(content_str.split('=')[-1].strip())
                    
                    if (key_val is not None) and (content_val is not None) and abs(key_val - content_val) < 1e-6:
                        return True

        return False


def _canonical_form(text: str) -> str:
    if not isinstance(text, str):
        return ""
    res = _unify_pi_approx(text).lower()

    # === 新增：把"线段/边/长度/line/segment/side/edge/|AB|/length of AB"等统一成 AB ===
    # 英文：line AB / segment AB / side AB / edge AB / length AB / distance AB / len AB
    res = re.sub(r'\b(line|segment|seg|side|edge|length|len|distance|dist)\s*([a-z])\s*([a-z])\b',
                 r'\2\3', res)
    # 英文：length of AB / distance of AB
    res = re.sub(r'\b(length|len|distance|dist)\s*of\s*([a-z])\s*([a-z])\b',
                 r'\2\3', res)
    # 中文：线段AB / 边AB / 边长AB / 长度AB
    res = re.sub(r'(线段|边长|边|长度)\s*([a-z])\s*([a-z])', r'\2\3', res)
    # 绝对值记法：|AB|
    res = re.sub(r'\|\s*([a-z])\s*([a-z])\s*\|', r'\1\2', res)

    # 角度：\angle/∠/angle + 三字母 -> 取中间字母为顶点：angle B
    res = re.sub(r'(?:\\angle|∠|angle)\s*([a-z])([a-z])([a-z])', r'angle\2', res, flags=re.IGNORECASE)
    # 角度：\angle/∠/angle + 单字母 -> 直接归一：angle B
    res = re.sub(r'(?:\\angle|∠|angle)\s*([a-z])', r'angle\1', res, flags=re.IGNORECASE)

    # 分数/小数统一
    def num_replacer(m):
        s = m.group(0).replace(" ", "")
        try:
            if '/' in s:
                return f'{float(Fraction(s)):.4f}'
            else:
                return s
        except:
            return s
    res = re.sub(r'(\b\d+\s*/\s*\d+\b|\b\d*\.\d+\b|\b\d+\b)', num_replacer, res)

    # 清理无关字符
    res = re.sub(r'[^\w.=\-+*/^()]', '', res)
    return res

def _split_answer_candidates(text: str):
    """把可能包含多个候选的答案按常见分隔符拆成列表（不在这里处理分数的斜杠）"""
    if not isinstance(text, str) or not text.strip():
        return [text]
    # 常见分隔：中文/英文逗号、顿号、分号、'or'、'and'、'或'、'和'
    parts = re.split(r'\s*(?:,|，|、|;|；|\bor\b|\band\b|或|和)\s*', text.strip(), flags=re.IGNORECASE)
    parts = [p for p in parts if p]  # 去空
    return parts if len(parts) >= 2 else [text]

def grade_answer(pred_answer: str, true_answer: str) -> bool:
    if not pred_answer or not true_answer:
        return False

    pred_clean = _rescue_latex_escapes(pred_answer).strip()
    true_clean = _rescue_latex_escapes(true_answer).strip()

    # 归一 Pred 的 "# The final answer is:" 形式，便于 _debug_one 直呼本函数时正确比较
    m_final = re.search(r'The\s+final\s+answer\s+is\s*:\s*(.+)', pred_clean, flags=re.IGNORECASE|re.DOTALL)
    if m_final:
        ans = m_final.group(1).strip()
        # 去括号/尾标点
        ans = re.sub(r'^[\s\(\[\{]+|[\s\)\]\}]+$', '', ans)
        ans = re.sub(r'[;,:!。；，：！]+$', '', ans)
        ans = re.sub(r'\.(?=\s*$)', '', ans)
        pred_clean = ans
    # 若包含 \boxed{...}，提取其中内容
    m_box = re.search(r'\\boxed\s*\{([^}]+)\}', pred_clean)
    if m_box:
        pred_clean = m_box.group(1).strip()

    # --- 新增：根号 vs 被开方数 的硬护栏（双向）
    def _num_or_none(x):
        x = x.strip()
        if re.fullmatch(r'[-+]?\d+(?:\.\d+)?', x):
            try: return float(x)
            except: return None
        if re.fullmatch(r'\d+\s*/\s*\d+', x):
            try: return float(Fraction(x))
            except: return None
        return None

    rad_pred = _extract_sqrt_radicand_if_literal(pred_clean)
    rad_true = _extract_sqrt_radicand_if_literal(true_clean)

    # 形如  pred = \sqrt{n}  vs  true = n（或 a/b，且等于 n） -> 直接 False
    if rad_pred is not None:
        tnum = _num_or_none(true_clean)
        if tnum is not None:
            try:
                if abs(tnum - float(rad_pred)) < 1e-12:
                    return False
            except Exception:
                pass

    # 形如  pred = n  vs  true = \sqrt{n} -> 直接 False
    if rad_true is not None:
        pnum = _num_or_none(pred_clean)
        if pnum is not None:
            try:
                if abs(pnum - float(rad_true)) < 1e-12:
                    return False
            except Exception:
                pass

    # ===== 原有的判定流程 =====
    pred_norm = _latex_to_python_expr(_normalize_symbols(pred_clean))
    true_norm = _latex_to_python_expr(_normalize_symbols(true_clean))
    if pred_norm == true_norm:
        return True

    if _is_radical_expr(pred_clean) ^ _is_radical_expr(true_clean):
        pn = _to_float_general(pred_clean)
        tn = _to_float_general(true_clean)
        if (pn is not None) and (tn is not None):
            return abs(pn - tn) < 2e-3
        return False

    if "=" in pred_norm or "=" in true_norm:
        if _soft_match_equalities(true_norm, pred_norm):
            return True
        # —— 新增：角名桥接等价（triad/单字母 ↔ angle X），并比较度数 ——
        def _angle_center_from_lhs(lhs: str) -> Optional[str]:
            if not isinstance(lhs, str):
                return None
            t = _rescue_latex_escapes(lhs or "").strip()
            # 先匹配显式角记号
            m = re.search(r'(?:\\angle|∠|angle)\s*([A-Za-z])', t)
            if m:
                return m.group(1).lower()
            # 再匹配三字母（取中间）
            m = re.search(r'\b([A-Za-z])([A-Za-z])([A-Za-z])\b', t)
            if m:
                return m.group(2).lower()
            # 兜底：单字母（如 “C”），视作 angle C 的简写
            m = re.search(r'\b([A-Za-z])\b', t)
            if m:
                return m.group(1).lower()
            return None

        def _rhs_degree_value(s: str) -> Optional[float]:
            if not isinstance(s, str):
                return None
            m = re.search(r'([-+]?\d+(?:\.\d+)?)\s*°', s)
            if m:
                try:
                    return float(m.group(1))
                except Exception:
                    return None
            return None

        # ===== 【新增】提取完整的三字母角名 =====
        def _extract_angle_triad(lhs: str) -> Optional[str]:
            """
            提取三字母角名，返回标准化的大写形式
            例如：'angle ABC' → 'ABC', '∠BOD' → 'BOD'
            """
            if not isinstance(lhs, str):
                return None
            t = _rescue_latex_escapes(lhs or "").strip()
            # 匹配 angle/∠/\angle + 三个字母
            m = re.search(r'(?:\\angle|∠|angle)?\s*([A-Za-z])([A-Za-z])([A-Za-z])', t, re.IGNORECASE)
            if m:
                return (m.group(1) + m.group(2) + m.group(3)).upper()
            return None
        
        # ===== 【新增】判断两个三字母角名是否等价（只允许首末对调）=====
        def _are_angles_equivalent(angle1: str, angle2: str) -> bool:
            """
            判断两个三字母角名是否表示同一个角
            规则：只有完全相同或首末对调才算匹配
            例如：'ABC' ≡ 'ABC' ✓
                  'ABC' ≡ 'CBA' ✓ (首末对调)
                  'ABC' ≡ 'CAB' ✗ (不同的角)
                  'ABC' ≡ 'BAC' ✗ (不同的角)
            """
            if not angle1 or not angle2:
                return False
            a1 = angle1.upper()
            a2 = angle2.upper()
            # 完全相同
            if a1 == a2:
                return True
            # 首末对调（ABC ↔ CBA）
            if a1 == a2[::-1]:
                return True
            return False

        if "=" in pred_clean and "=" in true_clean:
            pred_lhs = pred_clean.split("=")[0]
            true_lhs = true_clean.split("=")[0]
            
            # 提取三字母角名
            pred_triad = _extract_angle_triad(pred_lhs)
            true_triad = _extract_angle_triad(true_lhs)
            
            # ===== 【关键修改】如果两边都有三字母角名，必须严格匹配 =====
            if pred_triad and true_triad:
                # 使用严格的角名等价判断（只允许完全相同或首末对调）
                if _are_angles_equivalent(pred_triad, true_triad):
                    # 三字母角名匹配，继续比较度数
                    pv = _rhs_degree_value(pred_clean)
                    tv = _rhs_degree_value(true_clean)
                    if (pv is not None) and (tv is not None) and abs(pv - tv) < 1e-3:
                        return True
                # 如果三字母不等价，直接跳过（不匹配）
            else:
                # ===== 原有逻辑：只有一边或都没有三字母时，用中心字母匹配 =====
                pc = _angle_center_from_lhs(pred_lhs)
                tc = _angle_center_from_lhs(true_lhs)
                if pc is not None and tc is not None and pc == tc:
                    pv = _rhs_degree_value(pred_clean)
                    tv = _rhs_degree_value(true_clean)
                    if (pv is not None) and (tv is not None) and abs(pv - tv) < 1e-3:
                        return True
        
    # ===== 新增护栏：RHS"系数×角度/π" vs "纯角度/π"时，禁止数值兜底 =====
    def _rhs_after_equal(s: str) -> str:
        t = _rescue_latex_escapes(s or "")
        return t.split('=')[-1] if '=' in t else t
    
    def _has_scaled_degree_or_pi(rhs: str) -> bool:
        r = rhs.strip()
        # 形如 2×60°、2 * 60°、2×\pi、2 * pi
        return bool(
            re.search(r'[-+]?\d+(?:\.\d+)?\s*(?:\*|×)\s*[-+]?\d+(?:\.\d+)?\s*°', r) or
            re.search(r'[-+]?\d+(?:\.\d+)?\s*(?:\*|×)\s*(?:\\pi|π|pi)\b', r, flags=re.IGNORECASE)
        )
    
    def _is_plain_degree_or_pi(rhs: str) -> bool:
        r = rhs.strip()
        return bool(
            re.fullmatch(r'[-+]?\d+(?:\.\d+)?\s*°', r) or
            re.fullmatch(r'[-+]?\d+(?:\.\d+)?\s*(?:\\pi|π|pi)\b', r, flags=re.IGNORECASE)
        )

    rhs_pred = _rhs_after_equal(pred_clean)
    rhs_true = _rhs_after_equal(true_clean)

    scaled_plain_conflict = (
        (_has_scaled_degree_or_pi(rhs_pred) and _is_plain_degree_or_pi(rhs_true)) or
        (_has_scaled_degree_or_pi(rhs_true) and _is_plain_degree_or_pi(rhs_pred))
    )
    # 若出现"系数×角度/π" vs "纯角度/π"的冲突，直接认定不等价
    if scaled_plain_conflict:
        return False

    # ===== 新增护栏：若任一端是"含变量的符号表达式"，禁止走纯数值兜底 =====
    def _has_symbolic_var_expr(s: str) -> bool:
        t = _rescue_latex_escapes(s or "")
        # 忽略 π 记号，避免把 \pi 当作变量
        t = re.sub(r'(\\pi|π|兀|pi)', '', t, flags=re.IGNORECASE)
        # 若出现  +x / - y *z ^t 等"运算符 + 变量"的模式，则视为符号表达式
        return bool(re.search(r'[\+\-\*/\^]\s*[A-Za-z]', t))
    
    # 仅当"两边都是纯数字/分数短语"时，才允许数值兜底
    if (_is_pure_number_phrase(pred_clean) and _is_pure_number_phrase(true_clean)):
        pred_num = _to_float_general(pred_clean)
        true_num = _to_float_general(true_clean)
        if (pred_num is not None) and (true_num is not None):
            if abs(pred_num - true_num) < 2e-3:
                return True

    try:
        if abs(float(Fraction(pred_clean)) - float(Fraction(true_clean))) < 1e-6:
            return True
    except Exception:
        pass
    
    pred_final = re.sub(r'[^\w]', '', pred_norm.lower())
    true_final = re.sub(r'[^\w]', '', true_norm.lower())
    return pred_final == true_final

def _has_inequality(s: str) -> bool:
    t = _rescue_latex_escapes(s or "")
    return bool(re.search(r'[<>≤≥]', t))

def _is_pure_number_phrase(s: str) -> bool:
    """是否为纯数字/分数短语：不含字母、不含等号、不含单位与π/角度等"""
    if not isinstance(s, str):
        return False
    t = _rescue_latex_escapes(s or "")
    # 去掉空格和千分位
    t = t.replace(',', '').strip()
    # 明确排除含字母、等号、度数、π 之类
    if re.search(r'[A-Za-z=]', t):
        return False
    if re.search(r'(\\pi|π|°)', t):
        return False
    # 只允许 纯小数/整数 或 纯 a/b 分数
    return bool(
        re.fullmatch(r'[-+]?\d+(?:\.\d+)?', t) or
        re.fullmatch(r'\d+\s*/\s*\d+', t)
    )

def _ratio_or_fraction_value(s: str):
    """把 'a:b'、'a/b'、'\\frac{a}{b}'（a,b 可为小数）解析成浮点；失败返回 None。"""
    if not isinstance(s, str) or not s.strip():
        return None
    t = _rescue_latex_escapes(s)

    # \frac{a}{b} —— 兼容被 Python 吞掉的单斜杠: \f -> \x0c
    m = re.search(r'(?:\\|\x0c)frac\{\s*([-+]?\d+(?:\.\d+)?)\s*\}\{\s*([-+]?\d+(?:\.\d+)?)\s*\}', t)
    if m:
        try:
            a = float(m.group(1)); b = float(m.group(2))
            return a / b if b != 0 else None
        except:
            return None

    # a/b（支持小数）
    m = re.search(r'\b([-+]?\d+(?:\.\d+)?)\s*/\s*([-+]?\d+(?:\.\d+)?)\b', t)
    if m:
        try:
            a = float(m.group(1)); b = float(m.group(2))
            return a / b if b != 0 else None
        except:
            return None

    # a:b（比例）
    m = re.search(r'\b([-+]?\d+(?:\.\d+)?)\s*[:：∶]\s*([-+]?\d+(?:\.\d+)?)\b', t)
    if m:
        try:
            a = float(m.group(1)); b = float(m.group(2))
            return a / b if b != 0 else None
        except:
            return None
    return None

def grade_answer_dir(pred_answer: str, true_answer: str) -> bool:
    """
    定向答案匹配：只承认 "GT 在 Pred 中被体现"。
    - 等式：用 _soft_match_equalities(true, pred)（方向为 GT->Pred）
    - 字面包含：只测 canonical(GT) in canonical(Pred)，不做反向
    - 数值兜底：仅当 GT 是"纯数值结论"（且不含不等号）时才启用
    - 保留根号 vs 被开方数的硬护栏
    """
    if not pred_answer or not true_answer:
        return False
    
    if pred_answer.lower() == true_answer.lower():
        return True
    

    pred_clean = _rescue_latex_escapes(pred_answer).strip()
    true_clean = _rescue_latex_escapes(true_answer).strip()

    # —— 分数/比例 ↔ 小数 的快通道（先于其它护栏）——
    vp = _ratio_or_fraction_value(pred_clean)
    vt = _ratio_or_fraction_value(true_clean)
    tp = _to_float_general(pred_clean)
    tt = _to_float_general(true_clean)

    # 两边都能解析成分数/比例
    if (vp is not None) and (vt is not None) and abs(vp - vt) < 1e-6:
        return True
    # 跨形态：一边分数/比例，另一边小数
    if (vp is not None and tt is not None and abs(vp - tt) < 1e-6) or \
    (vt is not None and tp is not None and abs(vt - tp) < 1e-6):
        return True
    
    m_final = re.search(r'The\s+final\s+answer\s+is\s*:\s*(.+)', pred_clean,
                        flags=re.IGNORECASE | re.DOTALL)
    if m_final:
        ans = m_final.group(1)
        ans = _rescue_latex_escapes(ans)                 # 先 rescue
        ans = ans.strip()
        ans = re.sub(r'^[\s\(\[]+|[\s\)\]]+$', '', ans)  # 去 ()[]
        ans = re.sub(r'[;,:!。；，：！]+$', '', ans)         # 去句尾标点
        ans = re.sub(r'\.(?=\s*$)', '', ans)             # 去尾孤立点
        pred_clean = ans

    # 硬护栏：一方是不等式，另一方是纯数字短语 -> 不匹配（避免范围“撞中”误判）
    if (_has_inequality(pred_clean) and _is_pure_number_phrase(true_clean)) or \
       (_has_inequality(true_clean) and _is_pure_number_phrase(pred_clean)):
        return False

    # 任何一侧包含不等式时，定向匹配不做范围/部分等式等“推断”，直接不匹配
    if _has_inequality(pred_clean) or _has_inequality(true_clean):
        return False

    pred_norm = _latex_to_python_expr(_normalize_symbols(pred_clean))
    true_norm = _latex_to_python_expr(_normalize_symbols(true_clean))

    # 等式软匹配（方向：GT -> Pred）
    if '=' in true_norm or '=' in pred_norm:
        if _soft_match_equalities(true_norm, pred_norm):
            return True

    # 数值兜底：仅当 GT 不含不等号，且（GT 不是"符号等式"或 Pred 也出现相同 LHS）时启用
    if not _has_inequality(true_clean):
        m_assign = re.match(r'^\s*([A-Za-z][A-Za-z0-9_]*)\s*=', true_clean)
        pred_can = _canonical_form(pred_clean)
        allow_numeric_fallback = True
        try:
            pred_clean_num = pred_clean.replace('：', '/').replace('∶', '/').replace(':', '/')
            true_clean_num = true_clean.replace('：', '/').replace('∶', '/').replace(':', '/')
            if abs(float(Fraction(pred_clean_num)) - float(Fraction(true_clean_num))) < 1e-6:
                return True
        except Exception:
            pass
        if m_assign:
            lhs_var = m_assign.group(1).lower()
            has_same_lhs_in_pred = re.search(
                rf'(?<![A-Za-z0-9_]){re.escape(lhs_var)}(?![A-Za-z0-9_])', pred_can
            ) is not None
            has_equ_in_pred = '=' in pred_can
            allow_numeric_fallback = has_same_lhs_in_pred or has_equ_in_pred
        if allow_numeric_fallback:
            pn = _to_float_general(pred_clean)
            tn = _to_float_general(true_clean)
            if (pn is not None) and (tn is not None) and abs(pn - tn) < 2e-3:
                return True
            try:
                if abs(float(Fraction(pred_clean)) - float(Fraction(true_clean))) < 1e-6:
                    return True
            except Exception:
                pass
            # 角度计算匹配（如 140° vs 180°-40°）
            def _match_angle_calculation(angle_str, calc_str):
                try:
                    angle_match = re.search(r'([-+]?\d+(?:\.\d+)?)\s*°', angle_str)
                    if not angle_match:
                        return False
                    angle_val = float(angle_match.group(1))
                    calc_clean = _rescue_latex_escapes(calc_str)
                    calc_match = re.search(r'([-+]?\d+(?:\.\d+)?)\s*°\s*([+\-])\s*([-+]?\d+(?:\.\d+)?)\s*°', calc_clean)
                    if calc_match:
                        val1 = float(calc_match.group(1))
                        op = calc_match.group(2)
                        val2 = float(calc_match.group(3))
                        calc_result = val1 + val2 if op == '+' else val1 - val2
                        return abs(angle_val - calc_result) < 1e-3
                except Exception:
                    pass
                return False
            if _match_angle_calculation(pred_clean, true_clean) or _match_angle_calculation(true_clean, pred_clean):
                return True
            
            # 分数格式匹配：1.0/3.0 与 \frac{1}{3} 的匹配
            def _match_fraction_formats(pred_str, true_str):
                try:
                    # 提取分数值
                    def extract_fraction_value(s):
                        # 匹配 1.0/3.0 格式
                        decimal_frac_match = re.search(r'(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)', s)
                        if decimal_frac_match:
                            num = float(decimal_frac_match.group(1))
                            den = float(decimal_frac_match.group(2))
                            return num / den if den != 0 else None
                        
                        # 匹配 \frac{1}{3} 格式
                        latex_frac_match = re.search(r'\\frac\{(\d+)\}\{(\d+)\}', s)
                        if latex_frac_match:
                            num = int(latex_frac_match.group(1))
                            den = int(latex_frac_match.group(2))
                            return num / den if den != 0 else None
                        
                        # 匹配 1/3 格式
                        simple_frac_match = re.search(r'(\d+)\s*/\s*(\d+)', s)
                        if simple_frac_match:
                            num = int(simple_frac_match.group(1))
                            den = int(simple_frac_match.group(2))
                            return num / den if den != 0 else None
                        
                        return None
                    
                    pred_val = extract_fraction_value(pred_str)
                    true_val = extract_fraction_value(true_str)
                    
                    if pred_val is not None and true_val is not None:
                        return abs(pred_val - true_val) < 1e-6
                    
                    return False
                except Exception:
                    return False
            
            # 比例格式匹配：1:3 与 1/3 的匹配
            def _match_ratio_formats(pred_str, true_str):
                try:
                    # 提取比例值
                    def extract_ratio_value(s):
                        # 匹配 1:3 格式
                        ratio_match = re.search(r'(\d+)\s*:\s*(\d+)', s)
                        if ratio_match:
                            num = int(ratio_match.group(1))
                            den = int(ratio_match.group(2))
                            return num / den if den != 0 else None
                        
                        # 匹配 1/3 格式
                        frac_match = re.search(r'(\d+)\s*/\s*(\d+)', s)
                        if frac_match:
                            num = int(frac_match.group(1))
                            den = int(frac_match.group(2))
                            return num / den if den != 0 else None
                        
                        return None
                    
                    pred_val = extract_ratio_value(pred_str)
                    true_val = extract_ratio_value(true_str)
                    
                    if pred_val is not None and true_val is not None:
                        return abs(pred_val - true_val) < 1e-6
                    
                    return False
                except Exception:
                    return False
            
            # 应用新的匹配逻辑
            if _match_fraction_formats(pred_clean, true_clean) or _match_fraction_formats(true_clean, pred_clean):
                return True
            
            if _match_ratio_formats(pred_clean, true_clean) or _match_ratio_formats(true_clean, pred_clean):
                return True

    return False

def grade_step_dir(pred_answer: str, true_answer: str) -> bool:
    """
    定向答案匹配：只承认 "GT 在 Pred 中被体现"。
    - 等式：用 _soft_match_equalities(true, pred)（方向为 GT->Pred）
    - 字面包含：只测 canonical(GT) in canonical(Pred)，不做反向
    - 数值兜底：仅当 GT 是"纯数值结论"（且不含不等号）时才启用
    - 保留根号 vs 被开方数的硬护栏
    """
    if not pred_answer or not true_answer:
        return False

    pred_clean = _rescue_latex_escapes(pred_answer).strip()
    true_clean = _rescue_latex_escapes(true_answer).strip()

    # 硬护栏：一方是不等式，另一方是纯数字短语 -> 不匹配（避免范围“撞中”误判）
    if (_has_inequality(pred_clean) and _is_pure_number_phrase(true_clean)) or \
       (_has_inequality(true_clean) and _is_pure_number_phrase(pred_clean)):
        return False

    # 任何一侧包含不等式时，定向匹配不做范围/部分等式等“推断”，直接不匹配
    if _has_inequality(pred_clean) or _has_inequality(true_clean):
        return False

    pred_norm = _latex_to_python_expr(_normalize_symbols(pred_clean))
    true_norm = _latex_to_python_expr(_normalize_symbols(true_clean))

    # 等式软匹配（方向：GT -> Pred）
    if '=' in true_norm or '=' in pred_norm:
        if _soft_match_equalities(true_norm, pred_norm):
            return True

    # 数值兜底：仅当 GT 不含不等号，且（GT 不是"符号等式"或 Pred 也出现相同 LHS）时启用
    if not _has_inequality(true_clean):
        m_assign = re.match(r'^\s*([A-Za-z][A-Za-z0-9_]*)\s*=', true_clean)
        pred_can = _canonical_form(pred_clean)
        allow_numeric_fallback = True
        try:
            pred_clean_num = pred_clean.replace('：', '/').replace('∶', '/').replace(':', '/')
            true_clean_num = true_clean.replace('：', '/').replace('∶', '/').replace(':', '/')
            if abs(float(Fraction(pred_clean_num)) - float(Fraction(true_clean_num))) < 1e-6:
                return True
        except Exception:
            pass
        if m_assign:
            lhs_var = m_assign.group(1).lower()
            has_same_lhs_in_pred = re.search(
                rf'(?<![A-Za-z0-9_]){re.escape(lhs_var)}(?![A-Za-z0-9_])', pred_can
            ) is not None
            has_equ_in_pred = '=' in pred_can
            allow_numeric_fallback = has_same_lhs_in_pred or has_equ_in_pred
        if allow_numeric_fallback:
            pn = _to_float_general(pred_clean)
            tn = _to_float_general(true_clean)
            if (pn is not None) and (tn is not None) and abs(pn - tn) < 2e-3:
                return True
            try:
                if abs(float(Fraction(pred_clean)) - float(Fraction(true_clean))) < 1e-6:
                    return True
            except Exception:
                pass
            # 角度计算匹配（如 140° vs 180°-40°）
            def _match_angle_calculation(angle_str, calc_str):
                try:
                    angle_match = re.search(r'([-+]?\d+(?:\.\d+)?)\s*°', angle_str)
                    if not angle_match:
                        return False
                    angle_val = float(angle_match.group(1))
                    calc_clean = _rescue_latex_escapes(calc_str)
                    calc_match = re.search(r'([-+]?\d+(?:\.\d+)?)\s*°\s*([+\-])\s*([-+]?\d+(?:\.\d+)?)\s*°', calc_clean)
                    if calc_match:
                        val1 = float(calc_match.group(1))
                        op = calc_match.group(2)
                        val2 = float(calc_match.group(3))
                        calc_result = val1 + val2 if op == '+' else val1 - val2
                        return abs(angle_val - calc_result) < 1e-3
                except Exception:
                    pass
                return False
            if _match_angle_calculation(pred_clean, true_clean) or _match_angle_calculation(true_clean, pred_clean):
                return True
            
            # 分数格式匹配：1.0/3.0 与 \frac{1}{3} 的匹配
            def _match_fraction_formats(pred_str, true_str):
                try:
                    # 提取分数值
                    def extract_fraction_value(s):
                        # 匹配 1.0/3.0 格式
                        decimal_frac_match = re.search(r'(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)', s)
                        if decimal_frac_match:
                            num = float(decimal_frac_match.group(1))
                            den = float(decimal_frac_match.group(2))
                            return num / den if den != 0 else None
                        
                        # 匹配 \frac{1}{3} 格式
                        latex_frac_match = re.search(r'\\frac\{(\d+)\}\{(\d+)\}', s)
                        if latex_frac_match:
                            num = int(latex_frac_match.group(1))
                            den = int(latex_frac_match.group(2))
                            return num / den if den != 0 else None
                        
                        # 匹配 1/3 格式
                        simple_frac_match = re.search(r'(\d+)\s*/\s*(\d+)', s)
                        if simple_frac_match:
                            num = int(simple_frac_match.group(1))
                            den = int(simple_frac_match.group(2))
                            return num / den if den != 0 else None
                        
                        return None
                    
                    pred_val = extract_fraction_value(pred_str)
                    true_val = extract_fraction_value(true_str)
                    
                    if pred_val is not None and true_val is not None:
                        return abs(pred_val - true_val) < 1e-6
                    
                    return False
                except Exception:
                    return False
            
            # 比例格式匹配：1:3 与 1/3 的匹配
            def _match_ratio_formats(pred_str, true_str):
                try:
                    # 提取比例值
                    def extract_ratio_value(s):
                        # 匹配 1:3 格式
                        ratio_match = re.search(r'(\d+)\s*:\s*(\d+)', s)
                        if ratio_match:
                            num = int(ratio_match.group(1))
                            den = int(ratio_match.group(2))
                            return num / den if den != 0 else None
                        
                        # 匹配 1/3 格式
                        frac_match = re.search(r'(\d+)\s*/\s*(\d+)', s)
                        if frac_match:
                            num = int(frac_match.group(1))
                            den = int(frac_match.group(2))
                            return num / den if den != 0 else None
                        
                        return None
                    
                    pred_val = extract_ratio_value(pred_str)
                    true_val = extract_ratio_value(true_str)
                    
                    if pred_val is not None and true_val is not None:
                        return abs(pred_val - true_val) < 1e-6
                    
                    return False
                except Exception:
                    return False
            
            # 应用新的匹配逻辑
            if _match_fraction_formats(pred_clean, true_clean) or _match_fraction_formats(true_clean, pred_clean):
                return True
            
            if _match_ratio_formats(pred_clean, true_clean) or _match_ratio_formats(true_clean, pred_clean):
                return True

    return False

# 提取第一个step序号，正常情况返回一个≥1的数字，没有这句话返回-1，step后面不是数字返回-2
def extract_first_step(predict_str: str) -> int:
    pattern = r"#?\s*Step\s+(\S+)\s*:"
    match = re.search(pattern, predict_str, re.IGNORECASE)
    if match:
        step_capture = match.group(1)
        try:
            return int(step_capture)
        except ValueError:
            return -2
    return -1

# 答案奖励
def acc_reward(predict_str: str, ground_truth: str, use_boxed: bool) -> float:
    
    # 先判断 GT 是否为多候选
    gt_candidates = _split_answer_candidates(ground_truth)
    gt_is_multi = len(gt_candidates) > 1

    # ---------- 新增：提取用于“直判”的预测尾巴 ----------
    def _extract_tail_or_full(s: str) -> str:
        # 优先取 "# The final answer is:" 的尾巴；否则用全文
        m = re.findall(r"The\s+final\s+answer\s+is\s*:\s*(.+?)(?:\n|$)", s, re.IGNORECASE | re.DOTALL)
        cand = m[-1] if m else s
        return cand.strip()

    # ---------- 新增：先不区分大小写直接来一次：

    if _extract_tail_or_full(predict_str).lower() == ground_truth.strip().lower():
        return 1.0

    # ---------- 判断gt是否选择：
    def is_multiple_choice_option(gt: str) -> bool:
        gt_stripped = gt.strip()
        
        allowed_patterns = [
            r'^[A-Za-z]$',                  # 单独A
            r'^[A-Za-z]\.$',                # 单独A.
            r'^\([A-Za-z]\)$',              # 单独(A)  ✓ 现在可以匹配了
            r'^[A-Za-z]\.\s*\S',            # A. xx
            r'^\([A-Za-z]\)\s*\S',          # (A) xx  ✓ 现在可以匹配了
            r'^[A-Za-z]:\s*\S',             # A: xx
            r'^[A-Za-z]\)\s*\S',            # A) xx
        ]
        
        for pattern in allowed_patterns:
            if re.match(pattern, gt_stripped, flags=re.IGNORECASE):
                return True
        
        # 调试
        #print(f"[{gt_stripped}] is not option]")
        return False

    def extract_option_letter(s: str) -> str: #找出pred中的选项
        s = s.strip()
        # 匹配第一个出现的字母（跳过前导的非字母数字字符）
        match = re.match(r'^[^\w]*([A-Za-z])', s, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
        return ""

    # ==================== MODIFICATION START ====================
    # (在这里插入新的 extract_option_content 函数)
    def extract_option_content(s: str) -> str:
        s = s.strip()
        
        # 匹配并捕获内容的模式
        # A. content
        # A: content
        # A) content
        # (A) content
        patterns = [
            r'^[A-Za-z]\.\s*(.+)',       # A. content
            r'^[A-Za-z]:\s*(.+)',       # A: content
            r'^[A-Za-z]\)\s*(.+)',      # A) content
            r'^\([A-Za-z]\)\s*(.+)',   # (A) content
        ]
        
        for pattern in patterns:
            # 使用 re.DOTALL 确保匹配跨行内容
            match = re.match(pattern, s, flags=re.IGNORECASE | re.DOTALL) 
            if match:
                return match.group(1).strip()
                
        # 如果是独立选项（A, A., (A)）或根本不是选项，则没有内容
        return ""

    if is_multiple_choice_option(ground_truth):  
        predict_str_tail = _extract_tail_or_full(predict_str)
        
        # 检查 1: Pred 也是一个选项 (例如 "A" or "A. Apple")
        if is_multiple_choice_option(predict_str_tail):
            if extract_option_letter(predict_str_tail) == extract_option_letter(ground_truth):
                return 1.0
            else:
                return 0.0 # 字母不匹配

        # 检查 2: (新增) Pred 不是选项，检查它是否匹配 GT 选项的 *内容*
        gt_content = extract_option_content(ground_truth)
        
        # 如果 GT 有内容 并且 pred 的尾巴匹配该内容
        if gt_content and grade_answer_dir(predict_str_tail, gt_content):
            return 1.0
    # ==================== MODIFICATION END ======================

    #提取出对应的选项，如何避免提取的gt为 像 Ant 这样以选项开头的
    # ---------- 新增规则 (1)：GT 为单独一个字母（选项）时，直接匹配 ----------
    if not gt_is_multi :
        gt_stripped = ground_truth.strip()
        # 规则(1)：GT 为单独一个字母（选项）
        letter_match = re.match(r'^[A-Za-z]', gt_stripped)
        if re.fullmatch(r"[A-Za-z]", gt_stripped, flags=re.IGNORECASE):
            letter = gt_stripped
            tail = _extract_tail_or_full(predict_str)

            # 允许的“独立出现”形态：A / A. / (A)    （可带尾部空白或换行）
            pure_ok = re.fullmatch(
                rf"""\s*(?:\(\s*{re.escape(letter)}\s*\)|{re.escape(letter)}\.?)\s*""",
                tail, flags=re.IGNORECASE | re.VERBOSE
            ) is not None

            # 允许的“带前缀标点后跟说明文字”形态：
            # A. text / A: text / A) text / (A) text
            punct_ok = (
                re.fullmatch(
                    rf"""\s*{re.escape(letter)}\s*(?:[.:)] )\s+\S.*""",
                    tail, flags=re.IGNORECASE | re.VERBOSE
                ) is not None
                or re.fullmatch(
                    rf"""\s*\(\s*{re.escape(letter)}\s*\)\s+\S.*""",
                    tail, flags=re.IGNORECASE | re.VERBOSE
                ) is not None
            )

            # 显式排除 "A text"（字母后直接空格接文字、没有 . : ) 或右括号）
            # 上面的两个条件都不会匹配到这种情况，因此无需额外正则排除

            if pure_ok or punct_ok:
                return 1.0
            else:
                return 0.0

        # ---------- 新增规则 (2)：GT 为 yes/yes./no/no. 时，直接匹配 ----------
        # 归一化为 base token（yes/no），忽略尾部句点与空白
        def _yn_norm(x: str) -> str:
            x = x.strip().lower()
            if x.endswith("."):
                x = x[:-1].strip()
            return x

        if _yn_norm(gt_stripped) in {"yes", "no"} and re.fullmatch(r"(yes\.?|no\.?)", gt_stripped.strip(), flags=re.IGNORECASE):
            tail = _extract_tail_or_full(predict_str)
            if _yn_norm(tail) == _yn_norm(gt_stripped):
                return 1.0
            else:
                return 0.0

    # 如果使用boxed（备用）
    if use_boxed:
        boxed_answer = extract_boxed_content(predict_str)
        if boxed_answer and boxed_answer.strip():
            # 若 GT 单解而Pred枚举多个 -> 判0，避免"猜一串"
            if not gt_is_multi and _has_multiple_candidates(boxed_answer):
                return 0.0
            return 1.0 if grade_answer_dir(boxed_answer.strip(), ground_truth.strip()) else 0.0
    
    predict_str = _rescue_latex_escapes(predict_str)
    # 如果直接用The final answer is:
    pattern = r"The\s+final\s+answer\s+is\s*:\s*(.+?)(?:\n|$)"
    match = re.findall(pattern, predict_str, re.IGNORECASE | re.DOTALL)
    if match:
        # 1) 先拿到“尾巴原文”
        raw_tail = match[-1]

        # 2) 先 rescue，再做“轻清洗”（只剥 () [] 外壳与句尾标点；不要动 {}，避免破坏 \frac）
        tail = _rescue_latex_escapes(raw_tail).strip()
        tail = re.sub(r'^[\s\(\[]+|[\s\)\]]+$', '', tail)    # 去 () []
        tail = re.sub(r'[;,:!。；，：！]+$', '', tail)        # 去句尾标点
        tail = re.sub(r'\.(?=\s*$)', '', tail)              # 去尾孤立点

        gt_clean = _rescue_latex_escapes(ground_truth).strip()
        gt_clean = re.sub(r'^[\s\(\[]+|[\s\)\]]+$', '', gt_clean)    # 去 () []
        gt_clean = re.sub(r'[;,:!。；，：！]+$', '', gt_clean)        # 去句尾标点
        gt_clean = re.sub(r'\.(?=\s*$)', '', gt_clean)              # 去尾孤立点
        
        if tail:
            # 单解时，若尾巴像“枚举多个候选”，直接判 0（和你原逻辑一致）
            if not gt_is_multi and _has_multiple_candidates(tail):
                return 0.0
            
            # === 护栏 1：Pred 是不等式且 GT 是纯数字 => 必须判 0（禁止任何兜底）===
            if _has_inequality(tail) and _is_pure_number_phrase(gt_clean):
                return 0.0
            # === 护栏 2：等式 vs 纯数字 => 直接 0 分（避免 "a=4" 被数值兜底成 True）===
            if _has_pure_equality(tail) and _is_pure_number_phrase(gt_clean):
                # print("等式尾巴 vs 纯数字 GT，直接 0")
                return 0.0

            # 3) 先用干净尾巴判一次
            if grade_answer_dir(tail, gt_clean):
                return 1.0
            
            # 4) 数值兜底（关键补丁）：分数/比例/数字等价，避免被其它护栏路径绕开 =
            vp = _ratio_or_fraction_value(tail)
            vt = _ratio_or_fraction_value(gt_clean)
            if (vp is not None) and (vt is not None) and abs(vp - vt) < 1e-6:
                return 1.0
            # 跨形态：分数/比例 vs 十进制
            tp = _to_float_general(tail)
            tt = _to_float_general(gt_clean)
            if (vp is not None and tt is not None and abs(vp - tt) < 1e-6) or \
            (vt is not None and tp is not None and abs(vt - tp) < 1e-6):
                return 1.0

            # 6) 仍不匹配则 0 分
            # print("答案不匹配:", repr(tail), " vs ", repr(gt_clean))  # 若要排查可打开
            return 0.0
    return 0.0

# 计算格式奖励
def fmt_reward(predict_str: str) -> float:
    has_final = re.search(r"#\s*The\s+final\s+answer\s+is\s*:", predict_str, re.IGNORECASE) is not None
    return 1.0 if has_final else 0.0

#计算步骤奖励
def stp_reward(predict_str: str, judging_step: int, step_num: int) -> float:
    if not predict_str: return 0.0
    first_step_num = extract_first_step(predict_str)
    
    if judging_step != 0: # 需要续写步骤
        if first_step_num == -1 or first_step_num == -2:
            return 0.0
        if first_step_num == judging_step:
            return 1.0
        else:
            return max(0.0, 1.0 - 0.25 * abs(first_step_num - judging_step))
    else: # 推理已完备，应直接输出答案
        if first_step_num == -1:
            return 1.0 # 正确行为
        elif first_step_num == step_num + 1:
            return 0.05 # 输出了冗余步骤
        else:
            return 0.0 # 输出了冗余且错误的步骤

def extract_step_chunks(predict_str: str, max_steps: int = None):
    """
    提取所有 '# Step k:' 的内容块，返回 [(k, text), ...]。
    只切到 '# Step ...'、'# The final answer is:' 或文本结束。
    """
    if not predict_str:
        return []
    pattern = re.compile(
        r"#\s*Step\s*(\d+)\s*:(.*?)(?=#\s*Step\s*\d|#\s*The\s+final\s+answer\s+is:|\Z)",
        re.IGNORECASE | re.DOTALL
    )
    chunks = [(int(m.group(1)), m.group(2).strip()) for m in pattern.finditer(predict_str)]
    if max_steps is not None:
        return chunks[:max_steps]
    return chunks

# 获取关键词并存到json文件
def get_keyword_with_json(sample_id: str, judging_step: int, step_ground_truth: str, step_num: int, key_json_path: str = None) -> str:
    sample_id_str = str(sample_id)
    judging_step_str = str(judging_step)

    # 选用调用方提供的路径，否则用全局默认
    json_path = key_json_path or os.getenv('KEY_JSON_PATH')
    if not json_path:
        raise ValueError("未指定 JSON 缓存文件路径，请设置 key_json_path 参数或环境变量 KEY_JSON_PATH。")

    # 确保缓存目录存在
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    # 加载JSON缓存文件
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            key_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        # 如果文件不存在或为空，初始化一个空字典
        key_data = {}

    # 查找json 如果关键词不为空
    if (sample_id_str in key_data and
        judging_step_str in key_data[sample_id_str].get("key_map", {}) and
        key_data[sample_id_str]["key_map"][judging_step_str] is not None):
        
        keyword = key_data[sample_id_str]["key_map"][judging_step_str]
        print(f"存在该关键词  ID {sample_id_str}, 步骤 {judging_step_str} -> 关键词: {keyword}")
        return keyword
    
    # 当前关键词为空，调用API提取
    print(f"不存在关键词  ID {sample_id_str}, 步骤 {judging_step_str}. 调用API...")
    keyword = call_api_for_keywords(step_ground_truth)

    # 更新json中的数据
    if sample_id_str not in key_data:
        new_key_map = {str(i): None for i in range(1, step_num + 1)}
        key_data[sample_id_str] = {
            "key_steps": [],
            "key_map": new_key_map
        }
    
    key_data[sample_id_str]["key_map"][judging_step_str] = keyword
    
    # 将更新后的完整数据写回JSON文件
    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(key_data, f, indent=2, ensure_ascii=False)
        print(f"json文件更新：  已将新关键词写入 {json_path}")
    except IOError as e:
        print(f"错误： 无法写入缓存文件: {e}")

    return keyword

# 调用qwen api提取关键词
def call_api_for_keywords(step_ground_truth: str) -> str:
    """
    使用 QWEN 从标准步骤文本中提取关键词。
    若 API 调用失败则自动重试，最多 5 次；全部失败后打印异常。
    """
    if qwen_client is None:
        print("API客户端未初始化，无法进行调用。")
        return ""
    
    # 构建提示语
    prompt = (
        "Extract ONE most key DERIVED conclusion (NOT a given) in no more than one short phrase, avoiding any descriptive words.\n"
        "If solving a mathematical problem, preferably a numerical value or equation. "
        "Else, it can be a comparison, trend, or qualitative statement.\n"
        "Make sure the key conclusion is explicitly stated in the given text.\n"
        "Do NOT infer or calculate anything beyond this text! (e.g., 'triangle ABF is half of the area of triangle ABE.', do NOT infer 'area of triangle ABE = 4' if it is not given in the text.)\n"
        "If the conclusion is an equation or inequality, output the entire relation including EVERY side (keep variables/symbols/units) and keep full chains (e.g., 'a=b=8/2=4', not '8/2=4'); do not truncate to just the number.\n"
        "If the text contains a phrase like 'Find', 'Calculate', or a colon ':' before the equation, treat the variable or symbol before the colon as the first part of the equation.\n\n"
        "Respond with only the extracted text:\n\n"
        f"Text:\n{step_ground_truth}"
    )

    # # 构建提示语（包含NONE）
    # prompt = (
    #     "Extract the key DERIVED conclusion (preferably a numerical value or equation) (not a given) "
    #     "in no more than one short phrase, avoiding any descriptive words. "
    #     "Make sure the key conclusion is included in the given text,"
    #     "do NOT infer or calculate anything beyond this text."
    #     "If the text has no explicit derived result or equation, respond with one token 'NONE'. "
    #     "If the conclusion is an equation or inequality, output the entire relation including EVERY side (keep variables/symbols/units) and keep full chains (e.g., 'a=b=8/2=4', not '8/2=4'); do not truncate to just the number. "
    #     "If the text contains a phrase like 'Find', 'Calculate', or a colon ':' before the equation, treat the variable or symbol before the colon as the first part of the equation. "
    #     "Respond with only the extracted text:\n\n"
    #     f"Text:\n{step_ground_truth}"
    # )

    # 多次尝试直至全部失败
    n = 5
    for attempt in range(1, n + 1):
        try:
            response = qwen_client.chat.completions.create(
                model=QWEN_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            keyword = response.choices[0].message.content.strip()
            if keyword:
                # --- 新增：去掉最外层 \text{...} 包裹 ---
                m = re.fullmatch(r'''\\text\s*\{(.*)\}''', keyword, flags=re.DOTALL)
                if m:
                    keyword = m.group(1).strip()
                print(f"步骤文本ground truth: {step_ground_truth}")
                print(f"[QWEN] 提取关键词成功(第{attempt}次): {keyword}")
                return keyword
            else:
                print(f"[QWEN] 返回内容为空(第{attempt}次)，准备重试…")
    
        except Exception as e:
            # 只打印，不抛
            print(f"[QWEN] 调用失败(第{attempt}次): {e}")
        # 非最后一次时做轻微退避
        if attempt < n:
            import time, random
            time.sleep(0.5 * attempt + random.uniform(0, 0.25))
    
    # 到这里代表 5 次都未成功
    print("[QWEN] 连续 5 次调用未成功，返回空关键词。")
    return ""

def _gpt_semantic_equiv(text: str, key_word: str) -> Optional[bool]:
    """
    用 QWEN 判断模型输出是否能得到与keyword相同的数学结论（数值/等式/不等式/角度/长度等）。
    返回 True/False；若 API 不可用或异常，返回 None（让上层做兜底）。
    若 API 调用失败则自动重试，最多 5 次；全部失败后打印异常。
    """
    # —— 打印输入（10000 字符截断）——
    try:
        ta = _truncate_for_log(text, 5000)
    except NameError:
        # 兜底：若项目中没有 _truncate_for_log
        def _trunc(s, n=5000):
            if s is None:
                return ""
            s = str(s)
            return s if len(s) <= n else s[:n] + f"...[truncated {len(s)-n} chars]"
        ta = _trunc(text, 5000)

    print(f"===> 【关键词】 {key_word!r}")
    print(f"===> 【步骤文本】 {ta!r}")

    if qwen_client is None:
        print("API客户端未初始化，无法进行调用。")
        return None
    # prompt = (
    #         "You are a strict mathematical fact checker.\n"
    #         "Decide if the Text conveys the SAME mathematical conclusion as the Keyword.\n"
    #         "Consider algebraic equivalence, chained equalities, LaTeX, units, and numeric equivalence (tolerance 1e-3). Treat 'measure/value of X' as 'X'.\n"
    #         "Answer strictly with a single token: YES or NO.\n\n"
    #         f"Text:\n{text}\n\nKeyword:\n{key_word}\n\nAnswer:"
    #     )
    # 对于多种类别
    system_prompt = (
        "You are a precise semantic equivalence evaluator.\n"
        "Determine whether the Text conveys the SAME conclusion or meaning as the Keyword.\n"
        "Consider synonyms, paraphrases, logical equivalence, tone consistency, and factual accuracy.\n"
        "If the Keyword expresses a mathematical statement, also perform simple arithmetic or algebraic reasoning before comparison, considering algebraic equivalence, chained equalities, LaTeX formatting, units, and numeric equivalence (tolerance 1e-3). Treat 'the measure/value of X' as equivalent to 'X'.\n"
        "For chart-related questions, apply a more flexible matching rule: treat any reference to the same data element, trend, or value in a chart as equivalent, even if one phrasing describes the observation or action (e.g., 'read', 'look at', 'find') and the other states the value explicitly."
        "Respond strictly with a single token: YES or NO.\n\n"
    )
    user_prompt = f"Text:\n{text}\n\nKeyword:\n{key_word}\n\nAnswer:"
    n = 5
    for attempt in range(1, n + 1):
        try:
            response = qwen_client.chat.completions.create(
                model=QWEN_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,
            )
            out = response.choices[0].message.content.strip().upper()
            # 只抓第一个 YES/NO 词
            m = re.search(r'\b(YES|NO)\b', out)
            if not m:
                print(f"[QWEN] 无法解析回答(第{attempt}次): {out[:80]!r}，准备重试…")
            else:
                token = m.group(1)
                if token == "YES":
                    print("===> QWEN判断: 步骤正确！")
                    return True
                else:  # "NO"
                    print("===> QWEN判断: 【步骤错误】")
                    return False
            
        except Exception as e:
            # 只打印，不抛
            print(f"[QWEN] 调用失败(第{attempt}次): {e}")
        # 非最后一次时做轻微退避
        if attempt < n:
            import time, random
            time.sleep(0.5 * attempt + random.uniform(0, 0.25))
    
    # 到这里代表 5 次都未成功
    print("[QWEN] 连续 5 次调用未成功，返回None。")
    return None

# 提取出现的前三个步骤的内容
def extract_following_steps_content(predict_str: str, num_steps: int = 3) -> str:
    if not predict_str: return ""
    pattern = re.compile(
        r"#\s*Step\s*\d+\s*:(.*?)(?=#\s*Step\s*\d|#\s*The\s+final\s+answer\s+is:|\Z)",
        re.IGNORECASE | re.DOTALL
    )
    matches = pattern.finditer(predict_str)
    
    contents = []
    for i, match in enumerate(matches):
        if i < num_steps:
            contents.append(match.group(1).strip())
        else:
            break
            
    # 将找到的多个步骤内容用空格连接成一个大字符串
    return " ".join(contents)

# 关键步骤奖励
def stp_acc_reward(predict_str: str, extra_info: dict) -> float:
    # ---- 早退护栏 ====
    if not predict_str:
        return 0.0
    
    # 动态阈值：基础3000 + 每步+1000，但不超过绝对上限
    dyn_limit = _dyn_pred_threshold(extra_info.get("judging_step", 0), extra_info.get("step_num", 0))
    if len(predict_str) > dyn_limit:
        print(f"[stp_acc] skip: predict_str too long ({len(predict_str)} > dyn_limit={dyn_limit}) -> 0")
        return 0.0
    
    step_marks = predict_str.count("# Step")
    if step_marks > STP_MAX_STEP_MARKS:
        print(f"[stp_acc] skip: too many step markers ({step_marks} > {STP_MAX_STEP_MARKS}) -> 0")
        return 0.0

    # 从extra_info中提取judging_step
    judging_step = extra_info.get("judging_step", 0)
    sample_id = extra_info.get("sample_id", None) 
    step_num = extra_info.get("step_num", 0)
    key_json_path = extra_info.get("key_json_path", None)

    if judging_step == 0:
        return 1.0

    if sample_id is None:
        print("错误  extra_info 中缺少 'sample_id'，无法使用关键词缓存。")
        return 0.0
        
    ground_truth_continuation = extra_info.get("answer", "")
    if not ground_truth_continuation:
        print("extra_info 中缺少 'answer' 字段或其值为空，无法计算奖励。")
        return 0.0

    step_ground_truth = extract_following_steps_content(ground_truth_continuation, 1)
    if not step_ground_truth:
        print(f"未能从'answer'字段提取到关键步骤 ({judging_step}) 的标准答案文本。")
        return 0.0

    key_str = get_keyword_with_json(sample_id, judging_step, step_ground_truth, step_num, key_json_path)
    if not key_str:
        print("未能从标准答案中提取到关键词，奖励为0。")
        return 0.0

    MAX_LOOKAHEAD_STEPS = 2
    step_chunks = extract_step_chunks(predict_str, max_steps = MAX_LOOKAHEAD_STEPS)
    if not step_chunks:
        print("未能在模型输出中找到步骤块，奖励为0.")
        return 0.0

    # 逐步严格匹配
    print("====> 开始 Soft Matching...")
    for k, step_text in step_chunks:
        print(f"===> 【关键词】 {key_str!r}")
        print(f"===> 【检查第 {k} 个步骤】 {step_text[:1000]!r}...")
        if _strict_match_key_in_step(key_str, step_text):
            print("===> Soft Matching: 步骤正确！")
            return 1.0

    # 如果软匹配未匹配到，再用qwen匹配一次
    print("====> Soft Matching 未匹配到，尝试用 QWEN 语义判断...")
    joined_steps_text = "\n".join(text for _, text in step_chunks).strip()
    if not joined_steps_text:
        return 0.0
    verdict = _gpt_semantic_equiv(joined_steps_text, key_str)
    if verdict is True:
        return 1.0
    if verdict is None:
        # QWEN 不可用/失败：回退到启发式判定
        try:
            if grade_step_dir(joined_steps_text, step_ground_truth):
                return 1.0
        except Exception:
            pass

    # 全部未匹配，不给分
    return 0.0

# 计算最终奖励（并在此处写入 sample_metrics.jsonl） 
def compute_score(*args, **kwargs) -> float: 
    """ 
        期望从下面位置获取写文件所需信息（任选其一或都给）： 
        - kwargs["rollout_dir"] / kwargs["global_step"] 
        - extra_info["rollout_dir"] / extra_info["global_step"] 
        只要拿到 rollout_dir，就会把一条 JSONL 记录追加到 rollout_dir/sample_metrics.jsonl 
        记录字段： sample_id, global_step, acc, fmt, stp_fmt, stp_acc 
    """ 
    print(f"===== 使用mulberry_with_steps计算了奖励函数 =====") 

    # ---- 取输入 ---- 
    predict_str = kwargs.get("solution_str", "")
    ground_truth = kwargs.get("ground_truth", "")
    use_boxed = kwargs.get("use_boxed", False)
    extra_info = kwargs.get("extra_info", None) 
    
    if extra_info is None: 
        raise ValueError("extra_info 不能为空，必须传入包含 judging_step 和 step_num 的 dict") 
    judging_step = extra_info.get("judging_step", 0) 
    step_num = extra_info.get("step_num", 0) 
    sample_id = extra_info.get("sample_id", 0) 
    
    # ---- 四个子分 ---- 
    # 只用尾部窗口做 acc/fmt, 避免异常长文本拖慢
    pred_for_simple = (predict_str or "")
    if len(pred_for_simple) > ACC_FMT_WINDOW:
        pred_for_simple = pred_for_simple[-ACC_FMT_WINDOW:]

    stp_acc_score = stp_acc_reward(predict_str, extra_info) 
    keyword_is_none = (stp_acc_score == 0.5)

    # NONE步骤则这条数据不参与更新
    if keyword_is_none:
        acc_score = None
        fmt_score = None
        stp_fmt_score = None
        final_score = 0.0

        print("=============== UPDATE =================") 
        print("========================================") 
        print(f"sample id: {sample_id}") 
        print(f"judging step: {judging_step}")
        print("----------------------------------------") 
        print(f"stp acc score: {stp_acc_score}") 
        print("关键词为 NONE，跳过该样本的奖励更新。")
        
    else:
        acc_score = acc_reward(pred_for_simple, ground_truth, use_boxed) 
        fmt_score = fmt_reward(pred_for_simple) 
        stp_fmt_score = stp_reward(predict_str, judging_step, step_num)
        # 最终分（步骤准确度权重1，答案准确度权重2）
        final_score = fmt_score + stp_acc_score + stp_fmt_score + 2 * acc_score

        print("=============== UPDATE =================") 
        print("========================================") 
        print(f"sample id: {sample_id}") 
        print(f"judging step: {judging_step}")
        print("----------------------------------------") 
        print(f"predict str: {_truncate_for_log(predict_str, 2000)}") 
        print(f"ground truth: {ground_truth}") 
        print("----------------------------------------") 
        print(f"acc score: {acc_score}") 
        print(f"fmt score: {fmt_score}") 
        print(f"stp fmt score: {stp_fmt_score}") 
        print(f"stp acc score: {stp_acc_score}") 
        print("----------------------------------------") 
        print(f"final score: {final_score}") 

    reward_extra_info = {
        "acc": acc_score,
        "fmt": fmt_score,
        "stp_fmt": stp_fmt_score,
        "stp_acc": stp_acc_score,
        "keyword_is_none": keyword_is_none,
        "sample_id": sample_id,
        "judging_step": judging_step,
        "step_num": step_num,
    }    

    # ---- 写 JSONL（若拿得到 rollout_dir）---- 
    # 支持两处来源：kwargs 优先，其次 extra_info 
    rollout_dir = "/outputs" 
    # 获取目录下的所有子文件夹 
    subdirs = [ d for d in os.listdir(rollout_dir) if os.path.isdir(os.path.join(rollout_dir, d)) ] 
    # 取名字最大的文件夹（即最新的时间戳） 
    parent_name = max(subdirs) 
    print(f"最新的文件夹:{parent_name}") 
    latest_path = os.path.join(rollout_dir, parent_name) 
    print(f"最新路径:{latest_path}") 

    
    if rollout_dir: 
        try: 
            os.makedirs(rollout_dir, exist_ok=True) 
            jsonl_path = os.path.join(latest_path, "sample_metrics.jsonl") 
            # 当前sample的idx自增 
            _sample_idx_map[sample_id] += 1 
            curr_idx = _sample_idx_map[sample_id] 
            # 记录基础信息（缺啥写 None） 
            sample_id = extra_info.get("sample_id") 
            row = { 
                "sample_id": str(sample_id) if sample_id is not None else None, 
                "idx": curr_idx,
                "judging_step": judging_step,
                "acc": float(acc_score) if acc_score is not None else None, 
                "fmt": float(fmt_score) if fmt_score is not None else None, 
                "stp_fmt": float(stp_fmt_score) if stp_fmt_score is not None else None, 
                "stp_acc": float(stp_acc_score) if stp_acc_score is not None else None, 
                "final_score": float(final_score) if final_score is not None else None,
            } 
            with open(jsonl_path, "a", encoding="utf-8") as f: 
                f.write(json.dumps(row, ensure_ascii=False) + "\n") 
                print("===== 写入了sample_metrics.jsonl =====") 
                
        
        except Exception as e: 
            # 打印但不影响训练 
            print(f"[warn] 写入 sample_metrics.jsonl 失败：{e}") 

    return final_score, reward_extra_info