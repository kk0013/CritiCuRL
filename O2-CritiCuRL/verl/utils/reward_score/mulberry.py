import re, math, ast
from fractions import Fraction
from openai import OpenAI
import os, json
from itertools import combinations

# ========= 轻量阈值 =========
STP_MAX_PRED_LEN = int(os.getenv("STP_MAX_PRED_LEN", "3000"))   # 预测文本超过这个长度，stp_acc_reward 直接=0
ACC_FMT_WINDOW   = int(os.getenv("ACC_FMT_WINDOW", "200"))     # 计算 acc/fmt 时，只看末尾这么多字符
STP_MAX_STEP_MARKS = int(os.getenv("STP_MAX_STEP_MARKS", "10"))# #Step 过多视为异常，stp_acc_reward=0

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

# 文件自带的匹配boxed{}函数
def extract_boxed_content(text: str) -> str:
    if not text:
        return ""
    pattern = r'\\boxed\s*\{([^}]+)\}'
    match = re.search(pattern, text)
    if match:
        return match.group(1).strip()
    return ""

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

# def grade_answer(pred_answer: str, true_answer: str) -> bool:
#     if not pred_answer or not true_answer:
#         return False

#     # ===== 优先处理“多答案”场景 =====
#     # 只允许GT多候选，Pred不拆
#     # 若 GT 含多个候选，逐个与 pred 比较；任一匹配即 True
#     true_candidates = _split_answer_candidates(true_answer)
#     if len(true_candidates) > 1:
#         return any(grade_answer(pred_answer, cand) for cand in true_candidates)

#     # # 若 pred 含多个候选，逐个与 GT 比较；任一匹配即 True
#     # pred_candidates = _split_answer_candidates(pred_answer)
#     # if len(pred_candidates) > 1:
#     #     return any(grade_answer(cand, true_answer) for cand in pred_candidates)

#     pred_clean = _rescue_latex_escapes(pred_answer).strip()
#     true_clean = _rescue_latex_escapes(true_answer).strip()

#     # --- 先符号归一化，解决 π≈3.14 的情况 ---
#     pred_norm = _latex_to_python_expr(_normalize_symbols(pred_clean))
#     true_norm = _latex_to_python_expr(_normalize_symbols(true_clean))
#     if pred_norm == true_norm:
#         return True

#     # --- 再做等式软匹配 ---
#     if "=" in pred_norm or "=" in true_norm:
#         if _soft_match_equalities(true_norm, pred_norm):
#             return True

#     # --- 数值比较，但只在足够接近时返回 True，不接近则继续 ---
#     pred_num = _to_float_general(pred_clean)
#     true_num = _to_float_general(true_clean)
#     if (pred_num is not None) and (true_num is not None):
#         if abs(pred_num - true_num) < 2e-3:   # 容差可以放宽到 2e-3
#             return True

#     # --- 再兜底：Fraction 比较 ---
#     try:
#         if abs(float(Fraction(pred_clean)) - float(Fraction(true_clean))) < 1e-6:
#             return True
#     except Exception:
#         pass

#     # --- 最后兜底：去标点和大小写 ---
#     pred_final = re.sub(r'[^\w]', '', pred_norm.lower())
#     true_final = re.sub(r'[^\w]', '', true_norm.lower())
#     # return pred_final == true_final or (true_final in pred_final or pred_final in true_final)
#     return pred_final == true_final

# # 提取第一个step序号，正常情况返回一个≥1的数字，没有这句话返回-1，step后面不是数字返回-2
# def extract_first_step(predict_str: str) -> int:
#     pattern = r"#?\s*Step\s+(\S+)\s*:"
#     match = re.search(pattern, predict_str, re.IGNORECASE)
#     if match:
#         step_capture = match.group(1)
#         try:
#             return int(step_capture)
#         except ValueError:
#             return -2
#     return -1

def _has_inequality(s: str) -> bool:
    t = _rescue_latex_escapes(s or "")
    return bool(re.search(r'[<>≤≥]', t))

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

_EQ_SIGN_RE = re.compile(r'(?<![<>!=])=(?![=<>])')  # 只匹配“纯等号”，排除 <= >= == !=

def _has_pure_equality(s: str) -> bool:
    if not isinstance(s, str) or not s:
        return False
    t = _rescue_latex_escapes(s)
    return _EQ_SIGN_RE.search(t) is not None

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
# # 过程格式奖励(Let's think step by step + The final answer is)
# def fmt_reward(predict_str: str, judging_step: int, step_num: int) -> float:
#     # fmt:格式奖励
#     has_think = "# Let's think step by step." in predict_str
#     has_final = re.search(r"#\s*The\s+final\s+answer\s+is\s*:", predict_str) is not None
#     fmt = 1.0 if (has_think and has_final) else 0.0
#     return fmt

# #计算步骤奖励
# def stp_reward(predict_str: str, judging_step: int, step_num: int) -> float:
#     if not predict_str: return 0.0
#     first_step_num = extract_first_step(predict_str)
    
#     if judging_step != 0: # 需要续写步骤
#         if first_step_num == -1 or first_step_num == -2:
#             return 0.0
#         if first_step_num == judging_step:
#             return 1.0
#         else:
#             return max(0.0, 1.0 - 0.25 * abs(first_step_num - judging_step))
#     else: # 推理已完备，应直接输出答案
#         if first_step_num == -1:
#             return 1.0 # 正确行为
#         elif first_step_num == step_num + 1:
#             return 0.05 # 输出了冗余步骤
#         else:
#             return 0.0 # 输出了冗余且错误的步骤

# # 格式奖励
# def format_reward(predict_str: str, judging_step: int, step_num: int) -> float:

#     # stp = stp_fmt_reward(predict_str, judging_step, step_num)
#     fmt = fmt_reward(predict_str, judging_step, step_num)

#     return fmt

# # 计算最终奖励
# def compute_score(predict_str: str, ground_truth: str, use_boxed: bool, extra_info: None) -> float:
#     # 从 extra_info 中提取 judging_step 和 step_num
#     judging_step = None
#     step_num = None
#     if extra_info is not None and isinstance(extra_info, dict):
#         judging_step = extra_info.get("judging_step", 0)
#         step_num = extra_info.get("step_num", 0)
#     acc_score = acc_reward(predict_str, ground_truth, use_boxed)
#     fmt_score = format_reward(predict_str, judging_step, step_num)
#     final_score = 0.7 * acc_score + 0.3 * fmt_score
#     if extra_info is None:
#         raise ValueError("extra_info 不能为空，必须传入包含 judging_step 和 step_num 的 dict")
#     return min(max(final_score, 0.0), 1.0)

# 计算最终奖励
def compute_score(*args, **kwargs) -> float:
    predict_str = kwargs.get("solution_str", "")
    ground_truth = kwargs.get("ground_truth", "")
    use_boxed = kwargs.get("use_boxed", False)
    extra_info = kwargs.get("extra_info", None)
    if extra_info is None:
        raise ValueError("extra_info 不能为空，必须传入包含 judging_step 和 step_num 的 dict")
    judging_step = extra_info.get("judging_step", 0)
    step_num = extra_info.get("step_num", 0)
    sample_id = extra_info.get("sample_id", 0) 
    
    # 只用尾部窗口做 acc/fmt, 避免异常长文本拖慢
    pred_for_simple = (predict_str or "")
    if len(pred_for_simple) > ACC_FMT_WINDOW:
        pred_for_simple = pred_for_simple[-ACC_FMT_WINDOW:]
    
    acc_score = acc_reward(pred_for_simple, ground_truth, use_boxed)
    # fmt_score = format_reward(pred_for_simple, judging_step, step_num)
    final_score = acc_score

    print("============== ROLLOUT =================") 
    print("========================================") 
    print(f"sample id: {sample_id}") 
    print("----------------------------------------") 
    print(f"predict str: {_truncate_for_log(predict_str, 2000)}") 
    print(f"ground truth: {ground_truth}") 
    print("----------------------------------------") 
    print(f"final score: {final_score}") 
    
    return final_score