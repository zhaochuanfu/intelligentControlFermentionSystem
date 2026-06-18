import struct
import re
import json
import time
import os
import logging
import threading
import io
import contextlib
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from dotenv import load_dotenv, set_key

import streamlit as st
from pydantic import BaseModel, Field, field_validator
from pymodbus.client.tcp import ModbusTcpClient
import requests

# ===================== 基础常量 =====================
CONFIG_FILE = "modbus_config.json"
SCRIPTS_DIR = "modbus_scripts"
WRITE_RETRY_COUNT = 3
WRITE_RETRY_DELAY = 0.5  # 秒
REFRESH_INTERVAL = 2  # 秒
FERMENTATION_DATA_FILE = "fermentation_history.json"
ENV_FILE = ".env"
DEEPSEEK_API_BASE = "https://api.deepseek.com/v1"

FERMENTATION_FIELDS = [
    "温度显示",
    "pH显示",
    "溶氧含量",
    "搅拌显示",
    "补料1",
    "补料2",
    "发酵时间",
]

# 创建目录
os.makedirs(SCRIPTS_DIR, exist_ok=True)

# 加载本地.env缓存密钥
load_dotenv(ENV_FILE)

# ===================== 日志配置 =====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# 发酵线程全局变量
_fermentation_thread: Optional[threading.Thread] = None
_fermentation_thread_stop_event = threading.Event()

# ===================== Modbus数据模型（无修改） =====================
class ModbusConfig(BaseModel):
    host: str = Field(default="192.168.103.159", description="Modbus服务器IP")
    port: int = Field(default=1003, ge=1, le=65535, description="Modbus端口")
    timeout: int = Field(default=3, ge=1, le=30, description="连接超时时间(秒)")

    @field_validator("host")
    def validate_host(cls, v):
        import ipaddress
        try:
            ipaddress.ip_address(v)
        except ValueError:
            raise ValueError(f"无效的IP地址: {v}")
        return v

class Collector(BaseModel):
    config: ModbusConfig

    @staticmethod
    def reg_to_float(reg_hi: int, reg_lo: int) -> float:
        raw_data = struct.pack(">HH", reg_hi, reg_lo)
        return struct.unpack(">f", raw_data)[0]

    @staticmethod
    def float_to_reg(value: float) -> Tuple[int, int]:
        raw_data = struct.pack(">f", value)
        return struct.unpack(">HH", raw_data)

    @staticmethod
    def _device_sort_key(name: str) -> Tuple:
        if name == "未知":
            return (1, 0, name)
        num_list = re.findall(r"\d+", name)
        num = int(num_list[0]) if num_list else 0
        return (0, num, name)

    @staticmethod
    def get_modifiable_params(channel_labels: List[Tuple]) -> List[Tuple]:
        return [
            (ch_num, dev_name, param_name)
            for ch_num, dev_name, param_name, is_modifiable in channel_labels
            if is_modifiable
        ]

    @staticmethod
    def validate_param_value(param_name: str, value: float, param_ranges: Dict) -> Tuple[bool, str]:
        param_range = param_ranges.get(param_name)
        if not param_range:
            logger.warning(f"参数 {param_name} 未配置范围限制，跳过校验")
            return True, ""
        min_val, max_val = param_range
        if not (min_val <= value <= max_val):
            error_msg = f"参数 {param_name} 值 {value} 超出有效范围 [{min_val}, {max_val}]"
            logger.error(error_msg)
            return False, error_msg
        return True, ""

    def decode_registers(
        self,
        regs: List[int],
        channel_labels: List[Tuple],
        only_modifiable: bool = False,
        only_read_only: bool = False
    ) -> Dict[str, Dict[str, float]]:
        if only_modifiable and only_read_only:
            raise ValueError("only_modifiable和only_read_only不能同时为True")
        grouped_data: Dict[str, Dict[str, float]] = {}
        for ch_num, dev_name, param, is_modifiable in channel_labels:
            if only_modifiable and not is_modifiable:
                continue
            if only_read_only and is_modifiable:
                continue
            idx = ch_num * 2 - 1
            if idx + 1 >= len(regs):
                logger.warning(f"通道 {ch_num} 寄存器索引越界，跳过")
                continue
            try:
                val = self.reg_to_float(regs[idx], regs[idx + 1])
                grouped_data.setdefault(dev_name, {})[param] = round(val, 2)
            except Exception as e:
                logger.error(f"解析通道 {ch_num} 失败: {e}")
                continue
        return dict(sorted(grouped_data.items(), key=lambda x: self._device_sort_key(x[0])))

    def read_modbus(self, channel_labels: List[Tuple]) -> Optional[Dict[str, Dict[str, float]]]:
        batch_size = 124
        total_regs = []
        with ModbusTcpClient(
            host=self.config.host,
            port=self.config.port,
            timeout=self.config.timeout
        ) as client:
            if not client.connect():
                logger.error(f"连接失败 -> {self.config.host}:{self.config.port}")
                return None
            logger.info(f"连接成功 -> {self.config.host}:{self.config.port}")
            for batch_idx in range(3):
                start_addr = batch_idx * batch_size
                resp = None
                for device_arg in ("device_id", "unit", "slave"):
                    try:
                        resp = client.read_holding_registers(
                            address=start_addr,
                            count=batch_size,
                            **{device_arg: 1}
                        )
                        break
                    except TypeError as err:
                        if f"unexpected keyword argument '{device_arg}'" in str(err):
                            continue
                        raise
                if resp is None or resp.isError():
                    logger.error(f"第{batch_idx+1}批读取失败，起始地址:{start_addr}")
                    return None
                total_regs.extend(resp.registers)
        return self.decode_registers(total_regs, channel_labels)

    def write_single_param(
        self,
        ch_num: int,
        value: float,
        param_ranges: Optional[Dict] = None,
        retry_count: int = WRITE_RETRY_COUNT
    ) -> bool:
        start_addr = ch_num * 2 - 1
        reg_hi, reg_lo = self.float_to_reg(value)
        logger.info(f"准备写入 -> 通道：{ch_num}，值：{value}")
        for attempt in range(1, retry_count + 1):
            try:
                with ModbusTcpClient(
                    host=self.config.host,
                    port=self.config.port,
                    timeout=self.config.timeout
                ) as client:
                    if not client.connect():
                        logger.error(f"第{attempt}次尝试：连接失败")
                        if attempt < retry_count:
                            time.sleep(WRITE_RETRY_DELAY)
                            continue
                        return False
                    for device_arg in ("device_id", "unit", "slave"):
                        try:
                            resp = client.write_registers(
                                address=start_addr,
                                values=[reg_hi, reg_lo],
                                **{device_arg: 1}
                            )
                            break
                        except TypeError as err:
                            if f"unexpected keyword argument '{device_arg}'" in str(err):
                                continue
                            raise
                    if resp and not resp.isError():
                        logger.info(f"第{attempt}次尝试：写入成功")
                        return True
                    else:
                        logger.error(f"第{attempt}次尝试：写入失败，响应：{resp}")
            except Exception as e:
                logger.error(f"第{attempt}次尝试：发生异常 - {e}")
            if attempt < retry_count:
                time.sleep(WRITE_RETRY_DELAY)
        logger.error(f"所有{retry_count}次尝试均失败")
        return False

    def write_batch_params(
        self,
        params: List[Tuple[int, float]],
        param_ranges: Optional[Dict] = None,
        retry_count: int = WRITE_RETRY_COUNT
    ) -> Dict[int, bool]:
        results = {}
        for ch_num, value in params:
            results[ch_num] = self.write_single_param(ch_num, value, param_ranges, retry_count)
        return results

# ===================== 配置文件读写（无修改） =====================
def load_default_config() -> dict:
    return {
        "connection": {
            "host": "192.168.103.159",
            "port": 1003,
            "timeout": 3
        },
        "channel_labels": [
            (50, "BX-4", "温度显示", False),
            (51, "BX-4", "pH显示", False),
            (52, "BX-4", "溶氧含量", False),
            (53, "BX-4", "搅拌显示", False),
            (54, "BX-4", "补料1", False),
            (55, "BX-4", "补料2", False),
            (56, "BX-4", "碱累积", False),
            (57, "BX-4", "酸累计", False),
            (58, "BX-4", "发酵时间", False),
            (59, "BX-4", "温度设定", True),
            (60, "BX-4", "pH设定", True),
            (61, "BX-4", "溶氧上限", True),
            (62, "BX-4", "溶氧下限", True),
            (63, "BX-4", "转速设定", True),
            (64, "BX-4", "消泡周期", True),
            (65, "BX-4", "消泡工作时间", True),
            (66, "BX-4", "补料1周期", True),
            (67, "BX-4", "补料1工作时间", True),
            (68, "BX-4", "补料2周期", True),
            (69, "BX-4", "补料2工作时间", True),
        ],
        "param_ranges": {
            "温度设定": [0.0, 100.0],
            "pH设定": [0.0, 14.0],
            "溶氧上限": [0.0, 100.0],
            "溶氧下限": [0.0, 100.0],
            "转速设定": [0.0, 1000.0],
            "消泡周期": [0.0, 3600.0],
            "消泡工作时间": [0.0, 300.0],
            "补料1周期": [0.0, 3600.0],
            "补料1工作时间": [0.0, 300.0],
            "补料2周期": [0.0, 3600.0],
            "补料2工作时间": [0.0, 300.0],
        }
    }

def load_config() -> dict:
    if not os.path.exists(CONFIG_FILE):
        default_config = load_default_config()
        save_config(default_config)
        return default_config
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            config = json.load(f)
            default_config = load_default_config()
            for key in default_config:
                if key not in config:
                    config[key] = default_config[key]
            return config
    except Exception as e:
        st.warning(f"配置文件读取失败，使用默认配置: {e}")
        return load_default_config()

def save_config(config: dict) -> None:
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=4)
        st.success("✅ 配置文件保存成功！")
    except Exception as e:
        st.error(f"❌ 配置文件保存失败: {e}")

def load_fermentation_history() -> List[Dict]:
    if not os.path.exists(FERMENTATION_DATA_FILE):
        return []
    try:
        with open(FERMENTATION_DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"加载发酵历史失败: {e}")
        return []

def save_fermentation_history(history: List[Dict]) -> None:
    try:
        with open(FERMENTATION_DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=4)
    except Exception as e:
        logger.error(f"保存发酵历史失败: {e}")

def extract_fermentation_record(data: Dict[str, Dict[str, float]]) -> Dict:
    record = {"timestamp": datetime.now().isoformat()}
    for field in FERMENTATION_FIELDS:
        record[field] = None
        for device_values in data.values():
            if field in device_values:
                record[field] = device_values[field]
                break
    return record

def _fermentation_worker(collector: Collector, channel_labels: List[Tuple], interval: int = 60) -> None:
    global _fermentation_thread_stop_event
    while not _fermentation_thread_stop_event.is_set():
        data = collector.read_modbus(channel_labels)
        if data:
            record = extract_fermentation_record(data)
            history = load_fermentation_history()
            history.append(record)
            save_fermentation_history(history)
            logger.info(f"已采集发酵数据: {record}")
        _fermentation_thread_stop_event.wait(interval)

def start_fermentation_thread(collector: Collector, channel_labels: List[Tuple]) -> None:
    global _fermentation_thread, _fermentation_thread_stop_event
    if _fermentation_thread and _fermentation_thread.is_alive():
        return
    _fermentation_thread_stop_event.clear()
    _fermentation_thread = threading.Thread(
        target=_fermentation_worker,
        args=(collector, channel_labels),
        daemon=True
    )
    _fermentation_thread.start()

def stop_fermentation_thread() -> None:
    global _fermentation_thread, _fermentation_thread_stop_event
    _fermentation_thread_stop_event.set()
    _fermentation_thread = None

# ===================== DeepSeek API 工具函数（读取页面输入的密钥） =====================
def check_deepseek_api(api_key: str, model: str, timeout: int) -> Tuple[bool, str]:
    if not api_key or not api_key.startswith("sk-"):
        return False, "API Key格式错误，必须以sk-开头"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 10
    }
    try:
        resp = requests.post(
            f"{DEEPSEEK_API_BASE}/chat/completions",
            headers=headers,
            json=payload,
            timeout=timeout
        )
        if resp.status_code == 200:
            return True, "连接正常"
        elif resp.status_code == 401:
            return False, "API Key无效/权限不足"
        elif resp.status_code == 429:
            return False, "调用频次超限，请稍后重试"
        else:
            return False, f"接口异常，状态码{resp.status_code}：{resp.text[:300]}"
    except requests.ConnectionError:
        return False, "无法连接DeepSeek服务器，请检查网络"
    except requests.Timeout:
        return False, "请求超时，增大超时时间或检查网络"
    except Exception as e:
        return False, f"检测失败：{str(e)}"

def generate_script(
    prompt: str,
    channel_labels: List[Tuple],
    param_ranges: Dict,
    api_key: str,
    model: str,
    timeout: int
) -> str:
    if not api_key:
        return "# 错误：未填写DeepSeek API密钥，请在侧边栏配置"
    system_prompt = f"""你是专业发酵罐Modbus Python脚本生成助手。
根据用户自然描述输出可直接运行控制代码，禁止多余文字、解释、markdown。
可用可写参数（通道号,设备,参数名）：
{json.dumps([c for c in channel_labels if c[3]], ensure_ascii=False, indent=2)}
参数取值范围：
{json.dumps(param_ranges, ensure_ascii=False, indent=2)}
允许函数：
1.collector.write_single_param(ch_num:int, value:float) -> bool
2.collector.write_batch_params([(通道号,数值),...])
3.time.sleep(秒)
4.print(文本)
约束：
1.仅输出纯Python代码，无```标记
2.数值严格匹配参数上下限
3.只使用提供通道号，不新增参数
4.不导入任何额外库
"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "temperature": 0.1,
        "max_tokens": 2500,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    }
    try:
        resp = requests.post(
            f"{DEEPSEEK_API_BASE}/chat/completions",
            headers=headers,
            json=payload,
            timeout=timeout
        )
        resp.raise_for_status()
        res_json = resp.json()
        code_raw = res_json["choices"][0]["message"]["content"]
        # 移除代码块标记
        code = re.sub(r"```(python)?\n?", "", code_raw)
        code = re.sub(r"```\n?", "", code)
        return code.strip()
    except requests.HTTPError as e:
        err_msg = f"API调用失败 {e.response.status_code if e.response else ''}: {e}"
        logger.error(err_msg)
        return f"# {err_msg}"
    except Exception as e:
        logger.error(f"生成脚本异常：{e}")
        return f"# 脚本生成出错：{str(e)}"

# ===================== 脚本文件操作（无修改） =====================
def save_script(name: str, code: str, description: str = "") -> Tuple[bool, str]:
    try:
        if not name.strip():
            return False, "脚本名称不能为空"
        script_data = {
            "name": name,
            "description": description,
            "code": code,
            "created_at": datetime.now().isoformat()
        }
        file_path = os.path.join(SCRIPTS_DIR, f"{name}.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(script_data, f, ensure_ascii=False, indent=4)
        logger.info(f"保存脚本：{name}")
        return True, ""
    except IOError as e:
        return False, f"文件读写错误：{str(e)}"
    except Exception as e:
        return False, f"保存失败：{str(e)}"

def load_script(name: str) -> Optional[Dict]:
    try:
        fp = os.path.join(SCRIPTS_DIR, f"{name}.json")
        if not os.path.exists(fp):
            return None
        with open(fp, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"加载脚本{name}失败：{e}")
        return None

def list_scripts() -> List[str]:
    names = []
    for f in os.listdir(SCRIPTS_DIR):
        if f.endswith(".json"):
            names.append(f[:-5])
    return sorted(names)

def execute_script(code: str, collector: Collector) -> str:
    safe_globals = {
        "collector": collector,
        "time": time,
        "print": print,
        "__builtins__": {
            "len": len,
            "range": range,
            "int": int,
            "float": float,
            "str": str,
            "list": list,
            "dict": dict,
            "print": print,
        }
    }
    output = io.StringIO()
    errors = io.StringIO()
    try:
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
            exec(code, safe_globals, {})
        out = output.getvalue().strip()
        err = errors.getvalue().strip()
        ret = ""
        if out:
            ret += f"【标准输出】\n{out}\n"
        if err:
            ret += f"【错误输出】\n{err}"
        return ret if ret else "脚本执行完成，无输出"
    except SyntaxError as e:
        return f"【语法错误】第{e.lineno}行：{e.msg}"
    except Exception as e:
        return f"【运行异常】{type(e).__name__}: {str(e)}"

# ===================== 主页面逻辑 =====================
def main():
    st.set_page_config(
        page_title="发酵罐监控系统",
        page_icon="🍺",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # 会话状态初始化DeepSeek配置，优先读取本地.env缓存
    if "deepseek_api_key" not in st.session_state:
        st.session_state.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY", "")
    if "deepseek_model" not in st.session_state:
        st.session_state.deepseek_model = "deepseek-coder"
    if "deepseek_timeout" not in st.session_state:
        st.session_state.deepseek_timeout = 60

    # 其他会话状态
    if "config" not in st.session_state:
        st.session_state.config = load_config()
    if "last_data" not in st.session_state:
        st.session_state.last_data = {}
    if "write_queue" not in st.session_state:
        st.session_state.write_queue = {}
    if "last_refresh_time" not in st.session_state:
        st.session_state.last_refresh_time = time.time()
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "current_script" not in st.session_state:
        st.session_state.current_script = ""
    if "should_refresh" not in st.session_state:
        st.session_state.should_refresh = False
    if "fermentation_running" not in st.session_state:
        st.session_state.fermentation_running = False
    if "fermentation_start_time" not in st.session_state:
        st.session_state.fermentation_start_time = None
    if "fermentation_history" not in st.session_state:
        st.session_state.fermentation_history = load_fermentation_history()
    if "last_fermentation_collection" not in st.session_state:
        st.session_state.last_fermentation_collection = 0

    # ========== 侧边栏：导航 + DeepSeek配置面板 ==========
    with st.sidebar:
        st.header("🧭 功能导航")
        page = st.selectbox(
            "功能菜单",
            ["实时监控与控制", "发酵数据采集", "系统配置", "AI脚本助手"]
        )
        st.divider()

        # DeepSeek网页配置区
        st.header("🤖 DeepSeek API配置")
        api_key_input = st.text_input(
            "API Key",
            value=st.session_state.deepseek_api_key,
            placeholder="sk-xxxxxxxxxxxxxxxx",
            type="password",
            help="前往 https://platform.deepseek.com 获取密钥"
        )
        model_sel = st.selectbox(
            "模型选择",
            ["deepseek-coder", "deepseek-v4-pro", "deepseek-chat"],
            index=["deepseek-coder", "deepseek-v4-pro", "deepseek-chat"].index(st.session_state.deepseek_model)
        )
        timeout_val = st.slider("请求超时(秒)", min_value=10, max_value=120, value=st.session_state.deepseek_timeout)

        # 更新会话缓存
        st.session_state.deepseek_api_key = api_key_input
        st.session_state.deepseek_model = model_sel
        st.session_state.deepseek_timeout = timeout_val

        col_save, col_test = st.columns(2)
        with col_save:
            if st.button("保存配置到本地.env", use_container_width=True):
                set_key(ENV_FILE, "DEEPSEEK_API_KEY", api_key_input)
                st.success("已保存，重启程序自动读取")
        with col_test:
            if st.button("测试连接", use_container_width=True):
                ok, msg = check_deepseek_api(api_key_input, model_sel, timeout_val)
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)

        st.divider()
        # 脚本管理（仅AI页面可见）
        if page == "AI脚本助手":
            st.subheader("📜 脚本管理")
            script_names = list_scripts()
            selected_script = st.selectbox("加载已有脚本", [""] + script_names)
            if selected_script:
                s_data = load_script(selected_script)
                if s_data:
                    st.session_state.current_script = s_data["code"]
                    st.info(f"已加载：{selected_script}")
                    st.write("描述：", s_data.get("description", "无"))
                    if st.button("删除脚本"):
                        os.remove(os.path.join(SCRIPTS_DIR, f"{selected_script}.json"))
                        st.rerun()
            st.subheader("💡 使用示例")
            st.write("- 将温度设定37.5，pH7.0")
            st.write("- 运行2小时后停止补料")

    # ===================== 页面1：实时监控与控制 =====================
    if page == "实时监控与控制":
        st.title("📊 发酵罐实时监控与参数控制")
        config = st.session_state.config
        modbus_config = ModbusConfig(**config["connection"])
        collector = Collector(config=modbus_config)
        channel_labels = config["channel_labels"]
        param_ranges = config["param_ranges"]
        param_info = {}
        for ch_num, dev_name, param_name, is_mod in channel_labels:
            param_info[(dev_name, param_name)] = (ch_num, is_mod)

        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            refresh_btn = st.button("🔄 刷新数据", type="primary", use_container_width=True)
            if refresh_btn:
                st.session_state.should_refresh = True
        with col2:
            auto_refresh = st.checkbox("自动刷新", False)
        with col3:
            if auto_refresh:
                st.info(f"每{REFRESH_INTERVAL}秒自动刷新")

        now = time.time()
        need_refresh = st.session_state.should_refresh or (auto_refresh and now - st.session_state.last_refresh_time >= REFRESH_INTERVAL)
        if need_refresh:
            with st.spinner("读取Modbus数据..."):
                data = collector.read_modbus(channel_labels)
                if data:
                    st.session_state.last_data = data
                else:
                    st.warning("Modbus读取失败，使用缓存数据")
                st.session_state.last_refresh_time = now
            # 批量写入队列
            if st.session_state.write_queue:
                with st.spinner("写入参数..."):
                    batch_params = list(st.session_state.write_queue.items())
                    res = collector.write_batch_params(batch_params, param_ranges)
                    succ = sum(res.values())
                    total = len(res)
                    if succ == total:
                        st.success(f"全部写入成功({succ}/{total})")
                    else:
                        st.error(f"部分写入失败({succ}/{total})")
                    st.session_state.write_queue = {}
            st.session_state.should_refresh = False

        data = st.session_state.last_data
        if data:
            for device, params in data.items():
                st.subheader(f"设备 {device}")
                read_only = []
                writable = []
                for name, val in params.items():
                    ch, mod = param_info.get((device, name), (None, False))
                    if mod and ch is not None:
                        writable.append((name, val, ch))
                    else:
                        read_only.append((name, val))
                st.markdown("#### 📈 只读监测参数")
                cols = st.columns(4)
                for idx, (n, v) in enumerate(read_only):
                    with cols[idx % 4]:
                        st.metric(n, v)
                st.markdown("#### ⚙️ 可控制参数")
                w_cols = st.columns(3)
                for idx, (name, val, ch) in enumerate(writable):
                    with w_cols[idx % 3]:
                        minv, maxv = param_ranges.get(name, (0, 100))
                        new_val = st.number_input(
                            name, value=val, min_value=minv, max_value=maxv, step=0.1, format="%.2f",
                            key=f"ctrl_{device}_{name}_{ch}"
                        )
                        if abs(new_val - val) > 1e-6:
                            st.session_state.write_queue[ch] = new_val
                            st.info(f"{name} 将在下一次刷新写入")
        else:
            st.warning("暂无设备数据，请检查Modbus连接")
            if st.button("重新连接"):
                st.session_state.last_refresh_time = 0
                st.rerun()

    # ===================== 页面2：发酵数据采集 =====================
    elif page == "发酵数据采集":
        st.title("🍺 发酵数据采集与曲线")
        config = st.session_state.config
        modbus_config = ModbusConfig(**config["connection"])
        collector = Collector(config=modbus_config)
        channel_labels = config["channel_labels"]
        st.session_state.fermentation_history = load_fermentation_history()
        if st.session_state.fermentation_running and (_fermentation_thread is None or not _fermentation_thread.is_alive()):
            start_fermentation_thread(collector, channel_labels)
        col1, col2, col3 = st.columns(3)
        with col1:
            if not st.session_state.fermentation_running:
                if st.button("▶️ 开始发酵采集", type="primary", use_container_width=True):
                    st.session_state.fermentation_running = True
                    st.session_state.fermentation_start_time = time.time()
                    start_fermentation_thread(collector, channel_labels)
                    st.success("采集已启动")
            else:
                if st.button("⏹ 停止采集", type="secondary", use_container_width=True):
                    st.session_state.fermentation_running = False
                    stop_fermentation_thread()
                    st.success("采集已停止")
        with col2:
            if st.button("📥 手动采集一次", use_container_width=True):
                d = collector.read_modbus(channel_labels)
                if d:
                    rec = extract_fermentation_record(d)
                    hist = load_fermentation_history()
                    hist.append(rec)
                    save_fermentation_history(hist)
                    st.session_state.fermentation_history = hist
                    st.success("手动采集完成")
                else:
                    st.error("采集失败，Modbus无数据")
        with col3:
            if st.button("🧹 清空全部历史", use_container_width=True):
                st.session_state.fermentation_history = []
                save_fermentation_history([])
                st.success("历史已清空")
        st.divider()
        run_text = "运行中" if st.session_state.fermentation_running else "已停止"
        count = len(st.session_state.fermentation_history)
        m1, m2, m3 = st.columns(3)
        m1.metric("采集状态", run_text)
        m2.metric("总记录数", count)
        if st.session_state.fermentation_start_time:
            run_min = int((time.time() - st.session_state.fermentation_start_time) / 60)
            m3.metric("已运行分钟", run_min)
        else:
            m3.metric("已运行分钟", 0)
        if count > 0:
            latest = st.session_state.fermentation_history[-1]
            st.subheader("最新采集数据")
            l_cols = st.columns(4)
            for idx, field in enumerate(FERMENTATION_FIELDS):
                with l_cols[idx % 4]:
                    st.metric(field, latest.get(field, "—"))
            st.subheader("发酵变化曲线")
            chart_data = {}
            for f in FERMENTATION_FIELDS:
                chart_data[f] = [r.get(f) for r in st.session_state.fermentation_history]
            st.line_chart(chart_data)
            st.subheader("完整历史记录表")
            st.dataframe(st.session_state.fermentation_history[::-1], use_container_width=True)
            json_str = json.dumps(st.session_state.fermentation_history, ensure_ascii=False, indent=2)
            st.download_button("下载JSON历史", json_str, "fermentation_history.json", mime="application/json")
        else:
            st.info("暂无采集记录，点击开始采集")

    # ===================== 页面3：系统配置 =====================
    elif page == "系统配置":
        st.title("⚙️ 系统配置管理")
        config = st.session_state.config
        tab1, tab2, tab3 = st.tabs(["Modbus连接", "通道标签", "参数范围"])
        with tab1:
            st.subheader("Modbus TCP连接参数")
            c = config["connection"]
            h = st.text_input("IP地址", value=c["host"])
            p = st.number_input("端口", value=c["port"], min_value=1, max_value=65535)
            t = st.number_input("超时秒", value=c["timeout"], min_value=1, max_value=30)
            if st.button("测试Modbus连接"):
                test_mod = ModbusConfig(host=h, port=p, timeout=t)
                with ModbusTcpClient(host=h, port=p, timeout=t) as cli:
                    if cli.connect():
                        st.success("Modbus连接正常")
                    else:
                        st.error("Modbus连接失败")
        with tab2:
            st.subheader("通道标签配置")
            ch_labels = config["channel_labels"]
            del_idx = set()
            new_ch_list = []
            for idx, (ch, dev, param, mod) in enumerate(ch_labels):
                colA, colB, colC, colD, colE = st.columns([1,1,2,1,1])
                with colA:
                    nc = st.number_input(f"通道{idx+1}", value=ch, key=f"ch_{idx}")
                with colB:
                    nd = st.text_input(f"设备{idx+1}", value=dev, key=f"dev_{idx}")
                with colC:
                    np = st.text_input(f"参数{idx+1}", value=param, key=f"pr_{idx}")
                with colD:
                    nm = st.checkbox(f"可写{idx+1}", value=mod, key=f"md_{idx}")
                with colE:
                    if st.button("删除", key=f"del_{idx}"):
                        del_idx.add(idx)
                if idx not in del_idx:
                    new_ch_list.append((nc, nd, np, nm))
            st.divider()
            st.subheader("新增通道")
            ac, ad, ap, am = st.columns(4)
            new_c = ac.number_input("通道号", value=70)
            new_d = ad.text_input("设备名", value="BX-4")
            new_p = ap.text_input("参数名")
            new_m = am.checkbox("允许修改")
            if st.button("添加通道"):
                new_ch_list.append((new_c, new_d, new_p, new_m))
                st.info("添加成功，保存配置生效")
        with tab3:
            st.subheader("参数上下限配置")
            pr = config["param_ranges"]
            new_pr = {}
            for name, (minv, maxv) in pr.items():
                c1, c2, c3 = st.columns([2,1,1])
                c1.write(name)
                mn = c2.number_input(f"{name}最小值", value=minv, key=f"min_{name}")
                mx = c3.number_input(f"{name}最大值", value=maxv, key=f"max_{name}")
                new_pr[name] = [mn, mx]
            st.divider()
            st.subheader("新增参数范围")
            na, nm, nx = st.columns(3)
            add_name = na.text_input("参数名称")
            add_min = nm.number_input("最小值", value=0.0)
            add_max = nx.number_input("最大值", value=100.0)
            if st.button("添加参数范围") and add_name.strip():
                new_pr[add_name] = [add_min, add_max]
                st.info("已添加，保存生效")
        st.divider()
        save_col, down_col = st.columns(2)
        with save_col:
            if st.button("保存全部配置", type="primary", use_container_width=True):
                new_cfg = {
                    "connection": {"host": h, "port": p, "timeout": t},
                    "channel_labels": new_ch_list,
                    "param_ranges": new_pr
                }
                st.session_state.config = new_cfg
                save_config(new_cfg)
                st.rerun()
        with down_col:
            cfg_json = json.dumps(st.session_state.config, ensure_ascii=False, indent=4)
            st.download_button("下载modbus_config.json", cfg_json, "modbus_config.json")

    # ===================== 页面4：AI脚本助手（使用页面输入的DeepSeek） =====================
    elif page == "AI脚本助手":
        st.title("🤖 AI发酵控制脚本助手")
        # 读取侧边栏配置
        dk = st.session_state.deepseek_api_key
        dmodel = st.session_state.deepseek_model
        dtimeout = st.session_state.deepseek_timeout
        if not dk:
            st.error("⚠️ 请先在左侧侧边栏填写DeepSeek API Key并测试连接")
            st.stop()
        st.success(f"当前使用模型：{dmodel} | 超时：{dtimeout}s")
        config = st.session_state.config
        modbus_config = ModbusConfig(**config["connection"])
        collector = Collector(config=modbus_config)
        channel_labels = config["channel_labels"]
        param_ranges = config["param_ranges"]
        # 聊天窗口
        chat_box = st.container(height=400)
        with chat_box:
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    if msg["role"] == "assistant":
                        st.code(msg["content"], language="python")
                    else:
                        st.write(msg["content"])
        user_text = st.chat_input("描述发酵控制流程，例如：37度恒温发酵12小时")
        if user_text:
            st.session_state.chat_history.append({"role": "user", "content": user_text})
            with st.spinner("AI生成脚本中..."):
                code_out = generate_script(
                    prompt=user_text,
                    channel_labels=channel_labels,
                    param_ranges=param_ranges,
                    api_key=dk,
                    model=dmodel,
                    timeout=dtimeout
                )
            st.session_state.chat_history.append({"role": "assistant", "content": code_out})
            st.session_state.current_script = code_out
            st.rerun()
        st.divider()
        st.subheader("✏️ 脚本编辑、保存、执行")
        ed_code = st.text_area("脚本编辑器", value=st.session_state.current_script, height=320, key="script_editor")
        c1, c2, c3 = st.columns([1,1,1])
        with c1:
            s_name = st.text_input("脚本名称")
            s_desc = st.text_input("脚本描述")
            if st.button("保存脚本", use_container_width=True):
                ok, err = save_script(s_name, ed_code, s_desc)
                if ok:
                    st.success("保存成功")
                    st.rerun()
                else:
                    st.error(err)
        with c2:
            if st.button("执行当前脚本", use_container_width=True):
                with st.spinner("运行脚本..."):
                    res = execute_script(ed_code, collector)
                st.subheader("执行结果")
                st.code(res)
        with c3:
            if st.button("清空聊天记录", use_container_width=True):
                st.session_state.chat_history = []
                st.session_state.current_script = ""
                st.rerun()

if __name__ == "__main__":
    main()

