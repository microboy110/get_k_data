import asyncio
import httpx
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
import numpy as np
import os, time
import math
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

sem = asyncio.Semaphore(30) 

# --- 全局标的管理类 ---
class SymbolManager:
    def __init__(self):
        self.active_symbols = []
        self.symbol_info = {}  # 存储 symbol -> {openTime, last_update_time}
        self.update_interval = 3600  # 1小时更新一次

    async def update_symbols_loop(self):
        """后台定时扫描任务"""
        while True:
            try:
                print("正在扫描活跃标的...")
                url = "https://fapi.binance.com/fapi/v1/ticker/24hr"
                async with httpx.AsyncClient() as client:
                    resp = await client.get(url, timeout=10.0)
                    if resp.status_code == 200:
                        data = resp.json()
                        df = pd.DataFrame(data)
                        
                        # 1. 基础过滤：仅限 USDT 永续合约
                        df = df[df['symbol'].str.endswith('USDT')]
                        # --- 新增过滤逻辑：过滤掉 openTime 在 24 小时以前的标的 ---
                        current_ms = int(time.time() * 1000)
                        twenty_four_hours_ago = current_ms - 86400000 - 600000

                        # 确保 openTime 是数值类型并进行过滤
                        df['openTime'] = pd.to_numeric(df['openTime'])
                        df = df[df['openTime'] >= twenty_four_hours_ago]
                        # ---------------------------------------------------
                        # 数值化
                        df['quoteVolume'] = pd.to_numeric(df['quoteVolume'])
                        df['high'] = pd.to_numeric(df['highPrice'])
                        df['low'] = pd.to_numeric(df['lowPrice'])
                        
                        # 2. 第一阶段：按成交额排序，取前 200
                        df = df.sort_values(by='quoteVolume', ascending=False).head(200)
                        
                        # 3. 第二阶段：计算振幅并排序，取前 50
                        df['amplitude'] = (df['high'] - df['low']) / df['low']
                        df = df.sort_values(by='amplitude', ascending=False).head(50)

                        new_symbols = df['symbol'].tolist()

                        # 更新 active_symbols 并记录每个symbol的openTime
                        self.active_symbols = new_symbols
                        for _, row in df.iterrows():
                            self.symbol_info[row['symbol']] = {
                                'openTime': row['openTime'],
                                'last_update_time': current_ms
                            }

                        print(f"扫描完成！当前监控标的：{new_symbols[:5]}...等50个")
            except Exception as e:
                print(f"扫描标的出错: {e}")
            
            await asyncio.sleep(self.update_interval)

symbol_manager = SymbolManager()

@app.on_event("startup")
async def startup_event():
    # 启动时立即运行一次扫描，并开启后台循环
    asyncio.create_task(symbol_manager.update_symbols_loop())

# --- 辅助工具 ---
def clean_val(val):
    if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
        return None
    return float(val)

async def get_data(client, url, params, label):
    async with sem:
        try:
            resp = await client.get(url, params=params, timeout=5.0)
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            pass
        return []

# --- 策略核心逻辑 ---
async def fetch_binance_data(symbol: str, is_init=False):
    fapi = "https://fapi.binance.com"
    #data_api = "https://fapi.binance.com/futures/data"

    async with httpx.AsyncClient() as client:
        tasks = [
            get_data(client, f"{fapi}/fapi/v1/klines", {"symbol": symbol, "interval": "5m", "limit": 200}, "K线")
        ]
        klines = await asyncio.gather(*tasks)
        klines = klines[0]

        # ⚠️ 保持原逻辑：不足110根直接返回None
        if not klines or len(klines) < 110:
            print(f"[INFO] Not enough data for {symbol}: {len(klines)} klines, skipping...")
            return None

        df = pd.DataFrame(klines).iloc[:, :6]
        df.columns = ['ts', 'open', 'high', 'low', 'close', 'vol']
        df = df.apply(pd.to_numeric)

        # --- 改为 EMA ---
        df['ema7'] = df['close'].ewm(span=7, adjust=False).mean()
        df['ema14'] = df['close'].ewm(span=14, adjust=False).mean()
        df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()

        # 趋势强度与一致性
        df['ema7_slope'] = df['ema7'] - df['ema7'].shift(3)
        df['ema14_slope'] = df['ema14'] - df['ema14'].shift(3)
        df['ema21_slope'] = df['ema21'] - df['ema21'].shift(3)
        slope_consistency = (df['ema7_slope'] > 0) & (df['ema14_slope'] > 0) & (df['ema21_slope'] > 0)

        # 动态阈值 (ATR)
        tr = pd.concat([(df['high'] - df['low']), (df['high'] - df['close'].shift()).abs(), (df['low'] - df['close'].shift()).abs()], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        dynamic_threshold = atr / df['close'] * 0.3

        spacing_7_14 = (df['ema7'] - df['ema14']).abs() / df['close']

        # 排列判断
        bullish = (df['close'] > df['ema7']) & (df['ema7'] > df['ema14']) & (df['ema14'] > df['ema21'])

        # 成交量确认
        vol_ratio = df['vol'] / df['vol'].rolling(14).mean()
		
        # --- 新增策略条件：当前收盘价 > 前三根K线最高价 ---
        df['prev3_max_high'] = df['high'].shift(1).rolling(window=3).max()  # 前三根K线的最高价（不包含当前）
        df['close_gt_prev3_high'] = df['close'] > df['prev3_max_high']		

        # --- 信号生成：同一个波段只触发一次 ---
        raw_condition = (bullish) & (slope_consistency) & (spacing_7_14 > dynamic_threshold) & (vol_ratio > 1.2)& (df['close_gt_prev3_high'])
        df['is_entry'] = raw_condition & (~raw_condition.shift(1).fillna(False).infer_objects())

        def format_row(idx):
            if idx < 0 or idx >= len(df): return None
            row = df.iloc[idx]
            return {
                "time": int(row['ts'] / 1000),
                "open": clean_val(row['open']),
                "high": clean_val(row['high']),
                "low": clean_val(row['low']),
                "close": clean_val(row['close']),
                "is_entry": bool(row['is_entry']),
                "metrics": {
                    "ema7": clean_val(row.get('ema7')),
                    "ema14": clean_val(row.get('ema14')),
                    "ema21": clean_val(row.get('ema21')),
                    "vol_ratio": clean_val(row.get('vol_ratio'))
                }
            }

        if is_init:
            return {"symbol": symbol, "type": "INIT", "data": [format_row(i) for i in range(len(df)) if format_row(i)]}
        else:
            return {"symbol": symbol, "type": "UPDATE", "data": [format_row(len(df)-2), format_row(len(df)-1)]}

# --- WebSocket 路由 ---
@app.websocket("/ws/strategy")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        # 初始等待标的扫描完成
        while not symbol_manager.active_symbols:
            await asyncio.sleep(1)
        
        current_symbols = symbol_manager.active_symbols.copy()
        initialized_symbols = set()  # 记录哪些symbol已经初始化过（已发送INIT）

        # 💥 第一次全量 INIT
        init_results = await asyncio.gather(*[fetch_binance_data(s, True) for s in current_symbols])
        valid_inits = [r for r in init_results if r]
        if valid_inits:
            await websocket.send_json(valid_inits)
            initialized_symbols.update([r["symbol"] for r in valid_inits])

        while True:
            await asyncio.sleep(60)
            
            new_active_list = symbol_manager.active_symbols
            added = set(new_active_list) - initialized_symbols
            removed = initialized_symbols - set(new_active_list)
            unchanged = set(new_active_list) & initialized_symbols

            # 🧹 Step 1: 清理旧symbols — 从 initialized_symbols 中移除
            if removed:
                print(f"[INFO] 移除监控symbol: {removed}")
                initialized_symbols -= removed  # 本地状态清理

            # 🔥 Step 2: 对新增symbol，尝试INIT推送（需满足>=110根K线）
            if added:
                print(f"[INFO] 新增监控symbol: {added}")
                init_tasks = []
                for s in added:
                    info = symbol_manager.symbol_info.get(s)
                    if not info:
                        continue

                    open_time = info['openTime']
                    current_ms = int(time.time() * 1000)
                    elapsed_minutes = (current_ms - open_time) / (60 * 1000)
                    estimated_klines = int(elapsed_minutes / 5)  # 每5分钟一根

                    if estimated_klines < 110:
                        print(f"[SKIP] Symbol {s} too new: only ~{estimated_klines} klines expected (<110)")
                        continue

                    init_tasks.append(fetch_binance_data(s, True))

                if init_tasks:
                    new_init_results = await asyncio.gather(*init_tasks)
                    valid_new_inits = [r for r in new_init_results if r]
                    if valid_new_inits:
                        await websocket.send_json(valid_new_inits)
                        initialized_symbols.update([r["symbol"] for r in valid_new_inits])

            # 🔄 Step 3: 对仍存在的symbol发送 UPDATE（最近2根）
            if unchanged:
                update_results = await asyncio.gather(*[fetch_binance_data(s, False) for s in unchanged])
                valid_updates = [r for r in update_results if r]
                if valid_updates:
                    await websocket.send_json(valid_updates)

    except Exception as e: 
        print(f"WS Error: {e}")

@app.get("/")
async def get_index():
    return HTMLResponse("<h1>Hello World!</h1>")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
