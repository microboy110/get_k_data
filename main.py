import asyncio
import httpx
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
import numpy as np
import os,time
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
						# 获取当前毫秒时间戳
                        current_ms = int(time.time() * 1000)
						# 计算 24 小时前的毫秒数 (24 * 60 * 60 * 1000)
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
                        
                        self.active_symbols = df['symbol'].tolist()
                        print(f"扫描完成！当前监控标的：{self.active_symbols[:5]}...等50个")
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
    data_api = "https://fapi.binance.com/futures/data"

    async with httpx.AsyncClient() as client:
        tasks = [
            get_data(client, f"{fapi}/fapi/v1/klines", {"symbol": symbol, "interval": "5m", "limit": 200}, "K线"),
            get_data(client, f"{data_api}/openInterestHist", {"symbol": symbol, "period": "5m", "limit": 200}, "持仓")
        ]
        klines, oi_raw = await asyncio.gather(*tasks)

        if not klines or len(klines) < 110: return None

        df = pd.DataFrame(klines).iloc[:, :6]
        df.columns = ['ts', 'open', 'high', 'low', 'close', 'vol']
        df = df.apply(pd.to_numeric)

        # 均线计算
        df['sma7'] = df['close'].rolling(7).mean()
        df['sma14'] = df['close'].rolling(14).mean()
        df['sma21'] = df['close'].rolling(21).mean()
        
        # 趋势强度与一致性
        df['sma7_slope'] = df['sma7'] - df['sma7'].shift(3)
        df['sma14_slope'] = df['sma14'] - df['sma14'].shift(3)
        df['sma21_slope'] = df['sma21'] - df['sma21'].shift(3)
        slope_consistency = (df['sma7_slope'] > 0) & (df['sma14_slope'] > 0) & (df['sma21_slope'] > 0)
        
        # 动态阈值 (ATR)
        tr = pd.concat([(df['high'] - df['low']), (df['high'] - df['close'].shift()).abs(), (df['low'] - df['close'].shift()).abs()], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        dynamic_threshold = atr / df['close'] * 0.3
        
        spacing_7_14 = (df['sma7'] - df['sma14']).abs() / df['close']
        
        # 排列判断
        bullish = (df['close'] > df['sma7']) & (df['sma7'] > df['sma14']) & (df['sma14'] > df['sma21'])
        
        # 成交量确认
        vol_ratio = df['vol'] / df['vol'].rolling(14).mean()
        
        # --- 信号生成：同一个波段只触发一次 ---
        # 原始条件满足状态
        raw_condition = (bullish) & (slope_consistency) & (spacing_7_14 > dynamic_threshold) & (vol_ratio > 1.2)
        
        # 上升沿触发：当前为真且上一个为假
        df['is_entry'] = raw_condition & (~raw_condition.shift(1).fillna(False))

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
                    "sma7": clean_val(row.get('sma7')),
					"sma14": clean_val(row.get('sma14')),
                    "sma21": clean_val(row.get('sma21')),
                    "vol_ratio": clean_val(row.get('vol_ratio'))
                }
            }

        if is_init:
            return {"symbol": symbol, "type": "INIT", "data": [format_row(i) for i in range(len(df)) if format_row(i)]}
        else:
            return {"symbol": symbol, "type": "UPDATE", "data": [format_row(len(df)-2),format_row(len(df)-1)]}

# --- WebSocket 路由 ---
@app.websocket("/ws/strategy")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        # 初始等待标的扫描完成
        while not symbol_manager.active_symbols:
            await asyncio.sleep(1)
            
        current_symbols = symbol_manager.active_symbols
        init_results = await asyncio.gather(*[fetch_binance_data(s, True) for s in current_symbols])
        await websocket.send_json([r for r in init_results if r])
        
        while True:
            await asyncio.sleep(60) 
            # 实时更新使用当前最新的标的列表
            active_list = symbol_manager.active_symbols
            update_results = await asyncio.gather(*[fetch_binance_data(s, False) for s in active_list])
            if update_results:
                await websocket.send_json([r for r in update_results if r])
    except Exception as e: 
        print(f"WS Error: {e}")

@app.get("/")
async def get_index():
    return HTMLResponse("<h1>Hello World !</h1>")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
