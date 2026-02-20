import asyncio
import httpx
import pandas as pd
import numpy as np
import os
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

# --- 新增：简单主页，用于 Zeabur 健康检查 ---
@app.get("/")
async def get_index():
    return HTMLResponse("""
        <html>
            <body style="display:flex;flex-direction:column;justify-content:center;align-items:center;height:100vh;background:#131722;color:white;font-family:sans-serif;">
                <h1 style="color:#2962ff;">Hello World!</h1>               
                <div style="font-size:12px;color:#5d606b;">Status: Online</div>
            </body>
        </html>
    """)


def clean_val(val):
    """将 Python/Pandas 的 NaN/Inf 转换为 JSON 兼容的 None (null)"""
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
            print(f"[{label}] 请求异常: {e}")
        return []

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

        df_oi = pd.DataFrame(oi_raw)
        if not df_oi.empty:
            df_oi = df_oi[['timestamp', 'sumOpenInterest']].apply(pd.to_numeric)
            df_oi.rename(columns={'timestamp': 'ts', 'sumOpenInterest': 'oi'}, inplace=True)
            df = pd.merge(df, df_oi, on='ts', how='left').ffill()
        else:
            df['oi'] = 0

        # --- 修改后的策略逻辑：趋势识别 + 能量确认 ---
        # 1. 计算三条均线（趋势识别核心）
        df['sma7'] = df['close'].rolling(7).mean()
        df['sma14'] = df['close'].rolling(14).mean()
        df['sma21'] = df['close'].rolling(21).mean()
        
        # 2. 计算均线间距（趋势强度确认）
        df['spacing_7_14'] = abs(df['sma7'] - df['sma14']) / df['close']
        df['spacing_14_21'] = abs(df['sma14'] - df['sma21']) / df['close']
        
        # 3. 计算均线斜率方向一致性（趋势稳定性）
        df['sma7_slope'] = df['sma7'] - df['sma7'].shift(3)
        df['sma14_slope'] = df['sma14'] - df['sma14'].shift(3)
        df['sma21_slope'] = df['sma21'] - df['sma21'].shift(3)
        df['slope_consistency'] = ((df['sma7_slope'] > 0) & 
                                  (df['sma14_slope'] > 0) & 
                                  (df['sma21_slope'] > 0)).astype(int)
        
        # 4. 计算成交量指标（能量确认核心）
        df['vol_ma'] = df['vol'].rolling(14).mean()
        df['vol_ratio'] = df['vol'] / df['vol_ma']
        
        # 5. 均线排列判断（趋势结构确认）
        df['bullish_arrangement'] = ((df['close'] > df['sma7']) & 
                                    (df['sma7'] > df['sma14']) & 
                                    (df['sma14'] > df['sma21'])).astype(int)
        
        df['bearish_arrangement'] = ((df['close'] < df['sma7']) & 
                                    (df['sma7'] < df['sma14']) & 
                                    (df['sma14'] < df['sma21'])).astype(int)
        
        # 6. 计算ATR用于动态间距阈值（自适应优化）
        tr = pd.concat([(df['high'] - df['low']), 
                        (df['high'] - df['close'].shift()).abs(), 
                        (df['low'] - df['close'].shift()).abs()], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()
        df['dynamic_spacing_threshold'] = df['atr'] / df['close'] * 0.3
        
        # 7. 策略信号生成 - 趋势识别条件
        trend_condition = (
            (df['bullish_arrangement'] == 1) &                    # 多头排列
            (df['slope_consistency'] == 1) &                      # 均线同向向上
            (df['spacing_7_14'] > df['dynamic_spacing_threshold']) &  # 动态间距确认
            (df['spacing_14_21'] > df['dynamic_spacing_threshold'] * 0.4)   # 动态间距确认
        )
        
        # 8. 能量确认条件
        energy_condition = (
            df['vol_ratio'] > 1.2                            # 成交量放大
            #(df['vol_ratio'] > 1.5) &                             # 成交量放大
            #(df['oi'] > df['oi'].shift(1)) &                      # 持仓量增加
            #(df['oi'] > df['oi'].rolling(14).mean() * 1.1)        # 持仓量相对平均水平增加
        )
        
        # 9. 综合入场信号
        df['is_entry'] = trend_condition & energy_condition

        def format_row(idx):
            if idx < 0 or idx >= len(df): return None
            row = df.iloc[idx]
            return {
                "time": int(row['ts'] / 1000),
                "open": clean_val(row['open']),
                "high": clean_val(row['high']),
                "low": clean_val(row['low']),
                "close": clean_val(row['close']),
                "is_entry": bool(row['is_entry']) if not pd.isna(row['is_entry']) else False,
                "metrics": {
                    "sma7": clean_val(row.get('sma7')),
                    "sma14": clean_val(row.get('sma14')),
                    "sma21": clean_val(row.get('sma21')),
                    "vol_ratio": clean_val(row.get('vol_ratio')),
                    "oi_current": clean_val(row.get('oi'))
                }
            }

        if is_init:
            return {"symbol": symbol, "type": "INIT", "data": [format_row(i) for i in range(len(df)) if format_row(i)]}
        else:
            return {"symbol": symbol, "type": "UPDATE", "data": [format_row(len(df)-2),format_row(len(df)-1)]}

@app.websocket("/ws/strategy/{symbols}")
async def websocket_endpoint(websocket: WebSocket, symbols: str):
    await websocket.accept()
    symbol_list = [s.strip().upper() for s in symbols.split(",")]
    try:
        init_results = await asyncio.gather(*[fetch_binance_data(s, True) for s in symbol_list])
        await websocket.send_json([r for r in init_results if r])
        while True:
            await asyncio.sleep(60) 
            update_results = await asyncio.gather(*[fetch_binance_data(s, False) for s in symbol_list])
            if update_results:
                await websocket.send_json([r for r in update_results if r])
    except Exception as e: print(f"WS Error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
