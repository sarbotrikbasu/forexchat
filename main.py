from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
import yfinance as yf
import pandas as pd
from typing import Optional

app = FastAPI(
    title="Forex Indicators API",
    description="API to calculate Trend, RSI, and EMA for any Forex ticker using yfinance. Designed for Custom GPT Actions.",
    version="1.0.0",
)

def format_ticker(symbol: str) -> str:
    # yfinance forex symbols typically have an =X suffix (e.g., EURUSD=X)
    symbol = symbol.upper().strip()
    if len(symbol) == 6 and "=" not in symbol:
        return f"{symbol}=X"
    return symbol

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    
    # Wilder's moving average
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

class IndicatorResponse(BaseModel):
    symbol: str
    indicator: str
    interval: str
    period: int
    latest_value: float

class TrendResponse(BaseModel):
    symbol: str
    interval: str
    TrendStart: str
    TrendEnd: str
    TrendDirection: str
    Fib1: float
    Fib2: float
    Fib3: float
    Fib4: float
    Fib5: float

@app.get("/rsi", response_model=IndicatorResponse, summary="Calculate RSI")
def get_rsi(
    symbol: str = Query(..., description="Forex ticker symbol (e.g., EURUSD or EURUSD=X)"),
    interval: str = Query("1h", description="Timeframe valid in yfinance (e.g., 5m, 15m, 1h, 1d)"),
    period: int = Query(14, description="Period for RSI calculation (default: 14)")
):
    try:
        yf_symbol = format_ticker(symbol)
        stock = yf.Ticker(yf_symbol)
        
        # Max for intraday data (under 1d) requires smaller period windows sometimes, but 60d generally covers 5m, 15m, 1h requests.
        hist = stock.history(period="60d", interval=interval)
        if hist.empty:
            raise HTTPException(status_code=404, detail=f"Ticker '{yf_symbol}' not found or interval invalid.")
        
        prices = hist['Close']
        if len(prices) < period * 2:
            raise HTTPException(status_code=400, detail="Not enough data to calculate reliable RSI.")
            
        rsi_series = calculate_rsi(prices, period)
        latest_rsi = rsi_series.iloc[-1]
        
        if pd.isna(latest_rsi):
            raise HTTPException(status_code=500, detail="Failed to calculate RSI value.")
            
        return {
            "symbol": yf_symbol,
            "indicator": "RSI",
            "interval": interval,
            "period": period,
            "latest_value": round(float(latest_rsi), 4)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ema", response_model=IndicatorResponse, summary="Calculate EMA")
def get_ema(
    symbol: str = Query(..., description="Forex ticker symbol (e.g., EURUSD or EURUSD=X)"),
    interval: str = Query("1h", description="Timeframe valid in yfinance (e.g., 5m, 15m, 1h, 1d)"),
    period: int = Query(20, description="Period for EMA calculation (default: 20)")
):
    try:
        yf_symbol = format_ticker(symbol)
        stock = yf.Ticker(yf_symbol)
        hist = stock.history(period="60d", interval=interval)
        if hist.empty:
            raise HTTPException(status_code=404, detail=f"Ticker '{yf_symbol}' not found or interval invalid.")
            
        prices = hist['Close']
        if len(prices) < period:
            raise HTTPException(status_code=400, detail="Not enough data to calculate EMA.")
            
        ema_series = prices.ewm(span=period, adjust=False).mean()
        latest_ema = ema_series.iloc[-1]
        
        if pd.isna(latest_ema):
            raise HTTPException(status_code=500, detail="Failed to calculate EMA value.")
            
        return {
            "symbol": yf_symbol,
            "indicator": "EMA",
            "interval": interval,
            "period": period,
            "latest_value": round(float(latest_ema), 5)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/trend", response_model=TrendResponse, summary="Calculate Trend and Fibonacci using yfinance")
def get_trend(
    symbol: str = Query(..., description="Forex ticker symbol (e.g., EURUSD or EURUSD=X)"),
    interval: str = Query("1h", description="Timeframe valid in yfinance (e.g., 5m, 15m, 1h)")
):
    try:
        yf_symbol = format_ticker(symbol)
        stock = yf.Ticker(yf_symbol)
        
        # Using 7d timeframe originally for trend calculation as per MT5
        df = stock.history(period="7d", interval=interval)
        
        if df.empty or len(df) < 5:
            # Maybe the market is closed or 7d isn't enough, fetch more if intraday
            if interval.endswith('m') or interval.endswith('h'):
                df = stock.history(period="30d", interval=interval)
                
            if df.empty or len(df) < 5:
                raise HTTPException(status_code=404, detail=f"Not enough data to calculate trend for '{yf_symbol}'.")
        
        df.reset_index(inplace=True)
        time_col = 'Datetime' if 'Datetime' in df.columns else 'Date'
        df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', time_col: 'time'}, inplace=True)
        
        if df['time'].dt.tz is not None:
             df["time"] = df["time"].dt.tz_convert(None)
             
        # CORE CALCULATIONS PORTED FROM APIbackup.py
        df["ind"] = ((df["close"] - df["open"]) / df["open"]) * 100
        df["cum"] = df["ind"].cumsum()
        df["mid"] = df["cum"] / 2
        df["X"] = df["cum"].diff()

        df["isRedLocal"] = False
        for i in range(1, len(df)):
            X = df.loc[i, "X"]
            prev_mid = df.loc[i - 1, "mid"]
            if pd.notna(X) and pd.notna(prev_mid):
                if (
                    ((X > 0 and prev_mid < 0) or
                     (X < 0 and prev_mid > 0))
                    and abs(X) > abs(prev_mid)
                ):
                    df.loc[i, "isRedLocal"] = True

        reference_tables = {}
        for ref_idx in range(len(df) - 1):
            rows, cum_val, prev_mid, prev_cum = [], None, None, None
            for i in range(ref_idx, len(df)):
                ind = df.loc[i, "ind"]
                cum_val = ind if cum_val is None else cum_val + ind
                mid = cum_val / 2
                red = False
                if prev_mid is not None:
                    X = cum_val - prev_cum
                    if (
                        ((X > 0 and prev_mid < 0) or
                         (X < 0 and prev_mid > 0))
                        and abs(X) > abs(prev_mid)
                    ):
                        red = True
                rows.append({"RedLocal": red})
                prev_cum, prev_mid = cum_val, mid
                
            red_local_any = any(r["RedLocal"] for r in rows[1:])
            reference_tables[df.loc[ref_idx, "time"]] = not red_local_any

        # Find first ref_time where finalGreen is True
        trend_start = None
        for ref_idx in range(len(df) - 1):
            ref_time = df.loc[ref_idx, "time"]
            if reference_tables.get(ref_time):
                trend_start = ref_time
                break

        if trend_start is None:
            return TrendResponse(
                symbol=yf_symbol, interval=interval,
                TrendStart="None", TrendEnd="None", TrendDirection="None",
                Fib1=0.0, Fib2=0.0, Fib3=0.0, Fib4=0.0, Fib5=0.0
            )

        start_idx = df.index[df["time"] == trend_start][0]

        df_after = df.loc[start_idx:].copy()
        df_after["cum_from_start"] = df_after["ind"].cumsum()

        max_idx = df_after["cum_from_start"].abs().idxmax()
        Trend_Direction = (
            "Uptrend"
            if df.loc[max_idx, "close"] > df.loc[start_idx, "close"]
            else "Downtrend"
        )

        FIB_RATIOS = {"Fib1": 0.236, "Fib2": 0.382, "Fib3": 0.5, "Fib4": 0.618, "Fib5": 0.786}
        start_row = df.loc[start_idx]
        max_row = df.loc[max_idx]

        if Trend_Direction == "Uptrend":
            high, low = max_row["high"], start_row["low"]
            rng = high - low
            fib_levels = {k: high - v * rng for k, v in FIB_RATIOS.items()}
        else:
            high, low = start_row["high"], max_row["low"]
            rng = high - low
            fib_levels = {k: low + v * rng for k, v in FIB_RATIOS.items()}

        fib_signal = {"Fib1": 0, "Fib2": 0, "Fib3": 0, "Fib4": 0, "Fib5": 0}
        n2 = df.iloc[-3]
        n1 = df.iloc[-2]

        fib_name, fib_price = min(fib_levels.items(), key=lambda x: abs(n2["close"] - x[1]))

        if Trend_Direction == "Uptrend":
            if (n2["close"] < n2["open"] and
                abs(n2["close"] - fib_price) <= 0.0005 * fib_price and
                n1["close"] > n1["open"] and
                n1["close"] > n2["close"]):
                fib_signal[fib_name] = 1
        else:
            if (n2["close"] > n2["open"] and
                abs(n2["close"] - fib_price) <= 0.0005 * fib_price and
                n1["close"] < n1["open"] and
                n1["close"] < n2["close"]):
                fib_signal[fib_name] = 1

        return TrendResponse(
            symbol=yf_symbol,
            interval=interval,
            TrendStart=str(df.loc[start_idx, "time"]),
            TrendEnd=str(df.loc[max_idx, "time"]),
            TrendDirection=Trend_Direction,
            Fib1=fib_signal["Fib1"],
            Fib2=fib_signal["Fib2"],
            Fib3=fib_signal["Fib3"],
            Fib4=fib_signal["Fib4"],
            Fib5=fib_signal["Fib5"]
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health", summary="Health Check")
def health_check():
    return {"status": "ok"}
