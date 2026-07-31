# -*- coding: utf-8 -*-
"""
common.py
  30문제 전체가 공유하는 가상 데이터 생성 모듈.
  - customers  : 고객 마스터 (customer_id, gender_ko, age)
  - orders     : 주문 헤더 (order_id, customer_id, order_datetime, channel, status)
  - order_items(oi) : 주문 상품 라인 (order_id, category, amount)
  실제 수업/원본 데이터가 없으므로 seed 고정 난수로 재현 가능한 데이터를 만든다.
"""
import numpy as np
import pandas as pd

SEED = 42
N_CUSTOMERS = 200
N_ORDERS = 1500
START = "2024-01-01"
END = "2024-06-30"

CHANNELS = ["web", "app", "offline"]
STATUSES = ["completed", "completed", "completed", "canceled", "returned"]  # completed 가중치↑
CATEGORIES = ["의류", "가전", "식품", "뷰티", "잡화"]


def generate_customers(n=N_CUSTOMERS, seed=SEED):
    rng = np.random.default_rng(seed)
    customer_id = np.arange(1, n + 1)
    gender_ko = rng.choice(["남", "여"], size=n)
    age = rng.integers(18, 70, size=n)
    return pd.DataFrame({
        "customer_id": customer_id,
        "gender_ko": gender_ko,
        "age": age,
    })


def generate_orders(customers: pd.DataFrame, n=N_ORDERS, seed=SEED):
    rng = np.random.default_rng(seed + 1)
    order_id = np.arange(1, n + 1)
    customer_id = rng.choice(customers["customer_id"], size=n)
    start_ts = pd.Timestamp(START).value // 10**9
    end_ts = pd.Timestamp(END).value // 10**9
    order_datetime = pd.to_datetime(
        rng.integers(start_ts, end_ts, size=n), unit="s"
    )
    channel = rng.choice(CHANNELS, size=n, p=[0.5, 0.35, 0.15])
    status = rng.choice(STATUSES, size=n)
    return pd.DataFrame({
        "order_id": order_id,
        "customer_id": customer_id,
        "order_datetime": order_datetime,
        "channel": channel,
        "status": status,
    })


def generate_order_items(orders: pd.DataFrame, seed=SEED):
    # 주문 1건당 1~3개 라인
    rng = np.random.default_rng(seed + 2)
    rows = []
    line_id = 1
    for oid in orders["order_id"]:
        n_lines = rng.integers(1, 4)
        for _ in range(n_lines):
            category = rng.choice(CATEGORIES)
            qty = rng.integers(1, 4)
            unit_price = rng.integers(5, 80) * 1000
            rows.append((line_id, oid, category, qty, unit_price * qty))
            line_id += 1
    oi = pd.DataFrame(rows, columns=["line_id", "order_id", "category", "qty", "amount"])
    return oi


def build_net_sales_ledger(verbose=False):
    """
    orders + order_items 를 합쳐 원장을 만들고,
    canceled/returned 를 제외한 net_ledger, 전체를 포함한 gross_ledger 를 반환.
    """
    customers = generate_customers()
    orders = generate_orders(customers)
    oi = generate_order_items(orders)

    gross_ledger = oi.merge(orders, on="order_id", how="inner")
    gross_ledger = gross_ledger.rename(columns={"amount": "line_amount"})
    net_ledger = gross_ledger[~gross_ledger["status"].isin(["canceled", "returned"])].copy()

    if verbose:
        print(f"[데이터 생성] 고객 {len(customers)}명 / 주문 {len(orders)}건 / 주문라인 {len(oi)}건")
        print(f"[원장] gross={len(gross_ledger)}행, net={len(net_ledger)}행")

    return {
        "customers": customers,
        "orders": orders,
        "oi": oi,
        "gross_ledger": gross_ledger,
        "net_ledger": net_ledger,
    }


def get_merged(seed=SEED):
    """oi + customers 를 merge 한 데이터(m) - 3장(merge) 문제에서 사용."""
    data = build_net_sales_ledger()
    m = data["oi"].merge(data["orders"], on="order_id", how="inner")
    m = m.merge(data["customers"], on="customer_id", how="inner")
    return m, data
