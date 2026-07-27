"""
common_order_items.py
1장 문제01에서 확정한 표준 순매출 라인 아이템 빌더.
2장~10장의 모든 스크립트가 이 모듈의 build_order_items()를 재사용한다.
"""

import pandas as pd


def build_order_items(
    orders_path="data/orders.csv",
    items_path="data/order_items.csv"
):
    """
    정제 순서: 중복 제거 -> 날짜 파싱(coerce) -> 기간 필터(2024 상반기)
             -> order_id 기준 inner join -> unit_price 결측 제외 -> line_amount 파생
    반환: 라인 아이템 단위 데이터프레임 (is_net 열: canceled/returned 제외 여부)
    """
    orders = pd.read_csv(orders_path)
    items = pd.read_csv(items_path)

    orders = orders.drop_duplicates()
    items = items.drop_duplicates()

    # print("items.shape[0] : ",items.shape[0], "라인 아이템 건수(중복제거)")

    orders["order_datetime"] = pd.to_datetime(orders["order_datetime"], errors="coerce")
    period_mask = (orders["order_datetime"] >= "2024-01-01") & (
        orders["order_datetime"] < "2024-07-01"
    )
    orders_in_period = orders[period_mask].copy()

    order_items = orders_in_period.merge(items, on="order_id", how="inner")    
    order_items = order_items.dropna(subset=["unit_price"])
    
    # line_amount(실구매가) = 주문수량 * 판매단가 * (1 - 할인율) 
    order_items["line_amount"] = (
        order_items["quantity"] * order_items["unit_price"] * (1 - order_items["discount"])
    )

    # 정상건
    order_items["is_net"] = ~order_items["status"].isin(["canceled", "returned"])

    return order_items


def net_order_items(orders_path="data/orders.csv", items_path="data/order_items.csv"):
    """취소·반품이 제외된 순매출 라인만 반환 (자주 쓰는 축약형)"""
    order_items = build_order_items(orders_path, items_path)
    return order_items[order_items["is_net"]].copy()


def clean_category(series):
    """카테고리 앞뒤 공백 오염 정제 공통 함수 (5장~에서 재사용)"""
    return series.astype(str).str.strip()


def clean_price(series):
    """products.price 문자열 오염 정제: 콤마·원 제거 후 숫자화, 형식오염은 NaN"""
    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("원", "", regex=False)
        .str.strip()
    )
    return pd.to_numeric(cleaned, errors="coerce")
