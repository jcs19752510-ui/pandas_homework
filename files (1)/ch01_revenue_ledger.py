"""
1장. 매출 성과 진단과 채널 리포트
문제 01: 신뢰할 수 있는 매출 원장 만들기
문제 02: 상반기 성장 기여 분해 브리핑

실행: python ch01_revenue_ledger.py
전제: data/orders.csv, data/order_items.csv 가 이 파일과 같은 폴더의 data/ 아래에 있어야 함
"""

import pandas as pd
import numpy as np

pd.set_option("display.float_format", lambda x: f"{x:,.0f}")


# ============================================================
# 문제 01. 신뢰할 수 있는 매출 원장 만들기
# ============================================================
def build_ledger(orders_path="data/orders.csv", items_path="data/order_items.csv"):
    """
    정제 순서: 중복 제거 -> 날짜 파싱 -> 기간 필터 -> 상태 필터
    각 단계 손실 건수를 별도로 세어 대사표 재료로 남긴다.
    """
    orders = pd.read_csv(orders_path)
    items = pd.read_csv(items_path)

    report = {}  # 단계별 손실 기록

    # --- 1) 중복 제거 -------------------------------------------------
    n0_orders, n0_items = len(orders), len(items)
    orders = orders.drop_duplicates()
    items = items.drop_duplicates()
    report["orders 중복 제거"] = n0_orders - len(orders)
    report["order_items 중복 제거"] = n0_items - len(items)

    # --- 2) 날짜 파싱(errors="coerce") ---------------------------------
    orders["order_datetime"] = pd.to_datetime(orders["order_datetime"], errors="coerce")
    n_bad_date = orders["order_datetime"].isna().sum()
    report["order_datetime 파싱 실패(결측 포함)"] = int(n_bad_date)

    # --- 3) 기간 필터 (2024년 상반기, 1/1~6/30) -------------------------
    before = len(orders)
    period_mask = (orders["order_datetime"] >= "2024-01-01") & (
        orders["order_datetime"] < "2024-07-01"
    )
    orders_in_period = orders[period_mask].copy()
    report["기간(2024 상반기) 밖 제거"] = before - len(orders_in_period)

    # --- 4) 주문×상품 결합 (order_id 기준 inner join) --------------------
    # inner join을 쓰는 이유: 주문 헤더가 없는 라인, 주문 라인이 없는 헤더는
    # 매출 계산에 쓸 수 없는 불완전 레코드이므로 자연스럽게 제외된다.
    ledger_all = orders_in_period.merge(items, on="order_id", how="inner")

    # --- 5) 라인 매출 계산 ------------------------------------------
    # unit_price 결측 라인은 매출 계산 불가 -> 별도 집계 후 제외
    n_price_na = ledger_all["unit_price"].isna().sum()
    report["unit_price 결측으로 매출 계산 제외"] = int(n_price_na)
    ledger_all = ledger_all.dropna(subset=["unit_price"])

    ledger_all["line_amount"] = (
        ledger_all["quantity"] * ledger_all["unit_price"] * (1 - ledger_all["discount"])
    )

    # --- 6) 총매출 vs 순매출 상태 규칙 ---------------------------------
    # 총매출: 모든 상태 포함 / 순매출: canceled·returned 제외
    ledger_all["is_net"] = ~ledger_all["status"].isin(["canceled", "returned"])

    return ledger_all, report


def monthly_reconciliation(ledger_all):
    """월별 총매출·순매출·취소반품 제외액·유효 주문수 대사표"""
    ledger_all = ledger_all.copy()
    ledger_all["month"] = ledger_all["order_datetime"].dt.to_period("M")

    gross = ledger_all.groupby("month")["line_amount"].sum().rename("총매출")
    net = (
        ledger_all[ledger_all["is_net"]]
        .groupby("month")["line_amount"]
        .sum()
        .rename("순매출")
    )
    excluded = (gross - net).rename("취소·반품 제외액")
    valid_orders = (
        ledger_all[ledger_all["is_net"]]
        .groupby("month")["order_id"]
        .nunique()
        .rename("유효 주문수")
    )

    table = pd.concat([gross, net, excluded, valid_orders], axis=1)
    return table


# ============================================================
# 문제 02. 상반기 성장 기여 분해 브리핑
#   순매출 = 구매고객수 x (주문수/구매고객수) x (순매출/주문수)
#          = 구매고객수 x 고객당 주문수 x AOV
# ============================================================
def growth_decomposition(ledger_all):
    net_lines = ledger_all[ledger_all["is_net"]].copy()
    net_lines["month"] = net_lines["order_datetime"].dt.to_period("M")

    g = net_lines.groupby("month")
    net_revenue = g["line_amount"].sum()
    n_orders = g["order_id"].nunique()
    n_customers = g["customer_id"].nunique()

    customers_per = n_customers.rename("구매고객수")
    orders_per_customer = (n_orders / n_customers).rename("고객당_주문수")
    aov = (net_revenue / n_orders).rename("AOV")

    decomp = pd.concat([customers_per, orders_per_customer, aov, net_revenue.rename("순매출")], axis=1)

    # 검산: 세 요인의 곱이 실제 순매출과 같은지 확인
    decomp["검산_순매출"] = (
        decomp["구매고객수"] * decomp["고객당_주문수"] * decomp["AOV"]
    )
    decomp["검산_오차"] = decomp["순매출"] - decomp["검산_순매출"]

    # 1월 대비 6월 변화율
    first, last = decomp.index.min(), decomp.index.max()
    growth = {
        "구매고객수_변화율": decomp.loc[last, "구매고객수"] / decomp.loc[first, "구매고객수"] - 1,
        "고객당_주문수_변화율": decomp.loc[last, "고객당_주문수"] / decomp.loc[first, "고객당_주문수"] - 1,
        "AOV_변화율": decomp.loc[last, "AOV"] / decomp.loc[first, "AOV"] - 1,
        "순매출_변화율": decomp.loc[last, "순매출"] / decomp.loc[first, "순매출"] - 1,
    }
    return decomp, growth, first, last


if __name__ == "__main__":
    print("=" * 70)
    print("문제 01. 신뢰할 수 있는 매출 원장 만들기")
    print("=" * 70)

    ledger_all, report = build_ledger()

    print("\n[정제 단계별 손실 대사]")
    for k, v in report.items():
        print(f"  - {k}: {v:,}건")

    recon = monthly_reconciliation(ledger_all)
    print("\n[월별 총매출·순매출·취소반품 제외액·유효 주문수 대사표]")
    print(recon.to_string())

    print("\n[매출 정의 문서화]")
    print(
        "총매출은 2024년 상반기 유효 기간 내, 중복 제거·날짜 정제를 거친 주문 라인 전체를\n"
        "quantity x unit_price x (1 - discount)로 합산한 값이다. 순매출은 그중 상태가\n"
        "canceled 또는 returned인 라인을 제외한 값으로, 재무·마케팅 두 팀이 서로 다른 숫자를\n"
        "본 원인은 이 상태 포함 범위의 차이였다. 이 표의 '순매출' 열을 이후 모든 분석의\n"
        "표준 기준으로 확정한다."
    )

    print("\n" + "=" * 70)
    print("문제 02. 상반기 성장 기여 분해 브리핑")
    print("=" * 70)

    decomp, growth, first, last = growth_decomposition(ledger_all)
    print("\n[월별 분해표: 구매고객수 x 고객당 주문수 x AOV = 순매출]")
    print(decomp.to_string())

    max_err = decomp["검산_오차"].abs().max()
    print(f"\n[검산] 항등식 최대 오차: {max_err:,.2f}원 (0에 가까우면 검산 통과)")

    print(f"\n[{first} 대비 {last} 변화율]")
    for k, v in growth.items():
        print(f"  - {k}: {v:+.1%}")

    driver = max(
        ["구매고객수_변화율", "고객당_주문수_변화율", "AOV_변화율"],
        key=lambda k: growth[k],
    )
    print(f"\n[성장 동력 판정] 가장 크게 증가한 요인: {driver}")
    print(
        "-> 이 요인이 가장 크게 오른 요인이라면, 하반기 마케팅 예산은 그에 맞는 전략\n"
        "   (고객수 증가 우세: 획득 마케팅 / 주문수 증가 우세: 재구매 유도 / AOV 증가 우세: 상향판매)\n"
        "   에 우선순위를 둬야 한다는 결론으로 이어진다."
    )
