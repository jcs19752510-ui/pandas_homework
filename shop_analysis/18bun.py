"""
문제 18. 코호트별 LTV 곡선과 획득비 상한

실행: python problem18_cohort_ltv_curve.py

전제: 같은 폴더(또는 PYTHONPATH)에 common_order_items.py가 있어야 합니다.
      해당 모듈의 net_order_items()로 표준 순매출 라인(취소·반품 제외)을 가져옵니다.
"""

import pandas as pd
from common_order_items import build_order_items

pd.set_option("display.float_format", lambda x: f"{x:,.2f}")

# 관측 종료일: common_order_items.py의 기간 필터(2024년 상반기)와 동일하게 맞춤
OBS_END = pd.Timestamp("2024-06-30")


def load_products():
    return pd.read_csv("data/products.csv").drop_duplicates()


def build_net_with_margin(order_items, products):
    """순매출 라인에 원가(cost)를 붙여 라인별 마진을 파생한다."""
    net = order_items.merge(products[["product_id", "cost"]], on="product_id", how="left")
    net["margin"] = net["quantity"] * (
        net["unit_price"] * (1 - net["discount"]) - net["cost"]
    )
    return net


def cohort_ltv_curve(net):
    """첫 구매월 코호트별 경과월에 따른 1인당 누적 매출·마진 곡선을 만든다."""
    net = net.copy()
    net["order_month"] = net["order_datetime"].dt.to_period("M")

    first_month = net.groupby("customer_id")["order_month"].min().rename("cohort")
    net = net.merge(first_month, on="customer_id")
    net["period_n"] = (net["order_month"] - net["cohort"]).apply(lambda x: x.n)

    monthly = net.groupby(["cohort", "period_n"]).agg(
        매출=("line_amount", "sum"), 마진=("margin", "sum")
    ).reset_index()

    cohort_size = net.groupby("cohort")["customer_id"].nunique()
    monthly["cohort_size"] = monthly["cohort"].map(cohort_size)
    monthly["1인당매출"] = monthly["매출"] / monthly["cohort_size"]
    monthly["1인당마진"] = monthly["마진"] / monthly["cohort_size"]

    monthly = monthly.sort_values(["cohort", "period_n"])
    monthly["누적1인당매출"] = monthly.groupby("cohort")["1인당매출"].cumsum()
    monthly["누적1인당마진"] = monthly.groupby("cohort")["1인당마진"].cumsum()

    curve = monthly.pivot(index="cohort", columns="period_n", values="누적1인당마진")
    return curve, monthly


def margin_90d(net):
    """첫 구매일 기준 실제 날짜 90일 창의 1인당 평균 마진.
    관측 종료일 안에 90일 창이 들어오는(관측 완료된) 고객만 평균에 포함한다.
    """
    first_date = net.groupby("customer_id")["order_datetime"].min()
    cutoff_ok = (first_date + pd.Timedelta(days=90)) <= OBS_END
    eligible_customers = cutoff_ok[cutoff_ok].index

    net90 = net[net["customer_id"].isin(eligible_customers)].copy()
    net90 = net90.merge(first_date.rename("first_date"), on="customer_id")
    net90 = net90[(net90["order_datetime"] - net90["first_date"]).dt.days <= 90]

    per_customer_margin = net90.groupby("customer_id")["margin"].sum()
    return per_customer_margin.mean(), len(eligible_customers)


if __name__ == "__main__":
    order_items = build_order_items()
    products = load_products()
    net = build_net_with_margin(order_items, products)

    print("=" * 70)
    print("문제 18. 코호트별 LTV 곡선과 획득비 상한")
    print("=" * 70)

    curve, monthly = cohort_ltv_curve(net)
    print("\n[코호트별 누적 1인당 마진 곡선]")
    print(curve.round(0).to_string())

    margin_90d_value, n_eligible = margin_90d(net)
    print(f"\n[90일 시점 1인당 평균 마진] {margin_90d_value:,.0f}원 (관측 완료 코호트 {n_eligible:,}명 대상)")

    print(
        "\n[가정 3개]\n"
        "  1) 마진 회수 기간을 90일로 본다\n"
        "  2) 90일 마진 전액을 획득비 상한으로 허용한다\n"
        "  3) 6개월 관측 데이터이므로 90일 이후의 장기 가치는 고려하지 않은 보수적 추정이다"
    )
    print(f"\n[획득비 상한 제안] 고객 1명당 획득비는 약 {margin_90d_value:,.0f}원을 넘지 않는 것을 권고한다.")