"""
3장. 코호트와 리텐션 분석
문제 05: 첫 구매 코호트 리텐션 매트릭스
문제 06: 2회차 구매 전환율 개선 과제
문제 07: 상·하반기 신규 코호트 질 비교

실행: python ch03_cohort_retention.py
"""

import pandas as pd
import numpy as np
from shop_analysis.common_order_items import net_ledger

pd.set_option("display.float_format", lambda x: f"{x:,.3f}")


# ============================================================
# 문제 05. 첫 구매 코호트 리텐션 매트릭스
# ============================================================
def retention_matrix(ledger):
    orders = ledger.groupby(["customer_id", "order_id"])["order_datetime"].first().reset_index()
    orders["order_month"] = orders["order_datetime"].dt.to_period("M")

    first_month = orders.groupby("customer_id")["order_month"].min().rename("cohort")
    orders = orders.merge(first_month, on="customer_id")
    orders["period_n"] = (orders["order_month"] - orders["cohort"]).apply(lambda x: x.n)

    pivot = pd.pivot_table(
        orders, index="cohort", columns="period_n", values="customer_id",
        aggfunc="nunique",
    )
    cohort_size = orders.groupby("cohort")["customer_id"].nunique()
    retention = pivot.div(cohort_size, axis=0)

    # 관측 종료(6/30) 밖이라 계산 불가능한 셀은 NaN으로 마스킹
    max_period = {c: (pd.Period("2024-06", "M") - c).n for c in retention.index}
    for c in retention.index:
        for p in retention.columns:
            if p > max_period[c]:
                retention.loc[c, p] = np.nan

    return retention, cohort_size


# ============================================================
# 문제 06. 2회차 구매 전환율 개선 과제
# ============================================================
def second_purchase_conversion(ledger, products_path="data/products.csv"):
    orders = ledger.groupby(["customer_id", "order_id"]).agg(
        order_date=("order_datetime", "first"),
        channel=("channel", "first"),
    ).reset_index().sort_values(["customer_id", "order_date"])

    orders["seq"] = orders.groupby("customer_id").cumcount()
    first_orders = orders[orders["seq"] == 0].copy()
    second_orders = orders[orders["seq"] == 1][["customer_id", "order_date"]].rename(
        columns={"order_date": "second_date"}
    )

    first_orders = first_orders.merge(second_orders, on="customer_id", how="left")
    first_orders["gap_days"] = (first_orders["second_date"] - first_orders["order_date"]).dt.days
    first_orders["converted_30d"] = first_orders["gap_days"] <= 30

    # 6월 첫 구매 고객은 30일 관찰 창을 채울 수 없어 제외
    before = len(first_orders)
    first_orders = first_orders[first_orders["order_date"] < "2024-06-01"].copy()
    print(f"[관찰 창 제외] 6월 첫구매 고객 {before - len(first_orders):,}명 제외 (30일 창 미충족)")

    # 첫 주문의 대표 카테고리·할인 여부
    products = pd.read_csv(products_path)
    products["category"] = products["category"].astype(str).str.strip()
    first_order_ids = ledger.merge(
        first_orders[["customer_id", "order_date"]],
        left_on=["customer_id", "order_datetime"],
        right_on=["customer_id", "order_date"],
    )
    first_order_lines = first_order_ids.merge(products[["product_id", "category"]], on="product_id", how="left")
    cat_by_cust = first_order_lines.groupby("customer_id")["category"].agg(lambda x: x.mode().iat[0])
    discount_by_cust = first_order_lines.groupby("customer_id")["discount"].max().gt(0).rename("first_order_discounted")

    first_orders = first_orders.merge(cat_by_cust.rename("first_category"), on="customer_id", how="left")
    first_orders = first_orders.merge(discount_by_cust, on="customer_id", how="left")

    by_channel = first_orders.groupby("channel")["converted_30d"].mean().rename("전환율").sort_values()
    by_category = first_orders.groupby("first_category")["converted_30d"].mean().rename("전환율").sort_values()
    by_discount = first_orders.groupby("first_order_discounted")["converted_30d"].mean().rename("전환율")

    return by_channel, by_category, by_discount


# ============================================================
# 문제 07. 상·하반기 신규 코호트 질 비교
# ============================================================
def cohort_quality_compare(ledger):
    orders = ledger.groupby(["customer_id", "order_id"]).agg(
        order_date=("order_datetime", "first"),
        order_amount=("line_amount", "sum"),
        channel=("channel", "first"),
    ).reset_index()

    first_purchase = orders.groupby("customer_id")["order_date"].min().rename("first_date")
    orders = orders.merge(first_purchase, on="customer_id")

    cohort_A = orders[(orders["first_date"] >= "2024-01-01") & (orders["first_date"] < "2024-03-01")].copy()
    cohort_B = orders[(orders["first_date"] >= "2024-04-01") & (orders["first_date"] < "2024-06-01")].copy()
    cohort_A["cohort"], cohort_B["cohort"] = "A(1~2월)", "B(4~5월)"

    def window_30d_metrics(df):
        df = df.copy()
        df["days_since_first"] = (df["order_date"] - df["first_date"]).dt.days
        w = df[(df["days_since_first"] >= 0) & (df["days_since_first"] <= 30)]
        n_cust = df["customer_id"].drop_duplicates().shape[0]
        return pd.Series({
            "고객수": n_cust,
            "30일누적순매출": w["order_amount"].sum(),
            "1인당누적매출": w["order_amount"].sum() / n_cust,
            "재구매율": w.groupby("customer_id")["order_id"].nunique().gt(1).mean(),
            "AOV": w["order_amount"].mean(),
        })

    fair = pd.DataFrame({
        "A(1~2월)": window_30d_metrics(cohort_A),
        "B(4~5월)": window_30d_metrics(cohort_B),
    }).T

    def unbounded_metrics(df):
        n_cust = df["customer_id"].drop_duplicates().shape[0]
        return pd.Series({
            "고객수": n_cust,
            "누적순매출(전체관측)": df["order_amount"].sum(),
            "1인당누적매출(전체관측)": df["order_amount"].sum() / n_cust,
        })

    biased = pd.DataFrame({
        "A(1~2월)": unbounded_metrics(cohort_A),
        "B(4~5월)": unbounded_metrics(cohort_B),
    }).T

    # 채널 층화
    all_cohort = pd.concat([cohort_A, cohort_B])
    all_cohort["days_since_first"] = (all_cohort["order_date"] - all_cohort["first_date"]).dt.days
    w_all = all_cohort[(all_cohort["days_since_first"] >= 0) & (all_cohort["days_since_first"] <= 30)]
    strat = w_all.groupby(["channel", "cohort"])["order_amount"].mean().unstack()

    return fair, biased, strat


if __name__ == "__main__":
    ledger = net_ledger()

    print("=" * 70)
    print("문제 05. 첫 구매 코호트 리텐션 매트릭스")
    print("=" * 70)
    retention, cohort_size = retention_matrix(ledger)
    print("\n[코호트 x M+n 리텐션율 매트릭스] (빈칸=관측 미완료)")
    print(retention.round(3).to_string())
    print("\n[M+1 리텐션 추이]")
    print(retention[1].dropna().to_string())
    print(
        "\n해석: 삼각형 우하단의 빈칸(예: 6월 코호트의 M+1)은 리텐션이 0%인 것이 아니라\n"
        "7월 데이터가 아직 없어 관측 자체가 불가능한 셀이다. 0으로 채우면 최근 코호트의\n"
        "리텐션이 실제보다 낮게 왜곡되므로 반드시 결측으로 남겨야 한다."
    )

    print("\n" + "=" * 70)
    print("문제 06. 2회차 구매 전환율 개선 과제")
    print("=" * 70)
    by_channel, by_category, by_discount = second_purchase_conversion(ledger)
    print("\n[채널별 30일 2회차 전환율]")
    print(by_channel.to_string())
    print("\n[첫구매 카테고리별 30일 2회차 전환율]")
    print(by_category.to_string())
    print("\n[첫구매 할인여부별 30일 2회차 전환율]")
    print(by_discount.to_string())
    worst_channel = by_channel.idxmin()
    print(f"\n[개선 과제] 채널 기준 최저 전환율 채널: {worst_channel} "
          f"({by_channel.min():.1%}) -> 해당 채널의 첫구매 온보딩(적립금 안내, 재구매 유도 알림) 강화 필요")

    print("\n" + "=" * 70)
    print("문제 07. 상·하반기 신규 코호트 질 비교")
    print("=" * 70)
    fair, biased, strat = cohort_quality_compare(ledger)
    print("\n[동일 관찰창(30일) 기준 공정 비교]")
    print(fair.to_string())
    print("\n[관찰창 미통제 - 편향된 비교(참고용)]")
    print(biased.to_string())
    print("\n[채널별 층화 - 30일 평균 주문금액]")
    print(strat.to_string())
    diff = fair.loc["B(4~5월)", "1인당누적매출"] / fair.loc["A(1~2월)", "1인당누적매출"] - 1
    verdict = "유지된다" if abs(diff) < 0.1 else ("개선되었다" if diff > 0 else "저하되었다")
    print(f"\n[판정] 동일 관찰창 기준 B코호트의 1인당 매출은 A 대비 {diff:+.1%} -> 신규 고객의 질은 {verdict}")
