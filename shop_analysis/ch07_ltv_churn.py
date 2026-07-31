import pandas as pd
import numpy as np
from shop_analysis.common_order_items import build_ledger, net_ledger

pd.set_option("display.float_format", lambda x: f"{x:,.2f}")

REF_DATE = pd.Timestamp("2024-07-01")


def load_products():
    return pd.read_csv("data/products.csv").drop_duplicates()


def ltv_distribution(ledger_all, products):
    net = ledger_all[ledger_all["is_net"]].merge(products[["product_id", "cost"]], on="product_id", how="left")
    net["margin"] = net["quantity"] * (net["unit_price"] * (1 - net["discount"]) - net["cost"])

    cust_ltv = net.groupby("customer_id").agg(매출LTV=("line_amount", "sum"), 마진LTV=("margin", "sum"))

    def dist_stats(s):
        return pd.Series({
            "평균": s.mean(), "중앙값": s.median(),
            "P90": s.quantile(0.9), "P99": s.quantile(0.99),
        })

    dist_summary = pd.DataFrame({"매출LTV": dist_stats(cust_ltv["매출LTV"]), "마진LTV": dist_stats(cust_ltv["마진LTV"])})

    top10_share = cust_ltv["매출LTV"].sort_values(ascending=False).head(int(len(cust_ltv) * 0.1)).sum() / cust_ltv["매출LTV"].sum()

    return cust_ltv, dist_summary, top10_share


def cohort_ltv_curve(ledger_all, products):
    net = ledger_all[ledger_all["is_net"]].merge(products[["product_id", "cost"]], on="product_id", how="left")
    net["margin"] = net["quantity"] * (net["unit_price"] * (1 - net["discount"]) - net["cost"])
    net["order_month"] = net["order_datetime"].dt.to_period("M")

    first_month = net.groupby("customer_id")["order_month"].min().rename("cohort")
    net = net.merge(first_month, on="customer_id")
    net["period_n"] = (net["order_month"] - net["cohort"]).apply(lambda x: x.n)

    monthly = net.groupby(["cohort", "period_n"]).agg(매출=("line_amount", "sum"), 마진=("margin", "sum")).reset_index()
    cohort_size = net.groupby("cohort")["customer_id"].nunique()
    monthly["cohort_size"] = monthly["cohort"].map(cohort_size)
    monthly["1인당매출"] = monthly["매출"] / monthly["cohort_size"]
    monthly["1인당마진"] = monthly["마진"] / monthly["cohort_size"]

    monthly = monthly.sort_values(["cohort", "period_n"])
    monthly["누적1인당매출"] = monthly.groupby("cohort")["1인당매출"].cumsum()
    monthly["누적1인당마진"] = monthly.groupby("cohort")["1인당마진"].cumsum()

    curve = monthly.pivot(index="cohort", columns="period_n", values="누적1인당마진")

    first_date = net.groupby("customer_id")["order_datetime"].min()
    cutoff_ok = (first_date + pd.Timedelta(days=90)) <= pd.Timestamp("2024-06-30")
    eligible_customers = cutoff_ok[cutoff_ok].index

    net90 = net[net["customer_id"].isin(eligible_customers)].copy()
    net90 = net90.merge(first_date.rename("first_date"), on="customer_id")
    net90 = net90[(net90["order_datetime"] - net90["first_date"]).dt.days <= 90]
    margin_90d_per_customer = net90.groupby("customer_id")["margin"].sum().mean()

    return curve, margin_90d_per_customer, len(eligible_customers)


def churn_threshold_sensitivity(net_lines):
    orders = net_lines.groupby(["customer_id", "order_id"])["order_datetime"].first().reset_index()
    orders = orders.sort_values(["customer_id", "order_datetime"])
    orders["prev_date"] = orders.groupby("customer_id")["order_datetime"].shift(1)
    orders["gap_days"] = (orders["order_datetime"] - orders["prev_date"]).dt.days

    gap_dist = orders["gap_days"].dropna()
    p75, p90 = gap_dist.quantile([0.75, 0.90])

    last_purchase = orders.groupby("customer_id")["order_datetime"].max()
    recency = (REF_DATE - last_purchase).dt.days

    candidates = [60, 90, 120, int(p75), int(p90)]
    sensitivity = pd.Series({f"{c}일": (recency > c).mean() for c in sorted(set(candidates))})

    n_single = orders.groupby("customer_id").size().eq(1).sum()

    return gap_dist, p75, p90, sensitivity, n_single, recency


def risk_scorecard(net_lines, recency, recency_cutoff=45):
    orders = net_lines.groupby(["customer_id", "order_id"]).agg(
        order_amount=("line_amount", "sum")
    ).reset_index()
    freq = orders.groupby("customer_id")["order_id"].nunique().rename("frequency")
    value = orders.groupby("customer_id")["order_amount"].sum().rename("value")

    freq_q = freq.quantile([0.33, 0.66])

    logs = pd.read_csv(
        "data/web_logs.csv", usecols=["customer_id", "event_time"], dtype={"customer_id": "float64"},
        parse_dates=["event_time"],
    )
    june_visitors = set(logs.loc[logs["event_time"] >= "2024-06-01", "customer_id"].dropna().unique())

    score_df = pd.concat([recency.rename("recency"), freq, value], axis=1).dropna()

    def r_score(r):
        return 2 if r <= recency_cutoff * 0.5 else (1 if r <= recency_cutoff else 0)

    def f_score(f):
        return 2 if f >= freq_q[0.66] else (1 if f >= freq_q[0.33] else 0)

    score_df["R점수"] = score_df["recency"].apply(r_score)
    score_df["F점수"] = score_df["frequency"].apply(f_score)
    score_df["방문점수"] = score_df.index.to_series().apply(lambda c: 2 if c in june_visitors else 0)
    score_df["위험점수"] = score_df["R점수"] + score_df["F점수"] + score_df["방문점수"]
    score_df["이탈위험점수"] = 6 - score_df["위험점수"]

    value_median = score_df["value"].median()
    score_df["고가치"] = score_df["value"] >= value_median
    score_df["고위험"] = score_df["이탈위험점수"] >= 4

    priority = score_df[score_df["고위험"] & score_df["고가치"]].sort_values("value", ascending=False)
    annualized_loss = priority["value"].sum() * 2

    return score_df, priority, annualized_loss


if __name__ == "__main__":
    ledger_all = build_ledger()
    net_lines = ledger_all[ledger_all["is_net"]].copy()
    products = load_products()

    print("=" * 70)
    print("문제 17. 6개월 실현 LTV와 가치 집중도")
    print("=" * 70)
    cust_ltv, dist_summary, top10_share = ltv_distribution(ledger_all, products)
    print("\n[LTV 분포 요약: 평균/중앙값/P90/P99]")
    print(dist_summary.to_string())
    print(f"\n[집중도] 상위 10% 고객이 전체 매출LTV의 {top10_share:.1%}를 차지")
    gap = dist_summary.loc["평균", "매출LTV"] / dist_summary.loc["중앙값", "매출LTV"] - 1
    print(f"[시사점] 평균이 중앙값보다 {gap:+.1%} 높다는 것은 소수 고액 고객이 평균을 끌어올렸다는 뜻이다.\n"
          f"따라서 '평균 LTV x 5만원'식 획득비 기준은 상위 소수 고객에 맞춰진 과대 기준일 위험이 크다.")

    print("\n" + "=" * 70)
    print("문제 18. 코호트별 LTV 곡선과 획득비 상한")
    print("=" * 70)
    curve, margin_90d, n_eligible = cohort_ltv_curve(ledger_all, products)
    print("\n[코호트별 누적 1인당 마진 곡선]")
    print(curve.round(0).to_string())
    print(f"\n[90일 시점 1인당 평균 마진] {margin_90d:,.0f}원 (관측 완료 코호트 {n_eligible:,}명 대상)")
    print(
        "[가정 3개] 1) 마진 회수 기간을 90일로 본다  2) 90일 마진 전액을 획득비 상한으로 허용한다\n"
        "3) 6개월 관측 데이터이므로 90일 이후의 장기 가치는 고려하지 않은 보수적 추정이다.\n"
        f"[획득비 상한 제안] 고객 1명당 획득비는 약 {margin_90d:,.0f}원을 넘지 않는 것을 권고한다."
    )

    print("\n" + "=" * 70)
    print("문제 19. 이탈 기준선 탐색과 민감도")
    print("=" * 70)
    gap_dist, p75, p90, sensitivity, n_single, recency = churn_threshold_sensitivity(net_lines)
    print(f"\n[재구매 간격 분포] P75={p75:.0f}일 / P90={p90:.0f}일 (주문 1회 고객 {n_single:,}명은 분포에서 제외)")
    print("\n[기준선 후보별 이탈률 민감도]")
    print(sensitivity.apply(lambda x: f"{x:.1%}").to_string())
    print(
        "\n[권고] 재구매 간격 P75가 60일 안팎이라면, 통념인 '90일 무구매=이탈'은 이 쇼핑몰 고객군에는\n"
        "느슨한 기준이다. P75를 반올림한 값을 1차 경보선으로 쓰고 P90을 확정 이탈선으로 이원화할 것을 권고한다.\n"
        "[검열 주의] 관측 종료일(6/30) 근처에 마지막 구매가 있는 고객은 '아직 이탈이 아닐' 수도 있는데\n"
        "관측이 끝나 확정할 수 없는 상태(검열, censoring)이므로 이탈률 해석 시 이 점을 명시해야 한다."
    )

    print("\n" + "=" * 70)
    print("문제 20. 이탈 위험 스코어카드")
    print("=" * 70)
    score_df, priority, annualized_loss = risk_scorecard(net_lines, recency)
    print("\n[스코어 규칙] recency(2점)+frequency(2점)+최근30일 방문여부(2점) = 0~6점, 점수가 낮을수록 고위험")
    print("\n[위험 x 가치 분포]")
    print(pd.crosstab(score_df["고위험"], score_df["고가치"]).to_string())
    print(f"\n[우선 관리군(고위험 x 고가치)] {len(priority):,}명")
    print(priority[["recency", "frequency", "value"]].head(10).to_string())
    print(f"\n[예상 손실 추정] 우선 관리군 이탈 시 연 환산(6개월x2 가정) 손실 약 {annualized_loss:,.0f}원")