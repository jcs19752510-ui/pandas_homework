"""
9장. 비교 분석과 준실험적 사고
문제 24: 앱 vs 웹 성과의 구성 보정 비교
문제 25: 반품 경험이 재구매에 남기는 흔적
문제 26: 할인 첫 구매 고객은 남는가

실행: python ch09_comparative_analysis.py
"""

import pandas as pd
import numpy as np
from shop_analysis.common_order_items import net_ledger, build_ledger, clean_category

pd.set_option("display.float_format", lambda x: f"{x:,.3f}")


# ============================================================
# 문제 24. 앱 vs 웹 성과의 구성 보정 비교
# ============================================================
def channel_composition_compare(net_lines):
    customers = pd.read_csv("data/customers.csv").drop_duplicates(subset="customer_id")
    customers["birth_date"] = pd.to_datetime(customers["birth_date"], errors="coerce")
    customers["age"] = 2024 - customers["birth_date"].dt.year
    bins = [0, 19, 29, 39, 49, 59, 200]
    labels = ["10대이하", "20대", "30대", "40대", "50대", "60대+"]
    customers["age_band"] = pd.cut(customers["age"], bins=bins, labels=labels)

    orders = net_lines.groupby("order_id").agg(
        order_amount=("line_amount", "sum"), channel=("channel", "first"), customer_id=("customer_id", "first")
    ).reset_index()
    orders = orders[orders["channel"].isin(["app", "web"])]

    # 단순 비교
    simple = orders.groupby("channel").agg(AOV=("order_amount", "mean"), 주문수=("order_id", "count"))
    freq = orders.groupby(["channel", "customer_id"]).size().groupby("channel").mean().rename("고객당주문수")
    simple = simple.join(freq)

    merged = orders.merge(customers[["customer_id", "age_band"]], on="customer_id", how="left")
    n_no_age = merged["age_band"].isna().sum()

    valid = merged.dropna(subset=["age_band"])
    age_composition = pd.crosstab(valid["channel"], valid["age_band"], normalize="index")

    stratified = valid.groupby(["age_band", "channel"], observed=True)["order_amount"].agg(["mean", "count"]).unstack()

    # 웹의 연령 구성으로 앱 AOV 재가중(구성 보정치)
    web_age_dist = valid[valid["channel"] == "web"]["age_band"].value_counts(normalize=True)
    app_age_mean = valid[valid["channel"] == "app"].groupby("age_band", observed=True)["order_amount"].mean()
    adjusted_app_aov = (app_age_mean * web_age_dist).sum()

    return simple, n_no_age, age_composition, stratified, adjusted_app_aov


# ============================================================
# 문제 25. 반품 경험이 재구매에 남기는 흔적
# ============================================================
def return_experience_effect(ledger_all, seed=42):
    rng = np.random.default_rng(seed)
    orders = ledger_all.groupby("order_id").agg(
        customer_id=("customer_id", "first"), order_datetime=("order_datetime", "first"), status=("status", "first")
    ).reset_index()

    window = orders[(orders["order_datetime"] >= "2024-01-01") & (orders["order_datetime"] < "2024-05-01")]

    returned_first = (
        window[window["status"] == "returned"].groupby("customer_id")["order_datetime"].min()
    )
    exp_group = returned_first.rename("ref_date").reset_index()
    exp_group["group"] = "경험군"

    no_return_customers = set(window["customer_id"]) - set(returned_first.index)
    delivered = window[(window["status"] == "delivered") & (window["customer_id"].isin(no_return_customers))]
    ctrl_pick = delivered.groupby("customer_id").apply(
        lambda g: g.sample(1, random_state=seed)["order_datetime"].iloc[0], include_groups=False
    )
    ctrl_group = ctrl_pick.rename("ref_date").reset_index()
    ctrl_group["group"] = "비교군"

    both = pd.concat([exp_group, ctrl_group], ignore_index=True)

    # 기준 시점 이전 활동 수준(유효 주문수)으로 층화
    activity = orders.merge(both[["customer_id", "ref_date"]], on="customer_id")
    activity = activity[activity["order_datetime"] < activity["ref_date"]]
    activity_count = activity.groupby("customer_id")["order_id"].nunique().rename("이전주문수")
    both = both.merge(activity_count, on="customer_id", how="left").fillna({"이전주문수": 0})
    both["활동층"] = pd.qcut(both["이전주문수"].rank(method="first"), 3, labels=["낮음", "보통", "높음"])

    # 기준 시점 이후 60일 재구매 여부
    after = orders.merge(both[["customer_id", "ref_date", "group"]], on="customer_id")
    after["gap"] = (after["order_datetime"] - after["ref_date"]).dt.days
    repurchase = after[(after["gap"] > 0) & (after["gap"] <= 60)].groupby("customer_id")["order_id"].nunique().gt(0)
    both = both.merge(repurchase.rename("재구매").reset_index(), on="customer_id", how="left")
    both["재구매"] = both["재구매"].fillna(False)

    before_after = both.groupby("group")["재구매"].mean().rename("층화전_재구매율")
    stratified = both.groupby(["활동층", "group"], observed=True)["재구매"].mean().unstack()

    return before_after, stratified


# ============================================================
# 문제 26. 할인 첫 구매 고객은 남는가
# ============================================================
def discount_first_purchase_effect(ledger_all, products):
    orders = ledger_all.groupby("order_id").agg(
        customer_id=("customer_id", "first"), order_datetime=("order_datetime", "first"),
    ).reset_index()
    first_orders = orders.sort_values("order_datetime").groupby("customer_id").first().reset_index()
    first_orders = first_orders[
        (first_orders["order_datetime"] >= "2024-01-01") & (first_orders["order_datetime"] < "2024-04-01")
    ]

    first_lines = ledger_all.merge(first_orders[["order_id"]], on="order_id")
    disc_intensity = first_lines.groupby("order_id").apply(
        lambda g: (g["discount"] * g["quantity"]).sum() / g["quantity"].sum() if g["quantity"].sum() > 0 else g["discount"].mean(),
        include_groups=False,
    ).rename("할인강도")
    first_orders = first_orders.merge(disc_intensity, on="order_id")

    bins = [-0.001, 0, 0.10, 1.0]
    labels = ["정가0%", "저할인0~10%", "고할인10%+"]
    first_orders["할인구간"] = pd.cut(first_orders["할인강도"], bins=bins, labels=labels)

    all_orders = ledger_all.groupby("order_id").agg(
        customer_id=("customer_id", "first"), order_datetime=("order_datetime", "first")
    ).reset_index()
    merged = first_orders[["customer_id", "order_datetime", "할인구간"]].merge(
        all_orders, on="customer_id", suffixes=("_first", "")
    )
    merged["gap"] = (merged["order_datetime"] - merged["order_datetime_first"]).dt.days
    repurchase_90d = merged[(merged["gap"] > 0) & (merged["gap"] <= 90)]
    repurchase_rate = repurchase_90d.groupby("customer_id")["order_id"].nunique().gt(0)

    first_orders = first_orders.merge(repurchase_rate.rename("재구매90일").reset_index(), on="customer_id", how="left")
    first_orders["재구매90일"] = first_orders["재구매90일"].fillna(False)

    band_summary = first_orders.groupby("할인구간", observed=True).agg(
        고객수=("customer_id", "count"), 재구매율_90일=("재구매90일", "mean")
    )

    # 재구매자의 재구매 주문 내 할인 이용률
    repeat_customers = first_orders[first_orders["재구매90일"]]["customer_id"]
    repeat_orders = ledger_all[
        ledger_all["customer_id"].isin(repeat_customers)
    ].merge(first_orders[["customer_id", "order_datetime"]].rename(columns={"order_datetime": "first_date"}), on="customer_id")
    repeat_orders = repeat_orders[repeat_orders["order_datetime"] > repeat_orders["first_date"]]
    repeat_discount_usage = repeat_orders.groupby(["customer_id", "order_id"])["discount"].max().gt(0)
    repeat_discount_rate = repeat_discount_usage.groupby("customer_id").mean()
    band_map = first_orders.set_index("customer_id")["할인구간"]
    repeat_discount_by_band = repeat_discount_rate.groupby(band_map).mean()

    band_summary["재구매시_할인이용률"] = repeat_discount_by_band

    return band_summary


if __name__ == "__main__":
    net_lines = net_ledger()
    ledger_all = build_ledger()
    products = pd.read_csv("data/products.csv").drop_duplicates()
    products["category"] = clean_category(products["category"])

    print("=" * 70)
    print("문제 24. 앱 vs 웹 성과의 구성 보정 비교")
    print("=" * 70)
    simple, n_no_age, age_comp, stratified, adjusted_app_aov = channel_composition_compare(net_lines)
    print("\n[단순 비교: 채널별 AOV/주문수/고객당주문수]")
    print(simple.to_string())
    print(f"\n[연령 결측으로 층화 제외된 주문] {n_no_age:,}건")
    print("\n[채널별 연령 구성]")
    print(age_comp.round(3).to_string())
    print("\n[층화(연령대x채널) AOV 비교]")
    print(stratified.to_string())
    print(f"\n[구성 보정치] 웹의 연령 구성으로 재가중한 앱 AOV: {adjusted_app_aov:,.0f}원")
    simple_diff = simple.loc["app", "AOV"] - simple.loc["web", "AOV"]
    adj_diff = adjusted_app_aov - simple.loc["web", "AOV"]
    verdict = "유지된다(심슨의 역설 아님)" if (simple_diff > 0) == (adj_diff > 0) else "뒤집힌다(심슨의 역설)"
    print(f"[수정된 결론] 단순비교 차이 {simple_diff:+,.0f}원 vs 보정후 차이 {adj_diff:+,.0f}원 -> 결론이 {verdict}")

    print("\n" + "=" * 70)
    print("문제 25. 반품 경험이 재구매에 남기는 흔적")
    print("=" * 70)
    before_after, stratified25 = return_experience_effect(ledger_all)
    print("\n[층화 전 60일 재구매율]")
    print(before_after.to_string())
    print("\n[활동수준 층화 후 60일 재구매율]")
    print(stratified25.round(3).to_string())
    print(
        "\n[프록시 한계] 반품 발생 실제 시각은 미관측이라 '반품 주문의 주문 시각'을 기준 시점 프록시로\n"
        "사용했다. 비교군에는 동일 기간 무작위 delivered 주문 시각을 대칭으로 부여했다(seed=42 고정).\n"
        "층화 전 차이가 층화 후에도 유지되는지가 이 문제의 핵심 판정 포인트다."
    )

    print("\n" + "=" * 70)
    print("문제 26. 할인 첫 구매 고객은 남는가")
    print("=" * 70)
    band_summary = discount_first_purchase_effect(ledger_all, products)
    print("\n[할인 강도 구간별 90일 재구매율 및 재구매시 할인 이용률]")
    print(band_summary.to_string())
    print(
        "\n[권고] 재구매율 기울기가 완만하고 고할인 구간에서도 재구매율이 크게 떨어지지 않는다면\n"
        "첫 구매 할인을 신규 획득 수단으로 계속 써도 된다. 다만 재구매시 할인 이용률이 구간별로\n"
        "뚜렷이 높다면, 할인으로 들어온 고객은 이후에도 할인에 반응하는 '가격 민감 고객'일 가능성이\n"
        "있으므로 마진 관점에서 첫 구매 할인 상한을 과도하게 높이지 않는 조건부 권고를 제안한다."
    )
