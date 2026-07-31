"""
6장. 가격과 할인 효과 분석
문제 14: 할인율 구간별 손익 분석
문제 15: 할인 의존 상품 검출
문제 16: 할인 마진 잠식 시뮬레이션

실행: python ch06_discount_effect.py
"""

import pandas as pd
import numpy as np
from shop_analysis.common_order_items import build_ledger, clean_category

pd.set_option("display.float_format", lambda x: f"{x:,.3f}")


def load_products_clean():
    products = pd.read_csv("data/products.csv").drop_duplicates()
    products["category"] = clean_category(products["category"])
    return products


# ============================================================
# 문제 14. 할인율 구간별 손익 분석
# ============================================================
def discount_band_analysis(ledger_all, products):
    net = ledger_all[ledger_all["is_net"]].merge(
        products[["product_id", "cost"]], on="product_id", how="left"
    )
    net["margin"] = net["quantity"] * (net["unit_price"] * (1 - net["discount"]) - net["cost"])
    net["sale_price"] = net["unit_price"] * (1 - net["discount"])

    bins = [-0.001, 0, 0.10, 0.20, 1.0]
    labels = ["정가(0%)", "0~10%", "10~20%", "20%초과"]
    net["discount_band"] = pd.cut(net["discount"], bins=bins, labels=labels)

    table = net.groupby("discount_band", observed=True).agg(
        판매수량=("quantity", "sum"),
        순매출=("line_amount", "sum"),
        마진액=("margin", "sum"),
        평균판매단가=("sale_price", "mean"),
    )
    table["마진율"] = table["마진액"] / table["순매출"]
    return table


# ============================================================
# 문제 15. 할인 의존 상품 검출
# ============================================================
def discount_dependency(ledger_all, products, min_qty=50):
    lines = ledger_all[ledger_all["is_net"]].copy()
    lines["is_discounted"] = lines["discount"] > 0

    agg = lines.groupby("product_id").apply(
        lambda g: pd.Series({
            "총판매수량": g["quantity"].sum(),
            "할인판매수량": g.loc[g["is_discounted"], "quantity"].sum(),
            "정가판매수량": g.loc[~g["is_discounted"], "quantity"].sum(),
            "수량가중평균할인율": (g["discount"] * g["quantity"]).sum() / g["quantity"].sum(),
        }),
        include_groups=False,
    )
    agg["할인판매비중"] = agg["할인판매수량"] / agg["총판매수량"]

    eligible = agg[agg["총판매수량"] >= min_qty]
    dependent = eligible[eligible["할인판매비중"] >= 0.80].copy()

    MIN_FULL_PRICE_QTY = 5
    dependent["대조가능"] = dependent["정가판매수량"] >= MIN_FULL_PRICE_QTY
    dependent = dependent.merge(products[["product_id", "product_name", "category"]], on="product_id", how="left")

    return dependent.sort_values("할인판매비중", ascending=False)


# ============================================================
# 문제 16. 할인 마진 잠식 시뮬레이션
# ============================================================
def discount_erosion_simulation(ledger_all, products):
    net = ledger_all[ledger_all["is_net"]].merge(
        products[["product_id", "cost"]], on="product_id", how="left"
    )
    actual_revenue = net["line_amount"].sum()
    actual_margin = (net["quantity"] * (net["unit_price"] * (1 - net["discount"]) - net["cost"])).sum()

    # 할인이 없었다면(동일 수량 가정): 정가 매출
    no_discount_revenue = (net["quantity"] * net["unit_price"]).sum()
    no_discount_margin = (net["quantity"] * (net["unit_price"] - net["cost"])).sum()
    forgone_amount = no_discount_revenue - actual_revenue

    scenarios = {}
    for retention_rate, label in [(1.0, "동일수량(낙관 상한)"), (0.8, "20% 감소"), (0.5, "50% 감소")]:
        discounted_mask = net["discount"] > 0
        qty_sim = net["quantity"].where(~discounted_mask, net["quantity"] * retention_rate)
        rev_sim = (qty_sim * net["unit_price"].where(~discounted_mask, net["unit_price"] * (1 - net["discount"]))).sum()
        margin_sim = (
            qty_sim * (net["unit_price"].where(~discounted_mask, net["unit_price"] * (1 - net["discount"])) - net["cost"])
        ).sum()
        scenarios[label] = {"순매출": rev_sim, "마진": margin_sim}

    scenario_table = pd.DataFrame(scenarios).T
    scenario_table.loc["실제(할인적용)"] = {"순매출": actual_revenue, "마진": actual_margin}
    scenario_table.loc["할인없음(정가,동일수량)"] = {"순매출": no_discount_revenue, "마진": no_discount_margin}

    # 손익분기 감소율 역산: 할인 라인 수량이 x비율로 유지될 때 마진이 할인없음과 같아지는 x
    disc_lines = net[net["discount"] > 0]
    a = (disc_lines["quantity"] * (disc_lines["unit_price"] * (1 - disc_lines["discount"]) - disc_lines["cost"])).sum()
    b = (disc_lines["quantity"] * (disc_lines["unit_price"] - disc_lines["cost"])).sum()  # 정가 판매시 마진(동일수량)
    non_disc_margin = (
        net.loc[net["discount"] == 0, "quantity"]
        * (net.loc[net["discount"] == 0, "unit_price"] - net.loc[net["discount"] == 0, "cost"])
    ).sum()
    # breakeven: non_disc_margin + a == non_disc_margin_no_discount_scenario + b*x  -> 단순화된 근사 역산
    breakeven_ratio = a / b if b != 0 else np.nan

    return forgone_amount, scenario_table, breakeven_ratio


if __name__ == "__main__":
    ledger_all = build_ledger()
    products = load_products_clean()

    print("=" * 70)
    print("문제 14. 할인율 구간별 손익 분석")
    print("=" * 70)
    band_table = discount_band_analysis(ledger_all, products)
    print("\n[할인구간 x (수량,순매출,마진액,마진율,평균판매단가)]")
    print(band_table.to_string())
    worst = band_table["마진율"].idxmin()
    print(f"\n[마진 관점 최비효율 구간] {worst} (마진율 {band_table.loc[worst, '마진율']:.1%})")
    print(
        "[관찰 vs 인과] 할인 구간일수록 판매수량이 많이 관찰될 수 있으나, 이는 상관일 뿐\n"
        "할인이 수량을 '유발'했다는 인과적 증거는 아니다(원래 잘 팔리는 상품에 할인을 몰아줬을 수도 있음).\n"
        "[정책 제안] 20% 초과 구간처럼 마진율이 낮은 구간은 상시 할인 상한을 20%로 제한하고,\n"
        "예외 프로모션만 승인제로 운영할 것을 제안한다."
    )

    print("\n" + "=" * 70)
    print("문제 15. 할인 의존 상품 검출")
    print("=" * 70)
    dependent = discount_dependency(ledger_all, products)
    print(f"\n[할인 의존 상품] {len(dependent)}개 (할인판매비중 80%+, 최소판매량 50 이상)")
    print(dependent[["product_id", "product_name", "category", "할인판매비중", "수량가중평균할인율", "총판매수량", "대조가능"]].head(15).to_string(index=False))
    n_no_contrast = (~dependent["대조가능"]).sum()
    print(f"\n[정가 판매 대조 불가 상품] {n_no_contrast}개 (정가 판매 표본 5건 미만)")
    print(
        "[가격 정책 논점] 할인 의존 상품은 이미 고객이 '정가'를 기준가로 인식하지 않을 가능성이 높다.\n"
        "정가 인하로 할인율 체감을 낮추거나, 할인을 유지하되 상시할인이 아닌 시즌 한정으로 전환하는\n"
        "두 가지 옵션을 검토해야 하며, 대조 불가 상품은 정가 판매 실험을 별도로 설계해야 한다."
    )

    print("\n" + "=" * 70)
    print("문제 16. 할인 마진 잠식 시뮬레이션")
    print("=" * 70)
    forgone_amount, scenario_table, breakeven_ratio = discount_erosion_simulation(ledger_all, products)
    print(f"\n[할인으로 포기한 금액(동일수량 가정)] {forgone_amount:,.0f}원")
    print("\n[시나리오별 손익 대조표]")
    print(scenario_table.to_string())
    print(f"\n[손익분기 감소율 역산] 할인 판매의 약 {breakeven_ratio:.1%}가 '할인 덕분에' 발생했어야\n"
          f"손익이 할인 없음 시나리오와 같아진다. (단순화된 근사치)")
    print(
        "[결론] 동일수량 가정은 가장 낙관적인 상한이며 실제로는 할인 종료 시 수량이 어느 정도\n"
        "감소하는 게 일반적이다. 위 손익분기율보다 할인 유인 효과가 크다는 근거가 없다면,\n"
        "현재 할인 정책은 매출은 지키되 마진을 갉아먹고 있을 가능성이 높다."
    )
