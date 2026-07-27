"""
5장. 상품 포트폴리오와 MD 전략
문제 11: 카테고리 확대·유지·축소 매트릭스
문제 12: 동시구매 분석과 번들 후보
문제 13: 상품 마스터 이상 진단과 지표 영향

실행: python ch05_product_portfolio.py
"""

import pandas as pd
import numpy as np
from shop_analysis.common_order_items import build_ledger, clean_category, clean_price

pd.set_option("display.float_format", lambda x: f"{x:,.3f}")


def load_products_clean():
    products = pd.read_csv("data/products.csv")
    products = products.drop_duplicates()
    products["category"] = clean_category(products["category"])
    return products


# ============================================================
# 문제 11. 카테고리 확대·유지·축소 매트릭스
# ============================================================
def category_matrix(ledger_all, products):
    ledger = ledger_all.merge(products[["product_id", "category", "cost"]], on="product_id", how="left")

    net = ledger[ledger["is_net"]].copy()
    net_rev = net.groupby("category")["line_amount"].sum().rename("순매출")

    net["margin"] = net["line_amount"] - net["quantity"] * net["cost"]
    margin = net.groupby("category")["margin"].sum().rename("마진액")
    margin_rate = (margin / net_rev).rename("마진율")

    all_orders = ledger.groupby(["category", "order_id"])["status"].first().reset_index()
    return_rate = all_orders.groupby("category")["status"].apply(
        lambda s: (s == "returned").sum() / (~s.isin(["canceled"])).sum()
    ).rename("반품률")

    table = pd.concat([net_rev, margin, margin_rate, return_rate], axis=1)

    # 순위 점수화 (매출·마진 클수록 좋음 / 반품률 작을수록 좋음), 가중치 명시
    weights = {"순매출": 0.3, "마진율": 0.4, "반품률": 0.3}
    table["점수_순매출"] = table["순매출"].rank()
    table["점수_마진율"] = table["마진율"].rank()
    table["점수_반품률"] = table["반품률"].rank(ascending=False)
    table["종합점수"] = (
        table["점수_순매출"] * weights["순매출"]
        + table["점수_마진율"] * weights["마진율"]
        + table["점수_반품률"] * weights["반품률"]
    )

    n = len(table)
    table["매출단독_등급"] = pd.qcut(table["순매출"].rank(), 3, labels=["축소", "유지", "확대"])
    table["종합_등급"] = pd.qcut(table["종합점수"], 3, labels=["축소", "유지", "확대"])
    changed = table[table["매출단독_등급"] != table["종합_등급"]]

    return table.sort_values("종합점수", ascending=False), changed, weights


# ============================================================
# 문제 12. 동시구매 분석과 번들 후보
# ============================================================
def co_purchase_analysis(ledger_all, products):
    net = ledger_all[ledger_all["is_net"]].merge(
        products[["product_id", "category"]], on="product_id", how="left"
    )

    # 주문x카테고리 고유쌍으로 축약 후 self-merge (메모리 폭발 방지)
    order_cat = net[["order_id", "category"]].drop_duplicates()
    pair = order_cat.merge(order_cat, on="order_id", suffixes=("_A", "_B"))
    pair = pair[pair["category_A"] < pair["category_B"]]

    cooccur = pair.groupby(["category_A", "category_B"]).size().rename("동시구매주문수").reset_index()

    cat_orders = order_cat.groupby("category")["order_id"].nunique()
    total_orders = order_cat["order_id"].nunique()

    cooccur["지지도"] = cooccur["동시구매주문수"] / total_orders
    cooccur["기대동시구매"] = cooccur.apply(
        lambda r: cat_orders[r["category_A"]] * cat_orders[r["category_B"]] / total_orders, axis=1
    )
    cooccur["lift"] = cooccur["동시구매주문수"] / cooccur["기대동시구매"]
    cooccur = cooccur.sort_values("lift", ascending=False)

    # 상품 수준 번들 후보 (상위 lift 카테고리 쌍 내에서 개별 상품 동시구매 상위 추출)
    order_prod = net[["order_id", "product_id"]].drop_duplicates()
    prod_pair = order_prod.merge(order_prod, on="order_id", suffixes=("_A", "_B"))
    prod_pair = prod_pair[prod_pair["product_id_A"] < prod_pair["product_id_B"]]
    prod_cooccur = prod_pair.groupby(["product_id_A", "product_id_B"]).size().rename("동시구매수").reset_index()
    top_bundles = prod_cooccur.sort_values("동시구매수", ascending=False).head(3)

    names = products.set_index("product_id")["product_name"]
    top_bundles = top_bundles.copy()
    top_bundles["상품A"] = top_bundles["product_id_A"].map(names)
    top_bundles["상품B"] = top_bundles["product_id_B"].map(names)

    return cooccur.head(10), top_bundles


# ============================================================
# 문제 13. 상품 마스터 이상 진단과 지표 영향
# ============================================================
def price_anomaly_diagnosis(products, ledger_all):
    raw_price = products["price"]
    numeric_price = clean_price(raw_price)

    is_format_error = numeric_price.isna() & raw_price.notna()
    is_missing = raw_price.isna()
    is_zero = numeric_price == 0
    is_negative = numeric_price < 0
    is_valid_positive = numeric_price > 0

    anomaly_type = pd.Series("정상", index=products.index)
    anomaly_type[is_format_error] = "형식오염"
    anomaly_type[is_missing] = "결측"
    anomaly_type[is_zero] = "0원"
    anomaly_type[is_negative] = "음수"

    anomaly_counts = anomaly_type.value_counts()

    # 실제 거래 단가와 대조 (진짜 역마진 vs 마스터 오류 판정)
    actual_price = ledger_all.groupby("product_id")["unit_price"].median().rename("실거래_중앙단가")
    diag = products[["product_id", "product_name", "cost"]].merge(
        actual_price, on="product_id", how="left"
    )
    diag["마스터_price"] = numeric_price.values
    diag["anomaly_type"] = anomaly_type.values
    diag["진짜역마진_여부"] = diag["실거래_중앙단가"] < diag["cost"]

    # 두 기준 마진 비교: 마스터 price 기준 vs 실거래 단가 기준
    ledger_with_cost = ledger_all.merge(products[["product_id", "cost"]], on="product_id", how="left")
    margin_actual = (
        (ledger_with_cost["quantity"] * (ledger_with_cost["unit_price"] - ledger_with_cost["cost"]))
        .sum()
    )
    valid_master = diag.dropna(subset=["마스터_price"])
    margin_master_based = (
        valid_master["마스터_price"] - valid_master["cost"]
    ).sum()  # 참고용 단순 합(수량 미가중, 마스터 기준 대략치)

    priority = diag[diag["anomaly_type"] != "정상"].sort_values("cost", ascending=False).head(10)

    return anomaly_counts, diag, margin_actual, margin_master_based, priority


if __name__ == "__main__":
    ledger_all = build_ledger()
    products = load_products_clean()

    print("=" * 70)
    print("문제 11. 카테고리 확대·유지·축소 매트릭스")
    print("=" * 70)
    table, changed, weights = category_matrix(ledger_all, products)
    print(f"\n[가중치] {weights}")
    print("\n[카테고리 3지표 종합표]")
    print(table[["순매출", "마진율", "반품률", "종합점수", "종합_등급"]].to_string())
    print("\n[매출 단독 판단과 결론이 달라진 카테고리]")
    if len(changed):
        print(changed[["매출단독_등급", "종합_등급"]].to_string())
    else:
        print("  없음 (매출 순위와 종합 점수 순위가 일치)")

    print("\n" + "=" * 70)
    print("문제 12. 동시구매 분석과 번들 후보")
    print("=" * 70)
    cooccur, top_bundles = co_purchase_analysis(ledger_all, products)
    print("\n[카테고리 쌍 동시구매 상위 10 (lift 기준)]")
    print(cooccur.to_string(index=False))
    print("\n[상품 번들 후보 3건]")
    print(top_bundles[["상품A", "상품B", "동시구매수"]].to_string(index=False))
    print(
        "\n[예상 효과 논리] lift가 1보다 크게 높은 카테고리 조합은 우연한 동시구매가 아니라\n"
        "실제 연관 소비 패턴을 반영한다. 상위 번들 상품을 묶어 할인하면 객단가 상승을 기대할 수 있다."
    )

    print("\n" + "=" * 70)
    print("문제 13. 상품 마스터 이상 진단과 지표 영향")
    print("=" * 70)
    anomaly_counts, diag, margin_actual, margin_master_based, priority = price_anomaly_diagnosis(products, ledger_all)
    print("\n[이상 유형별 건수]")
    print(anomaly_counts.to_string())
    print(f"\n[마진 비교] 실거래 단가 기준 총마진: {margin_actual:,.0f}원")
    print(f"[마진 비교] 마스터 price 기준(단순, 참고용): {margin_master_based:,.0f}원")
    print("\n[정비 우선순위 상위 10 (원가 기준 영향 큰 순)]")
    print(priority[["product_id", "product_name", "anomaly_type", "cost", "실거래_중앙단가"]].to_string(index=False))
