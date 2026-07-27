"""
문제 17. 6개월 실현 LTV와 가치 집중도

실행: python problem17_ltv_distribution.py
"""

import pandas as pd
from common_order_items import build_order_items

pd.set_option("display.float_format", lambda x: f"{x:,.2f}")


def load_products():
    return pd.read_csv("data/products.csv").drop_duplicates()


def load_customers():
    return pd.read_csv("data/customers.csv")


def ltv_distribution(order_items, products):
    # 원가(cost)만 상품 마스터에서 결합, 판매 단가는 order_items.unit_price 그대로 사용
    items_pro = order_items.merge(products[["product_id", "cost"]], on="product_id", how="left")

    # margin = 주문수량 * (판매단가 * (1 - 할인율) - 원가)
    # 1 * 1,000 * (1 - 0.1) - 800 = 100
    items_pro["margin"] = items_pro["quantity"] * (
        items_pro["unit_price"] * (1 - items_pro["discount"]) - items_pro["cost"]
    )

    cust_ltv = items_pro.groupby("customer_id").agg(
        매출LTV=("line_amount", "sum"), # line_amount(실구매가)
        마진LTV=("margin", "sum"),
    )

    def dist_stats(s):
        return pd.Series({
            "평균": s.mean(),
            "중앙값": s.median(),
            "P90": s.quantile(0.9),
            "P99": s.quantile(0.99),
        })

    dist_summary = pd.DataFrame({
        "매출LTV": dist_stats(cust_ltv["매출LTV"]),
        "마진LTV": dist_stats(cust_ltv["마진LTV"]),
    })

    top10_share = (
        cust_ltv["매출LTV"].sort_values(ascending=False)
        .head(int(len(cust_ltv) * 0.1)).sum()
        / cust_ltv["매출LTV"].sum()
    )

    return cust_ltv, dist_summary, top10_share







if __name__ == "__main__":
    order_items = build_order_items()      # 정상건 orders / orders_items (취소/반품건 제외)
    products  = load_products()            # 상품 중복건 제거
    customers = load_customers()           # 고객정보 조회

    print("=" * 70)
    print("문제 17. 6개월 실현 LTV와 가치 집중도")
    print("=" * 70)

    cust_ltv, dist_summary, top10_share = ltv_distribution(order_items, products)

    print("\n[LTV 분포 요약: 평균/중앙값/P90/P99]")
    print(dist_summary.to_string())

    print(f"\n[집중도] 상위 10% 고객이 전체 매출LTV의 {top10_share:.1%}를 차지")

    gap = dist_summary.loc["평균", "매출LTV"] / dist_summary.loc["중앙값", "매출LTV"] - 1
    print(f"\n[평균-중앙값 간극] 평균이 중앙값보다 {gap:+.1%} 높음")

    n_ltv_customers = cust_ltv.index.nunique()
    n_master_customers = customers["customer_id"].nunique()
    print(
        f"\n[고객 모수 확인] LTV 계산에 등장한 고객 수 {n_ltv_customers:,}명 "
        f"vs 고객 마스터 등록 고객 수 {n_master_customers:,}명 "
        f"(마스터에 없는 고객 {n_ltv_customers - n_master_customers:,}건)"
    )

    print(
        f"\n[시사점] 평균이 중앙값보다 {gap:+.1%} 높고 상위 10% 고객이 매출LTV의 "
        f"{top10_share:.1%}를 차지한다는 것은 소수 고액 고객이 평균을 끌어올렸다는 뜻이다.\n"
        "따라서 '평균 LTV x N배수'식 획득비 상한은 상위 소수 고객에 맞춰진 과대 기준일 위험이 크며,\n"
        "중앙값이나 코호트별 곡선(문제 18) 같은 왜곡에 강한 지표로 다시 세워야 한다."
    )