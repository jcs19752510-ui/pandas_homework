"""
2장. 고객 세분화 - RFM과 타깃 목록
문제 03: RFM 스코어 산출과 세그먼트 정의
문제 04: 멤버십 등급제 설계 시뮬레이션

실행: python ch02_rfm_segmentation.py
"""

import pandas as pd
import numpy as np
from shop_analysis.common_order_items import net_ledger

pd.set_option("display.float_format", lambda x: f"{x:,.1f}")

REF_DATE = pd.Timestamp("2024-07-01")  # RFM 기준일 고정


# ============================================================
# 문제 03. RFM 스코어 산출과 세그먼트 정의
# ============================================================
def compute_rfm(ledger, customers_path="data/customers.csv"):
    customers = pd.read_csv(customers_path)

    # 고객 단위 R/F/M
    order_level = (
        ledger.groupby(["customer_id", "order_id"])
        .agg(order_amount=("line_amount", "sum"), order_date=("order_datetime", "first"))
        .reset_index()
    )

    cust_rfm = order_level.groupby("customer_id").agg(
        last_purchase=("order_date", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum"),
    )
    cust_rfm["recency"] = (REF_DATE - cust_rfm["last_purchase"]).dt.days

    # 구매 이력이 없는 고객: customers 전체와 left join, 미구매 고객은 별도 세그먼트로
    rfm = customers[["customer_id"]].drop_duplicates().merge(
        cust_rfm, on="customer_id", how="left"
    )
    has_purchase = rfm["frequency"].notna()
    print(f"[미구매 고객] {(~has_purchase).sum():,}명은 RFM 점수 대상에서 제외하고 '미구매' 세그먼트로 별도 처리")

    scored = rfm[has_purchase].copy()

    # 동점이 많은 축(recency·frequency)은 rank(method="first")로 순위를 매긴 뒤 분위 분할
    # R은 값이 작을수록(최근일수록) 좋은 점수여야 하므로 오름차순 순위에 라벨 방향을 반대로 둔다
    scored["R_score"] = pd.qcut(
        scored["recency"].rank(method="first"), 5, labels=[5, 4, 3, 2, 1]
    ).astype(int)
    scored["F_score"] = pd.qcut(
        scored["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]
    ).astype(int)
    scored["M_score"] = pd.qcut(scored["monetary"], 5, labels=[1, 2, 3, 4, 5]).astype(int)

    def segment(row):
        r, f, m = row["R_score"], row["F_score"], row["M_score"]
        if r >= 4 and f >= 4 and m >= 4:
            return "챔피언"
        if f >= 4 and m >= 3:
            return "충성"
        if r >= 4 and f <= 2:
            return "잠재 충성"
        if r <= 2 and f >= 3:
            return "이탈 위험"
        if r <= 2 and f <= 2:
            return "휴면"
        return "일반"

    scored["segment"] = scored.apply(segment, axis=1)

    # 미구매 고객 별도 세그먼트로 합치기
    no_purchase = rfm[~has_purchase].copy()
    no_purchase["segment"] = "미구매"
    rfm_full = pd.concat([scored, no_purchase], ignore_index=True, sort=False)

    return rfm_full


def segment_summary(rfm_full):
    total_monetary = rfm_full["monetary"].fillna(0).sum()
    summary = rfm_full.groupby("segment").agg(
        인원=("customer_id", "count"),
        매출합계=("monetary", "sum"),
    )
    summary["매출비중"] = summary["매출합계"].fillna(0) / total_monetary
    return summary.sort_values("매출합계", ascending=False)


# ============================================================
# 문제 04. 멤버십 등급제 설계 시뮬레이션
# ============================================================
def membership_simulation(rfm_full):
    valid = rfm_full[rfm_full["monetary"].notna()].copy()

    # 안 A: qcut으로 상위 5% / 다음 15% / 나머지 80%
    valid["grade_A"] = pd.qcut(
        valid["monetary"], q=[0, 0.80, 0.95, 1.0], labels=["실버", "골드", "VIP"]
    )

    # 안 B: cut으로 고정 금액 컷 (30만원, 100만원)
    valid["grade_B"] = pd.cut(
        valid["monetary"],
        bins=[-np.inf, 300_000, 1_000_000, np.inf],
        labels=["실버", "골드", "VIP"],
    )

    def grade_table(col):
        t = valid.groupby(col, observed=True).agg(
            인원=("customer_id", "count"), 매출합계=("monetary", "sum")
        )
        t["매출커버리지"] = t["매출합계"] / valid["monetary"].sum()
        return t

    table_A = grade_table("grade_A")
    table_B = grade_table("grade_B")

    # 적립률 가정 (등급별)
    accrual_rate = {"실버": 0.01, "골드": 0.03, "VIP": 0.05}
    budget_assumption = pd.Series(accrual_rate, name="적립률_가정")

    def budget(table):
        rates = np.array([accrual_rate[g] for g in table.index], dtype=float)
        return float((table["매출합계"].to_numpy() * rates).sum())

    budget_A = budget(table_A)
    budget_B = budget(table_B)

    return table_A, table_B, budget_assumption, budget_A, budget_B


if __name__ == "__main__":
    ledger = net_ledger()

    print("=" * 70)
    print("문제 03. RFM 스코어 산출과 세그먼트 정의 (기준일: 2024-07-01)")
    print("=" * 70)

    rfm_full = compute_rfm(ledger)
    print("\n[고객별 RFM 테이블 샘플]")
    print(rfm_full[["customer_id", "recency", "frequency", "monetary", "segment"]].head(8).to_string(index=False))

    print("\n[세그먼트 정의 규칙]")
    print(
        "  챔피언   : R>=4, F>=4, M>=4 (최근에도 자주, 많이 산 고객)\n"
        "  충성     : F>=4, M>=3 (자주 사고 금액도 준수)\n"
        "  잠재 충성: R>=4, F<=2 (최근 유입, 빈도는 아직 낮음)\n"
        "  이탈 위험: R<=2, F>=3 (예전엔 자주 샀으나 최근 조용)\n"
        "  휴면     : R<=2, F<=2 (오래 조용하고 빈도도 낮음)\n"
        "  미구매   : 상반기 유효 구매 이력 없음"
    )

    summary = segment_summary(rfm_full)
    print("\n[세그먼트별 인원·매출 비중]")
    print(summary.to_string())

    print("\n" + "=" * 70)
    print("문제 04. 멤버십 등급제 설계 시뮬레이션")
    print("=" * 70)

    table_A, table_B, budget_assumption, budget_A, budget_B = membership_simulation(rfm_full)

    print("\n[안 A: qcut 상위 5%/15%/80%]")
    print(table_A.to_string())
    print("\n[안 B: cut 고정금액 30만/100만]")
    print(table_B.to_string())

    print("\n[적립률 가정]")
    print(budget_assumption.to_string())
    print(f"\n[예산 추정] 안 A: {budget_A:,.0f}원 / 안 B: {budget_B:,.0f}원")

    print("\n[권고안]")
    vip_diff = abs(table_A.loc["VIP", "인원"] - table_B.loc["VIP", "인원"])
    print(
        f"  VIP 인원 차이 {vip_diff}명. qcut(안 A)은 인원이 고정되어 예산 예측이 쉽지만 매출 분포가\n"
        f"  바뀌면 등급 경계 금액이 매달 흔들린다. cut(안 B)은 금액 기준이 고정되어 고객이\n"
        f"  체감하는 등급 기준이 안정적이나, 매출 성장에 따라 VIP 인원이 계속 늘어 예산이\n"
        f"  불안정할 수 있다. 초기 도입은 예산 통제가 쉬운 안 A를 권고하고, 등급 기준을\n"
        f"  공지할 때는 안 B의 금액 기준을 함께 안내하는 절충을 제안한다."
    )
