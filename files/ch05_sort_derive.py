# -*- coding: utf-8 -*-
"""
5장. 정렬과 파생변수
  문제 21: sort_values 단일 컬럼 정렬
  문제 22: sort_values 다중 컬럼 + 정렬방향 다르게
  문제 23: pd.cut으로 연령대를 만들고 카테고리 순서 확인
  문제 24: 정렬 후 reset_index(drop=True)로 인덱스 정리
  문제 25: 매출 구간(cut)별 주문 건수 분포
"""
import pandas as pd
from common import build_net_sales_ledger, generate_customers


def problem_21(cust: pd.DataFrame = None):
    print("\n########## 문제 21: sort_values 단일 컬럼 정렬 ##########")
    if cust is None:
        cust = generate_customers()

    sorted_by_age = cust.sort_values(by="age")
    print("\n[나이 오름차순 정렬 - 상위 5명]")
    print(sorted_by_age.head().to_string(index=False))

    sorted_desc = cust.sort_values(by="age", ascending=False)
    print("\n[나이 내림차순 정렬 - 상위 5명]")
    print(sorted_desc.head().to_string(index=False))
    return sorted_by_age


def problem_22(cust: pd.DataFrame = None):
    print("\n########## 문제 22: 다중 컬럼 정렬 (방향 다르게) ##########")
    if cust is None:
        cust = generate_customers()

    bins = [0, 29, 49, 100]
    labels = ['20대이하', '30대-40대', '50대이상']
    cust = cust.copy()
    cust["age_group"] = pd.cut(cust["age"], bins=bins, labels=labels)

    result = cust.sort_values(by=["age_group", "age"], ascending=[True, False])
    print("\n[연령대 오름차순 + 연령대 내에서는 나이 내림차순]")
    print(result.head(10).to_string(index=False))
    return result


def problem_23(cust: pd.DataFrame = None):
    print("\n########## 문제 23: pd.cut 카테고리 순서 확인 ##########")
    if cust is None:
        cust = generate_customers()

    bins = [0, 29, 49, 100]
    labels = ['20대이하', '30대-40대', '50대이상']
    cust = cust.copy()
    cust["age_group"] = pd.cut(cust["age"], bins=bins, labels=labels)

    print("age_group dtype:", cust["age_group"].dtype)
    print("카테고리 순서:", cust["age_group"].cat.categories.tolist())
    print("정렬 시 문자열이 아닌 지정된 카테고리 순서를 따름을 확인:")
    print(cust.sort_values("age_group")["age_group"].unique())
    return cust


def problem_24(cust: pd.DataFrame = None):
    print("\n########## 문제 24: 정렬 후 reset_index(drop=True) ##########")
    if cust is None:
        cust = generate_customers()

    sorted_df = cust.sort_values(by="age")
    print("정렬 직후 인덱스(뒤섞임):", sorted_df.index[:5].tolist())

    reset_df = sorted_df.reset_index(drop=True)
    print("reset_index 후 인덱스   :", reset_df.index[:5].tolist())
    assert reset_df.index[0] == 0
    return reset_df


def problem_25(net_ledger: pd.DataFrame = None):
    print("\n########## 문제 25: 매출 구간별 주문 건수 분포 ##########")
    if net_ledger is None:
        net_ledger = build_net_sales_ledger()["net_ledger"]

    order_amount = net_ledger.groupby("order_id")["line_amount"].sum()

    bins = [0, 50000, 100000, 200000, float("inf")]
    labels = ["5만원 미만", "5-10만원", "10-20만원", "20만원 이상"]
    amount_group = pd.cut(order_amount, bins=bins, labels=labels)

    dist = amount_group.value_counts().sort_index()
    print("\n[주문 금액 구간별 건수]")
    print(dist)
    print("\n[비중(%)]")
    print((dist / dist.sum() * 100).round(1))
    return dist


if __name__ == "__main__":
    cust = generate_customers()
    problem_21(cust)
    problem_22(cust)
    problem_23(cust)
    problem_24(cust)
    problem_25(build_net_sales_ledger()["net_ledger"])
