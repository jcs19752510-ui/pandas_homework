# -*- coding: utf-8 -*-
"""
3장. merge로 데이터 병합하기
  문제 11: 주문상품(oi) + 고객(customers) inner merge 후 shape 검증
  문제 12: merge 방식(inner/left)별 행 수 비교 - 데이터 유실 확인
  문제 13: 고객 키 중복(fan-out) 문제 확인 및 처리
  문제 14: 병합 데이터로 카테고리별 고객 연령대 분포 분석
  문제 15: 병합 데이터로 채널별 성별 매출 비교
"""
import pandas as pd
from common import build_net_sales_ledger


def problem_11():
    print("\n########## 문제 11: oi + customers inner merge ##########")
    data = build_net_sales_ledger()
    oi, orders, customers = data["oi"], data["orders"], data["customers"]

    # oi 에는 customer_id가 없으므로 orders를 거쳐야 함 -> 먼저 orders와 결합
    oi_with_order = oi.merge(orders, on="order_id", how="inner")

    m = oi_with_order.merge(
        right=customers,
        on="customer_id",
        how="inner",
    )
    print("병합 전 oi 행 수:", oi.shape[0])
    print("병합 후 m 행 수 :", m.shape[0])
    print("m.shape:", m.shape)
    return m, data


def problem_12(data: dict = None):
    print("\n########## 문제 12: merge 방식별(inner/left) 행 수 비교 ##########")
    if data is None:
        data = build_net_sales_ledger()
    oi, orders, customers = data["oi"], data["orders"], data["customers"]
    oi_with_order = oi.merge(orders, on="order_id", how="inner")

    # 일부러 고객 마스터에서 일부 고객을 제거해 "유실" 상황을 재현
    customers_missing = customers[customers["customer_id"] % 7 != 0]

    m_inner = oi_with_order.merge(customers_missing, on="customer_id", how="inner")
    m_left = oi_with_order.merge(customers_missing, on="customer_id", how="left")

    print("원본 oi_with_order 행 수:", oi_with_order.shape[0])
    print("inner merge 행 수      :", m_inner.shape[0], "(고객정보 없는 행은 제거됨)")
    print("left merge 행 수       :", m_left.shape[0], "(원본 행 수 그대로 유지)")
    print("left merge 후 결측(고객정보 없음) 건수:", m_left["gender_ko"].isna().sum())
    return m_inner, m_left


def problem_13(data: dict = None):
    print("\n########## 문제 13: 고객 키 중복(fan-out) 문제 확인 ##########")
    if data is None:
        data = build_net_sales_ledger()
    orders, customers = data["orders"], data["customers"]

    print("병합 전 orders 행 수:", orders.shape[0])
    dup_count = customers["customer_id"].duplicated().sum()
    print("customers customer_id 중복 개수:", dup_count, "(정상: 0)")

    # 일부러 고객 마스터를 중복시켜 fan-out 문제를 재현
    customers_dup = pd.concat([customers, customers[customers["customer_id"] <= 5]], ignore_index=True)
    print("일부러 중복시킨 customers_dup 중복 개수:",
          customers_dup["customer_id"].duplicated().sum())

    m_dup = orders.merge(customers_dup, on="customer_id", how="inner")
    print("중복 마스터로 merge한 결과 행 수:", m_dup.shape[0],
          "(정상 대비 늘어났으면 fan-out 발생)")

    # 해결책: merge 전 중복 제거
    customers_clean = customers_dup.drop_duplicates(subset="customer_id")
    m_clean = orders.merge(customers_clean, on="customer_id", how="inner")
    print("중복 제거 후 merge 결과 행 수:", m_clean.shape[0], "(orders 행 수와 같아야 정상)")
    assert m_clean.shape[0] == orders.shape[0]
    return m_clean


def problem_14(m: pd.DataFrame = None):
    print("\n########## 문제 14: 카테고리별 고객 연령대 분포 ##########")
    if m is None:
        m, _ = problem_11()

    bins = [0, 29, 49, 100]
    labels = ['20대이하', '30대-40대', '50대이상']
    m = m.copy()
    m["age_group"] = pd.cut(m["age"], bins=bins, labels=labels)

    dist = pd.crosstab(m["category"], m["age_group"], normalize="index").round(3)
    print("\n[카테고리별 연령대 구성비] (행 합=1)")
    print(dist)
    return dist


def problem_15(m: pd.DataFrame = None):
    print("\n########## 문제 15: 채널별 성별 매출 비교 ##########")
    if m is None:
        m, _ = problem_11()

    pd.set_option('display.float_format', '{:,.0f}'.format)
    pt = pd.pivot_table(
        data=m, index="gender_ko", columns="channel",
        values="amount", aggfunc="sum", margins=True, margins_name="합", fill_value=0,
    )
    print("\n[성별 x 채널 매출]")
    print(pt)
    return pt


if __name__ == "__main__":
    m, data = problem_11()
    problem_12(data)
    problem_13(data)
    problem_14(m)
    problem_15(m)
