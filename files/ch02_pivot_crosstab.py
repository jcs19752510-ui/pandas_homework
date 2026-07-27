# -*- coding: utf-8 -*-
"""
2장. 피벗과 크로스탭으로 보는 분포
  문제 06: 카테고리별 채널 매출 - 차트용 데이터 준비 (margins 제거)
  문제 07: 고객 연령대 구간화 (pd.cut)
  문제 08: 연령대 x 성별 교차표 - 성별 내부 구성비 (normalize='columns')
  문제 09: 연령대 x 성별 교차표 - 연령대 내부 구성비 (normalize='index')
  문제 10: 같은 지표를 pivot_table / crosstab 두 방식으로 각각 구현 후 비교
"""
import pandas as pd
from common import build_net_sales_ledger, generate_customers


def problem_06(gross_ledger: pd.DataFrame = None):
    print("\n########## 문제 06: 카테고리 x 채널 매출 - 차트용 데이터 준비 ##########")
    if gross_ledger is None:
        gross_ledger = build_net_sales_ledger()["gross_ledger"]

    pt = pd.pivot_table(
        data=gross_ledger, index="category", columns="channel",
        values="line_amount", aggfunc="sum",
        margins=True, margins_name="합", fill_value=0,
    )
    # 차트를 그릴 때는 합계 행/열을 제외해야 왜곡되지 않는다
    pt_chart = pt.drop(index="합", columns="합")
    print("\n[차트용 데이터] (합계 행/열 제외)")
    print(pt_chart)

    # 각 카테고리에서 매출 비중이 가장 큰 채널
    top_channel_per_category = pt_chart.idxmax(axis=1)
    print("\n[카테고리별 매출 1위 채널]")
    print(top_channel_per_category)
    return pt_chart


def problem_07(seed=None):
    print("\n########## 문제 07: 고객 연령대 구간화 (pd.cut) ##########")
    cust = generate_customers()

    bins = [0, 29, 49, 100]
    labels = ['20대이하', '30대-40대', '50대이상']
    cust["age_group"] = pd.cut(cust["age"], bins=bins, labels=labels)

    print("\n[연령대별 고객 수]")
    print(cust["age_group"].value_counts().sort_index())
    print("\n[샘플 확인]")
    print(cust.head(10).to_string(index=False))
    return cust


def problem_08(cust: pd.DataFrame = None):
    print("\n########## 문제 08: 연령대 x 성별 교차표 (normalize='columns') ##########")
    if cust is None:
        cust = problem_07()

    freq = pd.crosstab(
        cust["age_group"], cust["gender_ko"], normalize='columns'
    ).round(3)
    print("\n[성별 내부에서 연령대 구성비] (각 열의 합 = 1)")
    print(freq)
    print("\n열 합계 검산:", freq.sum(axis=0).round(3).to_dict())
    return freq


def problem_09(cust: pd.DataFrame = None):
    print("\n########## 문제 09: 연령대 x 성별 교차표 (normalize='index') ##########")
    if cust is None:
        cust = problem_07()

    freq = pd.crosstab(
        cust["age_group"], cust["gender_ko"], normalize='index'
    ).round(3)
    print("\n[연령대 내부에서 성별 구성비] (각 행의 합 = 1)")
    print(freq)
    print("\n행 합계 검산:", freq.sum(axis=1).round(3).to_dict())
    return freq


def problem_10(cust: pd.DataFrame = None):
    print("\n########## 문제 10: pivot_table vs crosstab 비교 ##########")
    if cust is None:
        cust = problem_07()

    # 같은 "연령대 x 성별 고객 수"를 두 가지 방식으로 각각 구현
    via_crosstab = pd.crosstab(cust["age_group"], cust["gender_ko"])

    via_pivot = pd.pivot_table(
        data=cust, index="age_group", columns="gender_ko",
        values="customer_id", aggfunc="count", fill_value=0,
    )
    # crosstab은 카운트 0인 조합도 int, pivot_table은 fill_value로 맞춰줌
    via_pivot = via_pivot.astype(via_crosstab.dtypes)

    print("\n[crosstab 결과]")
    print(via_crosstab)
    print("\n[pivot_table 결과]")
    print(via_pivot)

    same = via_crosstab.equals(via_pivot)
    print(f"\n두 결과가 동일한가? -> {same}")
    assert same, "crosstab과 pivot_table 결과가 다릅니다"
    return via_crosstab


if __name__ == "__main__":
    pt_chart = problem_06()
    cust = problem_07()
    problem_08(cust)
    problem_09(cust)
    problem_10(cust)
