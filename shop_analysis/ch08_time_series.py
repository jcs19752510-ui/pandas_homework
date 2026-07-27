import pandas as pd
import numpy as np
from shop_analysis.common_order_items import net_ledger

pd.set_option("display.float_format", lambda x: f"{x:,.0f}")


def weekly_trend(net_lines):
    daily = net_lines.set_index("order_datetime")["line_amount"].resample("D").sum()
    weekly = daily.resample("W-SUN").sum()
    weekly_ma4 = weekly.rolling(4).mean()

    table = pd.DataFrame({"주간매출": weekly, "4주이동평균": weekly_ma4})

    stable_ma = weekly_ma4.dropna()
    growth_rate = stable_ma.iloc[-1] / stable_ma.iloc[0] - 1
    return table, growth_rate


def hourly_demand_profile(net_lines):
    orders = net_lines.groupby("order_id").agg(
        order_datetime=("order_datetime", "first"), channel=("channel", "first")
    )
    orders["hour"] = orders["order_datetime"].dt.hour
    orders["is_weekend"] = orders["order_datetime"].dt.dayofweek >= 5
    orders["date"] = orders["order_datetime"].dt.date

    n_weekday_days = orders.loc[~orders["is_weekend"], "date"].nunique()
    n_weekend_days = orders.loc[orders["is_weekend"], "date"].nunique()

    weekday_profile = (
        orders[~orders["is_weekend"]].groupby("hour").size() / n_weekday_days
    ).rename("평일_시간당평균주문")
    weekend_profile = (
        orders[orders["is_weekend"]].groupby("hour").size() / n_weekend_days
    ).rename("주말_시간당평균주문")
    profile = pd.concat([weekday_profile, weekend_profile], axis=1)

    overall_hourly = orders.groupby("hour").size()
    threshold = overall_hourly.quantile(0.75)
    peak_hours = overall_hourly[overall_hourly >= threshold].index.tolist()

    channel_hourly = orders.groupby(["channel", "hour"]).size().unstack(fill_value=0)
    channel_peak = channel_hourly.apply(lambda row: row.nlargest(3).index.tolist(), axis=1)

    return profile, sorted(peak_hours), channel_peak


def july_forecast(net_lines):
    daily = net_lines.set_index("order_datetime")["line_amount"].resample("D").sum()

    def forecast_A(daily_series):
        monthly = daily_series.resample("ME").sum()
        recent3 = monthly.iloc[-3:]
        growth_rates = recent3.pct_change().dropna()
        avg_growth = growth_rates.mean()
        return monthly.iloc[-1] * (1 + avg_growth)

    def forecast_B(daily_series, target_days_in_month=31, target_year=2024, target_month=7):
        by_dow = daily_series.groupby(daily_series.index.dayofweek).mean()
        dates = pd.date_range(f"{target_year}-{target_month:02d}-01", periods=target_days_in_month, freq="D")
        return sum(by_dow[d.dayofweek] for d in dates)

    forecast_A_value = forecast_A(daily)
    forecast_B_value = forecast_B(daily)

    daily_train = daily[daily.index < "2024-06-01"]
    actual_june = daily[(daily.index >= "2024-06-01") & (daily.index < "2024-07-01")].sum()

    backtest_A = forecast_A(daily_train)
    backtest_B = forecast_B(daily_train, target_days_in_month=30, target_month=6)

    err_A = abs(backtest_A - actual_june) / actual_june
    err_B = abs(backtest_B - actual_june) / actual_june

    return forecast_A_value, forecast_B_value, actual_june, backtest_A, backtest_B, err_A, err_B


if __name__ == "__main__":
    net_lines = net_ledger()

    print("=" * 70)
    print("문제 21. 주간 매출 추세와 이동평균")
    print("=" * 70)
    table, growth_rate = weekly_trend(net_lines)
    print("\n[주간 매출 및 4주 이동평균]")
    print(table.to_string())
    print(f"\n[추세 판정] 이동평균 안정 구간 기준 성장률: {growth_rate:+.1%}")
    print(
        "[보고 문구] 일별 매출은 요일 구성과 무작위 변동으로 톱니처럼 출렁이므로, 하루치 등락만 보고\n"
        "'회복세'나 '꺾임'을 판단하면 잘못된 신호에 반응하게 된다. 4주 이동평균처럼 노이즈를 걷어낸\n"
        "지표로 판단해야 한다."
    )

    print("\n" + "=" * 70)
    print("문제 22. 시간 단위 수요 프로파일과 피크 운영 계획")
    print("=" * 70)
    profile, peak_hours, channel_peak = hourly_demand_profile(net_lines)
    print("\n[평일/주말 시간대별 평균 주문건수]")
    print(profile.round(1).to_string())
    print(f"\n[피크 시간대(상위 25%)] {peak_hours}")
    print("\n[채널별 피크 상위 3시간대]")
    print(channel_peak.to_string())
    print(
        "\n[운영 배치 제안] 피크 시간대에 CS 인력과 당일출고 마감을 집중 배치한다.\n"
        "[문제09와의 차이] 문제09는 '세션이 시작된 시각' 기준 전환율(광고 입찰 관점)이고,\n"
        "이 문제는 '실제 결제가 완료된 시각' 기준 주문 건수(운영 배치 관점)이므로 서로 다른 지표다.\n"
        "세션이 몰리는 시간과 실제 결제가 몰리는 시간은 체류시간 차이로 인해 다를 수 있다."
    )

    print("\n" + "=" * 70)
    print("문제 23. 7월 수요 나이브 예측과 한계 보고")
    print("=" * 70)
    fA, fB, actual_june, bA, bB, errA, errB = july_forecast(net_lines)
    print(f"\n[A안: 최근3개월 평균성장률 외삽] 7월 전망 {fA:,.0f}원")
    print(f"[B안: 6월 요일별 일평균 x 7월 달력] 7월 전망 {fB:,.0f}원")
    print("\n[가정표]")
    print("  A안: 최근 3개월(4~6월) 월간 성장률의 평균이 7월에도 유지된다고 가정")
    print("  B안: 6월의 요일별 평균 패턴이 7월에도 그대로 반복된다고 가정 (계절성 없음 가정)")
    print(f"\n[6월 백테스트] 실제 6월 순매출: {actual_june:,.0f}원")
    print(f"  A안 예측: {bA:,.0f}원 (오차 {errA:.1%})")
    print(f"  B안 예측: {bB:,.0f}원 (오차 {errB:.1%})")
    better = "A안" if errA < errB else "B안"
    print(f"\n[권고] 백테스트 오차가 더 작은 {better}을 채택한다.")
    lo, hi = min(fA, fB) * 0.9, max(fA, fB) * 1.1
    print(f"[전망 구간] 보수 {lo:,.0f}원 ~ 낙관 {hi:,.0f}원")