@ app.get("/ssq/backtest", tags=["双色球历史数据"])
async def ssq_backtest(periods: int = 200, mode: str = "day_gan"):
    """
    双色球玄学维度回测验证

    对历史每期数据，用/ganzhi相同逻辑计算各维度号码，与实际开奖号码对比。
    - **periods**: 回测最近N期，默认200，最大2144
    - **mode**: 旺行判定逻辑，可选：
      - day_gan（默认）：日柱天干五行
      - day_zhi：日柱地支五行
      - majority：六柱综合众数
      - all：三种模式并行回测，对比结果并推荐最优

    返回每个维度在红球/蓝球中的命中率，与随机概率对比。
    当mode="all"时，额外返回三种模式的对比结果和推荐。
    """
    if not _SSQ_HISTORY:
        raise HTTPException(status_code=503, detail="历史数据未加载")
    periods = min(periods, len(_SSQ_HISTORY))
    data = _SSQ_HISTORY[:periods]

    if mode not in ("day_gan", "day_zhi", "majority", "all"):
        raise HTTPException(
            status_code=400,
            detail="mode参数错误，可选：day_gan / day_zhi / majority / all"
        )

    # 如果mode="all"，并行跑三种模式
    if mode == "all":
        results_all = {}
        for m in ["day_gan", "day_zhi", "majority"]:
            result = await _run_backtest(data, periods, m)
            results_all[m] = result
        # 对比三种模式，生成推荐
        recommend = _compare_backtest_modes(results_all, periods)
        return {
            "periods_tested": periods,
            "mode": "all",
            "results_all": results_all,
            "formatted_backtest_all": recommend["formatted"],
            "recommend_mode": recommend["recommend_mode"],
            "recommend_reason": recommend["reason"],
        }

    # 单模式回测
    result = await _run_backtest(data, periods, mode)
    return result


async def _run_backtest(data, periods, mode):
    """
    内部函数：对指定数据和mode执行回测
    """
    dim_stats = {
        "旺行": {"red_hit": 0, "red_total": 0, "blue_hit": 0, "blue_total": 0, "red_pool": 0, "blue_pool": 0},
        "生我行": {"red_hit": 0, "red_total": 0, "blue_hit": 0, "blue_total": 0, "red_pool": 0, "blue_pool": 0},
        "我生行·泄": {"red_hit": 0, "red_total": 0, "blue_hit": 0, "blue_total": 0, "red_pool": 0, "blue_pool": 0},
        "克我行": {"red_hit": 0, "red_total": 0, "blue_hit": 0, "blue_total": 0, "red_pool": 0, "blue_pool": 0},
        "我克行": {"red_hit": 0, "red_total": 0, "blue_hit": 0, "blue_total": 0, "red_pool": 0, "blue_pool": 0},
        "纳音五行": {"red_hit": 0, "red_total": 0, "blue_hit": 0, "blue_total": 0, "red_pool": 0, "blue_pool": 0},
        "六柱干支": {"red_hit": 0, "red_total": 0, "blue_hit": 0, "blue_total": 0, "red_pool": 0, "blue_pool": 0},
        "飞星方位": {"red_hit": 0, "red_total": 0, "blue_hit": 0, "blue_total": 0, "red_pool": 0, "blue_pool": 0},
    }

    _TIANGAN_LIST = ['甲', '乙', '丙', '丁', '戊', '己', '庚', '辛', '壬', '癸']
    _DIZHI_LIST = ['子', '丑', '寅', '卯', '辰', '巳', '午', '未', '申', '酉', '戌', '亥']
    month_dz_map_bt = {1: '丑', 2: '寅', 3: '卯', 4: '辰', 5: '巳', 6: '午',
                      7: '未', 8: '申', 9: '酉', 10: '戌', 11: '亥', 12: '子'}
    tg_start_map_bt = {'甲': '丙', '己': '丙', '乙': '戊', '庚': '戊', '丙': '庚', '辛': '庚',
                       '丁': '壬', '壬': '壬', '戊': '甲', '癸': '甲'}
    month_dz_order_bt = ['寅', '卯', '辰', '巳', '午', '未', '申', '酉', '戌', '亥', '子', '丑']

    from datetime import date as date_cls
    from lunarcalendar import Converter, Solar
    from collections import Counter

    base_date = date_cls(2000, 1, 7)

    for rec in data:
        date_str = rec["date"]
        if not date_str:
            continue
        parts = date_str.split('-')
        try:
            solar_date = date_cls(int(parts[0]), int(parts[1]), int(parts[2]))
        except:
            continue

        # 干支计算
        y_offset = solar_date.year - 1984
        year_gan = _TIANGAN_LIST[y_offset % 10]
        year_zhi = _DIZHI_LIST[y_offset % 12]
        diff = (solar_date - base_date).days
        day_gan = _TIANGAN_LIST[diff % 10]
        day_zhi = _DIZHI_LIST[diff % 12]
        month_zhi = month_dz_map_bt[solar_date.month]
        start_tg = tg_start_map_bt[year_gan]
        start_idx = _TIANGAN_LIST.index(start_tg)
        month_dz_idx = month_dz_order_bt.index(month_zhi)
        month_gan = _TIANGAN_LIST[(start_idx + month_dz_idx) % 10]

        # 旺行判定
        six_wx = [
            _TIANGAN_MAP[year_gan]["wuxing"], _DIZHI_RED_MAP[year_zhi]["wuxing"],
            _TIANGAN_MAP[month_gan]["wuxing"], _DIZHI_RED_MAP[month_zhi]["wuxing"],
            _TIANGAN_MAP[day_gan]["wuxing"], _DIZHI_RED_MAP[day_zhi]["wuxing"],
        ]
        wx_counter = Counter(six_wx)

        if mode == "day_zhi":
            day_wuxing = _DIZHI_RED_MAP[day_zhi]["wuxing"]
        elif mode == "majority":
            day_wuxing = wx_counter.most_common(1)[0][0]
        else:
            day_wuxing = _TIANGAN_MAP[day_gan]["wuxing"]

        shengke = _get_shengke_info(day_wuxing)

        # 阴历
        try:
            solar = Solar(solar_date.year, solar_date.month, solar_date.day)
            lunar = Converter.Solar2Lunar(solar)
            lunar_day = lunar.day
        except:
            continue

        # 实际开奖号码
        actual_red = set(rec["red"])
        actual_blue = rec["blue"]

        # 逐维度统计
        def _count_dim(dim_name, red_set, blue_set=None):
            dim_stats[dim_name]["red_hit"] += len(actual_red & red_set)
            dim_stats[dim_name]["red_total"] += 6
            dim_stats[dim_name]["red_pool"] = len(red_set)
            if blue_set is not None:
                dim_stats[dim_name]["blue_hit"] += 1 if actual_blue in blue_set else 0
                dim_stats[dim_name]["blue_total"] += 1
                dim_stats[dim_name]["blue_pool"] = len(blue_set)

        # 旺行
        _count_dim("旺行",
                    set(_WUXING_MAP[shengke["旺行"]]["red_balls"]),
                    set(_WUXING_MAP[shengke["旺行"]]["blue_balls"]))
        # 生我行
        _count_dim("生我行",
                    set(_WUXING_MAP[shengke["生我行"]]["red_balls"]),
                    set(_WUXING_MAP[shengke["生我行"]]["blue_balls"]))
        # 我生行·泄
        _count_dim("我生行·泄",
                    set(_WUXING_MAP[shengke["我生行(泄)"]]["red_balls"]))
        # 克我行
        _count_dim("克我行",
                    set(_WUXING_MAP[shengke["克我行"]]["red_balls"]),
                    set(_WUXING_MAP[shengke["克我行"]]["blue_balls"]))
        # 我克行
        _count_dim("我克行",
                    set(_WUXING_MAP[shengke["我克行"]]["red_balls"]))

        # 纳音五行
        day_ganzhi = day_gan + day_zhi
        day_nayin = _NAYIN_MAP.get(day_ganzhi, "")
        nayin_wuxing = _NAYIN_WUXING.get(day_nayin, "")
        if nayin_wuxing:
            _count_dim("纳音五行",
                        set(_WUXING_MAP[nayin_wuxing]["red_balls"]),
                        set(_WUXING_MAP[nayin_wuxing]["blue_balls"]))

        # 六柱干支
        liuzhu_red = set()
        liuzhu_blue = set()
        for tg, dz in [(year_gan, year_zhi), (month_gan, month_zhi), (day_gan, day_zhi)]:
            liuzhu_red.update(_TIANGAN_MAP[tg]["red_balls"])
            liuzhu_red.update(_DIZHI_RED_MAP[dz]["red_balls"])
            liuzhu_blue.add(_DIZHI_BLUE_MAP[dz])
        _count_dim("六柱干支", liuzhu_red, liuzhu_blue)

        # 飞星方位
        bagua = _DIZHI_BAGUA_MAP.get(day_zhi, {})
        if bagua:
            _count_dim("飞星方位",
                        set(bagua["red_balls"]),
                        set(bagua["blue_balls"]))

    # 计算命中率
    results = []
    for dim_name, stats in dim_stats.items():
        if stats["red_total"] == 0:
            continue
        red_hit_rate = round(stats["red_hit"] / stats["red_total"] * 100, 2)
        red_expected = round(stats["red_pool"] / 33 * 100, 2) if stats["red_pool"] > 0 else 0
        red_lift = round(red_hit_rate - red_expected, 2)

        blue_hit_rate = round(stats["blue_hit"] / stats["blue_total"] * 100, 2) if stats["blue_total"] > 0 else 0
        blue_expected = round(stats["blue_pool"] / 16 * 100, 2) if stats["blue_pool"] > 0 else 0
        blue_lift = round(blue_hit_rate - blue_expected, 2)

        if red_lift > 2:
            verdict = "✅有效"
        elif red_lift < -2:
            verdict = "❌负面"
        else:
            verdict = "⚠️中性"

        results.append({
            "dimension": dim_name,
            "red_hit": stats["red_hit"],
            "red_total": stats["red_total"],
            "red_hit_rate": red_hit_rate,
            "red_expected_rate": red_expected,
            "red_lift": red_lift,
            "blue_hit": stats["blue_hit"],
            "blue_total": stats["blue_total"],
            "blue_hit_rate": blue_hit_rate,
            "blue_expected_rate": blue_expected,
            "blue_lift": blue_lift,
            "verdict": verdict,
        })

    # 格式化输出
    lines = [f"【双色球玄学维度回测验证（近{periods}期，模式={mode}）】", ""]
    lines.append(f"基准：红球随机命中率≈{round(6 / 33 * 100, 2)}%/球，蓝球随机命中率≈{round(1 / 16 * 100, 2)}%")
    lines.append(f"提升值=实际命中率-期望命中率，>0=优于随机，<0=劣于随机")
    lines.append("")
    lines.append(f"{'维度':<10} {'红球命中':>8} {'红球命中率':>8} {'期望率':>6} {'提升':>6} {'蓝球命中率':>8} {'蓝球提升':>6} {'判定'}")
    lines.append("-" * 80)

    for r in results:
        blue_info = f"{r['blue_hit_rate']}%" if r['blue_total'] > 0 else "N/A"
        blue_lift_info = f"{r['blue_lift']}%" if r['blue_total'] > 0 else "N/A"
        lines.append(
            f"{r['dimension']:<10} {r['red_hit']:>5}/{r['red_total']:>3} "
            f"{r['red_hit_rate']:>7}% {r['red_expected_rate']:>5}% {r['red_lift']:>+5}% "
            f"{blue_info:>8} {blue_lift_info:>6} {r['verdict']}"
        )

    lines.append("")
    lines.append("💡 提升值>2%=有效维度，<-2%=负面维度，其余≈随机")

    formatted_backtest = "\n".join(lines)

    return {
        "periods_tested": periods,
        "mode": mode,
        "formatted_backtest": formatted_backtest,
        "details": results,
    }


def _compare_backtest_modes(results_all, periods):
    """
    对比三种模式的回测结果，生成formatted输出和推荐
    """
    mode_scores = {}
    for mode, result in results_all.items():
        details = result.get("details", [])
        red_lift_sum = sum(d["red_lift"] for d in details)
        blue_lift_sum = sum(d["blue_lift"] for d in details if d["blue_total"] > 0)
        valid_count = sum(1 for d in details if d["verdict"] == "✅有效")
        negative_count = sum(1 for d in details if d["verdict"] == "❌负面")
        mode_scores[mode] = {
            "red_lift_sum": red_lift_sum,
            "blue_lift_sum": blue_lift_sum,
            "valid_count": valid_count,
            "negative_count": negative_count,
            "score": red_lift_sum + blue_lift_sum * 0.5,
        }

    sorted_modes = sorted(mode_scores.items(), key=lambda x: x[1]["score"], reverse=True)
    best_mode = sorted_modes[0][0]
    best_score = sorted_modes[0][1]

    lines = [f"【多模式回测对比（近{periods}期）】", ""]
    lines.append(f"{'模式':<12} {'红球提升∑':>10} {'蓝球提升∑':>10} {'有效维度':>8} {'负面维度':>8} {'综合得分':>8}")
    lines.append("-" * 70)

    mode_names = {"day_gan": "日干模式", "day_zhi": "日支模式", "majority": "六柱众数"}
    for mode, scores in sorted_modes:
        lines.append(
            f"{mode_names.get(mode, mode):<12} "
            f"{scores['red_lift_sum']:>+8}% "
            f"{scores['blue_lift_sum']:>+8}% "
            f"{scores['valid_count']:>6}个 "
            f"{scores['negative_count']:>6}个 "
            f"{scores['score']:>+7.2f}"
        )

    lines.append("")
    lines.append(f"🏆 推荐模式：{mode_names.get(best_mode, best_mode)}")
    lines.append(f"   综合得分最高（{best_score['score']:+.2f}），红球提升∑{best_score['red_lift_sum']:+.2f}%，有效维度{best_score['valid_count']}个")

    lines.append("")
    lines.append("📊 各模式有效维度（提升值>2%）：")
    for mode, result in results_all.items():
        valid_dims = [d for d in result.get("details", []) if d["verdict"] == "✅有效"]
        dim_str = "、".join([d["dimension"] for d in valid_dims]) if valid_dims else "无"
        lines.append(f"  {mode_names.get(mode, mode)}：{dim_str}")

    lines.append("")
    lines.append("💡 使用建议：")
    lines.append(f"  1. 在/ssq/pick接口中使用 mode={best_mode} 参数")
    lines.append(f"  2. 融合选号将自动采用「{mode_names.get(best_mode, best_mode)}」的计算结果")

    formatted = "\n".join(lines)

    return {
        "formatted": formatted,
        "recommend_mode": best_mode,
        "reason": f"综合得分最高（{best_score['score']:+.2f}），红球提升∑{best_score['red_lift_sum']:+.2f}%",
    }
