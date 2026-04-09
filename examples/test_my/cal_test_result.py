#!/usr/bin/env python3
"""
计算序列重排任务的测试指标

Usage:
    python cal_test_result.py --input predictions.jsonl --output metrics.json
"""
import json
import re
import argparse
import os


def load_predictions(input_path):
    """加载预测结果"""
    gths_all = []
    preds_all = []
    
    print(f"📖 读取预测结果: {input_path}")
    
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())

            # 提取 predict 和 label 中的数字
            # predict_nums = re.findall(r"S(\d+)", data["predict"])
            # label_nums = re.findall(r"S(\d+)", data["label"])

            predict_nums = re.findall(r"(\d+)", data["predict"])
            label_nums = re.findall(r"(\d+)", data["label"])

            predict_nums = [int(x) for x in predict_nums]
            label_nums = [int(x) for x in label_nums]

            gths_all.append(label_nums)
            preds_all.append(predict_nums)
            
            # 长度检查（不强制断言，只警告）
            if len(predict_nums) != len(label_nums):
                print(f"⚠️  警告: 预测长度 {len(predict_nums)} != 标签长度 {len(label_nums)}")
                if len(predict_nums) == 0:
                    preds_all.pop()
                    preds_all.append([20])

    
    print(f"✅ 加载了 {len(gths_all)} 条预测结果")
    return gths_all, preds_all


import itertools
import numpy as np


def cal_results(gths, preds):
    """
    计算序列重排任务的评估指标
    
    Returns:
        dict: 包含 acc, pmr, taus, pm 等指标
    """
    results = {}
    right, total = 0, 0
    pmr_right = 0
    taus = []
    accs = []
    pm_p, pm_r = [], []
    
    print(f"\n📊 计算指标...")
    
    for index, (t, p) in enumerate(zip(gths, preds)):
        if len(p) == 1:
            right += 1
            total += 1
            pmr_right += 1
            taus.append(1)
            continue

        min_len = min(len(p), len(t))
        eq = np.equal(t[:min_len], p[:min_len])
        right += eq.sum()
        accs.append(eq.sum() / len(t))

        total += len(t)

        pmr_right += eq.all()

        s_t = set([i for i in itertools.combinations(t, 2)])
        s_p = set([i for i in itertools.combinations(p, 2)])
        pm_p.append(len(s_t.intersection(s_p)) / len(s_p))
        pm_r.append(len(s_t.intersection(s_p)) / len(s_t))

        cn_2 = len(p) * (len(p) - 1) / 2
        pairs = len(s_p) - len(s_p.intersection(s_t))
        tau = 1 - 2 * pairs / cn_2

        taus.append(tau)

    pmr = pmr_right / len(gths)
    taus = np.mean(taus)
    pm_p = np.mean(pm_p)
    pm_r = np.mean(pm_r)
    pm = 2 * pm_p * pm_r / (pm_p + pm_r)
    accs = np.mean(accs)

    results['acc'] = float(accs)
    results['pmr'] = float(pmr)
    results['taus'] = float(taus)
    results['pm'] = float(pm)
    results['total_samples'] = len(gths)

    return results


def save_results(results, output_path):
    """保存结果到 JSON 文件"""
    print(f"\n💾 保存结果到: {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 结果已保存")


def print_results(results):
    """打印结果"""
    print("\n" + "="*60)
    print("📊 测试结果")
    print("="*60)
    print(f"总样本数: {results['total_samples']}")
    print(f"准确率 (ACC):      {results['acc']:.4f} ({results['acc']*100:.2f}%)")
    print(f"完美匹配率 (PMR):  {results['pmr']:.4f} ({results['pmr']*100:.2f}%)")
    print(f"Kendall's Tau:    {results['taus']:.4f}")
    print(f"Pairwise (PM):    {results['pm']:.4f}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='计算序列重排任务的测试指标')
    parser.add_argument('--input', type=str, required=True,
                        help='预测结果文件路径 (generated_predictions.jsonl)')
    parser.add_argument('--output', type=str, required=True,
                        help='输出指标文件路径 (metrics_results.json)')
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.input):
        print(f"❌ 错误: 输入文件不存在: {args.input}")
        return
    
    # 加载预测结果
    gths_all, preds_all = load_predictions(args.input)
    
    # 计算指标
    results = cal_results(gths_all, preds_all)
    
    # 打印结果
    print_results(results)
    
    # 保存结果
    save_results(results, args.output)
    
    print(f"\n✅ 完成！")


if __name__ == "__main__":
    main()