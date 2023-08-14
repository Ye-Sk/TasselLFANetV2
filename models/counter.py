"""
@author: Jianxiong Ye
"""

import asyncio
import numpy as np
from pathlib import Path

from models.utils.plot import plot_mc_curve

# ------------------------------------------------------conf range------------------------------------------------------ #
conf_th = np.linspace(0, 1, 1000)  # (1000 points)
# ------------------------------------------------------conf range------------------------------------------------------ #

# ------------------------------------------------------count eval------------------------------------------------------ #
class CountMetrics:
    @staticmethod
    def mae(gt, pd):
        return np.mean(np.abs(np.array(gt) - np.array(pd)))

    @staticmethod
    def rmse(gt, pd):
        return np.sqrt(np.mean((np.array(gt) - np.array(pd)) ** 2))

    @staticmethod
    def rmae(gt, pd):
        return np.mean(np.abs(np.array(pd)[np.array(gt) > 0] - np.array(gt)[np.array(gt) > 0]) / np.array(gt)[np.array(gt) > 0])

    @staticmethod
    def rrmse(gt, pd):
        return np.sqrt(np.mean((np.array(pd)[np.array(gt) > 0] - np.array(gt)[np.array(gt) > 0])**2 / np.array(gt)[np.array(gt) > 0]**2))

    @staticmethod
    def rsquared(gt, pd, epsilon=1e-16):
        return np.corrcoef(np.array(gt) + epsilon, np.array(pd) + epsilon)[0, 1] ** 2
# ------------------------------------------------------count eval------------------------------------------------------ #

metrics = [('MAE', CountMetrics.mae), ('RMSE', CountMetrics.rmse), ('rMAE', CountMetrics.rmae), ('rRMSE', CountMetrics.rrmse), ('R-squared', CountMetrics.rsquared)]

def extract_metric_values(data, metric_name):
    """
        Extract the values of a specific metric from a data list.

        Args:
            data: list of metric-value pairs.
            metric_name: str, name of the metric to extract values for.

        Returns:
            values: numpy array, extracted values of the specified metric.
    """
    values = [metric[1] for item in data for metric in item if metric[0] == metric_name]
    return np.array(values)

async def plot_count_curves(data, save_dir, names):
    coroutines = []

    for metric_name in ['MAE', 'RMSE', 'rMAE', 'rRMSE', 'R-squared']:
        metric_values = extract_metric_values(data, metric_name)
        metric_values = np.reshape(metric_values, (1, -1))
        min_value = np.min(metric_values)
        max_value = np.max(metric_values)
        metric_values = (metric_values - min_value) / (max_value - min_value)

        coroutine = plot_mc_curve(conf_th, metric_values, Path(save_dir) / f'{metric_name}_curve.png', names, ylabel=metric_name, task='counting')
        coroutines.append(coroutine)

    await asyncio.gather(*coroutines)

def count_parse(matrix):
    """
        Count the number of detections above different confidence thresholds.

        Args:
            matrix: numpy array, matrix of detection results with confidence scores.

        Returns:
            detection_counts: list, counts of detections above each confidence threshold.
    """
    detection_counts = []
    for threshold in conf_th:
        detections_above_threshold = matrix[matrix[:, 4] >= threshold]
        detection_count = detections_above_threshold.shape[0]
        detection_counts.append(detection_count)

    return detection_counts

def calculate_count_values(gt, pd):
    results = [[metric_name, round(metric_func(gt, pd), 4)] for metric_name, metric_func in metrics]
    return results

def extract_best_values(data):
    best_values = {}

    for metric_name, _ in metrics:
        values = extract_metric_values(data, metric_name)
        if metric_name in ['MAE', 'rMAE', 'RMSE', 'rRMSE']:
            best_value = min(values)
        else:
            best_value = max(values)
        best_values[metric_name] = best_value

    return best_values

def ct_eval(targets, preds, save_dir, names):
    conf_result = []
    data = [[len(targets[i]), count_parse(preds[i][0])] for i in range(len(preds))]

    for j, threshold in enumerate(conf_th):
        pd_data = [data[k][1][j] for k in range(len(data))]
        gt_data = [data[k][0] for k in range(len(data))]
        result = calculate_count_values(gt_data, pd_data)
        conf_result.append(result)
        # logger.info(f"Threshold: {threshold}, gt: {gt_data}, pd: {pd_data}, result: {result}")

    # coroutine plot
    asyncio.run(plot_count_curves(conf_result, save_dir, names))
    return extract_best_values(conf_result)