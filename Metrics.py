import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def Metrics_Calculate(pred, target):
    pred_log = pred
    pred = torch.exp(pred)
    # calculate threshold: % of y with y_pred/y_target < threshold
    ratio = torch.max(pred / target, target / pred)
    total = torch.numel(ratio)
    t1 = torch.numel(ratio[ratio < 1.25]) / total
    t2 = torch.numel(ratio[ratio < 1.25**2]) / total
    t3 = torch.numel(ratio[ratio < 1.25**3]) / total

    t1 /= total
    t2 /= total
    t3 /= total

    # calculate abs relative difference
    abs_error = torch.abs(pred-target) / target
    abs_error = torch.mean(abs_error)

    # calculate squared relative difference
    squared_error = torch.pow(pred - target, 2) / target
    squared_error = torch.mean(squared_error)

    # calculate RMSE(linear)
    rmse_linear = torch.pow(pred - target, 2)
    rmse_linear = torch.sqrt(torch.mean(rmse_linear))

    # calculate RMSE(log)
    rmse_log = torch.pow(pred_log - torch.log(target), 2)
    rmse_log = torch.sqrt(torch.mean(rmse_log))

    return t1, t2, t3, abs_error, squared_error, rmse_linear, rmse_log
