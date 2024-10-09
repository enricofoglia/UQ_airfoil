'''
Evaluate the quality of the predictions and their uncertainty. 
'''
import warnings

from typing import (
    Optional,
    Union,
    Tuple
)

import torch 

from scipy.integrate import simpson
import scipy.stats as stats

import numpy as np
import matplotlib.pyplot as plt


# @title AUCE metric
def auce_plot(y:np.ndarray, preds:np.ndarray, std:np.ndarray,
              plot:Optional[bool]=True, get_values:Optional[bool]=False
              ) -> Union[float, Tuple[float,np.ndarray,np.ndarray]]:
    r'''Plot the quatile calibration curve and return the Area Under the 
    Calibration Error curve metric. The computation of the predicted confidence
    interval is preformed based on the hypothesis that :math:`p(y\vert x,\theta)` 
    is a Gaussian with mean :obj:`preds` and standard deviation :obj:`std`.

    Args:
        y (numpy.ndarray): ground truth data
        preds (numpy.ndarray): predicted mean
        std (numpy.ndarray): predicted standard deviation
        plot (bool, optional): wheater to plot the results (default :obj:`True`)
        get_values (bool, optional): wheter to return the p_err and p_pred
          arrays (default :obj:`False`)
    '''
    err = y - preds
    abs_err = np.abs(err)

    # CDF and quantiles of the error
    abs_err_sorted = np.sort(abs_err)
    p_err = np.arange(0,len(abs_err))/(len(abs_err)-1)

    p_pred = []
    for p in p_err:
        q = stats.halfnorm.ppf(p, scale=std)
        count = np.sum(abs_err<=q)
        p_pred.append(count/len(q))
    p_pred = np.array(p_pred)

    # AUCE score
    indices = np.argsort(p_pred)
    auce = simpson(np.abs(p_err[indices] - p_pred[indices]),p_pred[indices])

    if plot:
        fig, ax = plt.subplots()
        ax.plot(p_err, p_pred)
        ax.fill_between(p_err, p_pred, p_err, color='tab:blue', alpha=0.3)
        ax.plot([0,1],[0,1],'k--')
        ax.set_ylabel('True probability')
        ax.set_xlabel('Predicted probability')
        ax.set_title(f'Calibration Plot, AUCE = {auce:.2f}')
        ax.set_xlim([0,1])
        ax.set_ylim([0,1])
        ax.legend()

    if get_values: return auce, p_err, p_pred
    return auce

# @title ENCE metric
def ece_plot(y_test:np.ndarray, mu:np.ndarray, var:np.ndarray,
              B:Optional[int] = 20, binning:Optional[str]='equal',
              plot:Optional[bool]=True, order:Optional[int]=1,
              get_values:Optional[bool]=False, use_last:Optional[bool]=False
              )->Union[float, Tuple[float,np.ndarray,np.ndarray]]:
    r'''Produce the variance-calibration curve and return the Expected 
    Calibration Error metric. Based of the definition of calibration:
    
    .. math::
        \widehat{\sigma}_{\theta}(x) = \mathbb{E}_{X,Y}[(\widehat{\mu}_{\theta}(X)-Y)^2
        \;\vert\; \widehat{\sigma}_{\theta}(X)=\widehat{\sigma}_{\theta}(x)]

    Args:
        y_test (numpy.ndarray): ground truth data
        preds (numpy.ndarray): predicted mean
        std (numpy.ndarray): predicted standard deviation
        B (int, optional): number of bins (default :obj:`20`)
        binning (str, optional): type of binning can be either equal width bins (:obj:`"equal"`), equal number of samples per bin (:obj:`"quantile"`) or by k-means clustering (:obj:`"k-means"`) (default :obj:`"equal"`)
        plot (bool, optional): wheater to plot the results (default :obj:`True`)
        order (int, optional): order of the calibration (default :obj:`1`)
        get_values (bool, optional): wheter to return the rmse and rmv arrays (default :obj:`False`)
        use_last (bool, optional): wheter to use the last bin in the calculation
    '''

    # sort and bin by increasing std
    indices = np.argsort(var)
    var = var[indices]
    y_test = y_test[indices]
    mu = mu[indices]
    if binning == 'equal':
        bin_edges = np.linspace(var.min(), var.max(), B + 1)
        bins = np.digitize(var, bin_edges)
    elif binning == 'quantile':
        bins = np.digitize(var, np.quantile(var, np.linspace(0,1,B + 1)))
    elif binning == 'kmeans':
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=B, random_state=0, init='k-means++', n_init=1).fit(var[..., None])
        bins = kmeans.labels_
    else:
        raise ValueError('binning must be equal or quantile')

    rmse = []
    rmv  = []
    rmv2 = []
    bin_size = []
    for i in np.unique(bins):
        y_sample = y_test[bins==i]
        bin_size.append(len(y_sample))
        pred_mu = mu[bins==i]
        pred_var = var[bins==i]

        rmse.append(np.sqrt(np.mean((y_sample.squeeze()-pred_mu.squeeze())**2)))
        rmv2.append(np.sqrt(np.var(np.abs(y_sample.squeeze()-pred_mu.squeeze()))))
        rmv.append(np.sqrt(np.mean(pred_var)))

    rmv = np.array(rmv)
    rmse = np.array(rmse)
    bin_size = np.array(bin_size)

    indices = np.argsort(rmv)
    rmv = rmv[indices]
    rmse = rmse[indices]
    bin_size = bin_size[indices]

    if not use_last:
        rmv = rmv[:-1]
        rmse = rmse[:-1]
        bin_size = bin_size[:-1]


    # ence = np.mean(np.abs(rmv-rmse)/rmv)
    # ence = simpson(np.abs(rmv - rmse),rmv) # area under the curve is uninformative if the x axis change!
    if order == 1: ence = np.sum(np.abs(rmv-rmse)*bin_size)/np.sum(bin_size) # true expectation over bins of different sizes
    elif order == 2:ence = np.sum(np.abs(rmv-rmv2)*bin_size)/np.sum(bin_size)
    
    cv = np.sqrt(np.sqrt(var).var())/np.mean(np.sqrt(var))

    if plot:
        fig, ax = plt.subplots()
        if order == 1:
            ax.scatter(rmv, rmse, s=bin_size*100/max(bin_size), edgecolors='k')
            ax.plot(rmv, rmse, color='tab:blue')
            ax.fill_between(rmv, rmse, rmv, color='tab:blue', alpha=0.3)
        if order == 2:
            ax.scatter(rmv, rmv2, s=bin_size*100/max(bin_size), edgecolors='k')
            ax.plot(rmv, rmv2, color='tab:blue')
            ax.fill_between(rmv, rmv2, rmv, color='tab:blue', alpha=0.3)
        ax.plot([min(rmv), max(rmv)], [min(rmv), max(rmv)], 'k--')
        ax.set_ylabel(f'RMSE-{order}')
        ax.set_xlabel('RMV')
        ax.set_title(f'Calibration plot; ENCE-{order} = {ence:.2f}, $c_v$= {cv:.2f}')
        ax.legend()
    if get_values: return ence, rmv, rmse
    else: return ence
