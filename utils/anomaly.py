import numpy as np

def compute_auc(seq, sos, eos):
    return np.sum(seq[sos:eos])

def classify_season(cumValDist, cumVal):
    """
    Dtermine if cumVal is critical according to ASAP rules
    :param cumValDist: historical values of the NDVI cumulative value (between sos and eos, included)
    :param cumVal: current year NDVI cumulative value
    :return: 1 if the current val is critical, 0 otherwise

    Pixels having a zNDVIc value smaller than -1 are flagged as critical only if also the following condition holds:
    mNDVId / HISTORICAL MEAN(mNDVI) * 100 < -10 [%]
    so if the mean NDVI diff or the current year / historical mean of NDVI is less that- 0.1
    """

    percThreshOnNDVId = -10
    zThresh = -1
    n = len(cumValDist)

    # compute historical mu and sd
    mu = np.nanmean(cumValDist)
    sd_sample = np.nanstd(cumValDist, ddof=1) #sd = np.nanstd(cumValDist)

    # z score
    zScore = (cumVal-mu)/sd_sample
    #print('mu sds Z', mu, sd_sample, zScore)

    # % mNDVId with respect to historical NDVIm
    delta = (cumVal/mu - 1) * 100
    #print('delta mNDVId %', delta)

    if (zScore < zThresh) and (delta < percThreshOnNDVId):
        return 1
    else:
        return 0
