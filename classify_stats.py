# Copyright 2013 Douglas Bagnall <douglas@halo.gen.nz> LGPL
import sys
from math import sqrt


def prepare_roc_data(results):
    results.sort()
    sum_true = sum(1 for x in results if x[1])
    sum_false = len(results) - sum_true

    tp_scale = 1.0 / (sum_true or 1)
    fp_scale = 1.0 / (sum_false or 1)
    return results, sum_true, sum_false, tp_scale, fp_scale


def draw_roc_curve(results, label='ROC', arrows=1):
    import matplotlib.pyplot as plt

    results, true_positives, false_positives, \
        tp_scale, fp_scale = prepare_roc_data(results)

    tp = []
    fp = []
    half = 0
    #distance from best
    dfb_p = (0, 0, 0)
    dfb_d = 99
    # distance from worst
    dfw_p = (0, 0, 0)
    dfw_d = 0
    # distance from diagonal
    dfd_p = (0, 0, 0)
    dfd_d = 0
    p95_p = None
    n95_p = (1.0, 1.0, 1.0)

    for result in results:
        score = result[0]
        target = result[1]
        false_positives -= not target
        true_positives -= target
        x = false_positives * fp_scale
        y = true_positives * tp_scale
        half += score < 0.5
        p = (x, y, score)

        if arrows > 1:
            #distance from worst
            d = (1 - x) * (1 - x) + y * y
            if d > dfw_d:
                dfw_p = p
                dfw_d = d
            #distance from best
            d = x * x + (1 - y) * (1 - y)
            if d < dfb_d:
                dfb_p = p
                dfb_d = d
        #distance from diagonal
        d = y - x
        if d > dfd_d:
            dfd_p = p
            dfd_d = d

        #positive 95
        if p95_p is None and y > 20.0 * x:
            p95_p = p

        # negative 95
        if 1.0 - x > 20.0 * (1.0 - y):
            n95_p = p

        fp.append(x)
        tp.append(y)

    if half < len(fp):
        hx = (fp[half - 1] + fp[half]) * 0.5
        hy = (tp[half - 1] + tp[half]) * 0.5
    else:
        hx = fp[half - 1]
        hy = tp[half - 1]

    fp.reverse()
    tp.reverse()
    plt.plot(fp, tp, label=label)
    if arrows:
        x, y, s = n95_p
        plt.annotate("95%% negative %.2g" % x, (x, y), (0.7, 0.7),
                     arrowprops={'width':1, 'color': '#0088aa'},
                     )
        if p95_p is not None:
            x, y, s = p95_p
            plt.annotate("95%% positive %.2g" % s, (x, y), (0.2, 0.2),
                         arrowprops={'width':1, 'color': '#8800aa'},
                     )
        plt.annotate("0.5", (hx, hy), (0.4, 0.4),
                     arrowprops={'width':1, 'color': '#00cc00'})
        x, y, s = dfd_p
        plt.annotate("furthest from diagonal %.2g" % s, (x, y), (0.5, 0.5),
                     arrowprops={'width':1, 'color': '#aa6600'},
                     )
    if arrows > 1:
        x, y, s = dfw_p
        plt.annotate("furthest from all bad %.2g" % s, (x, y), (0.3, 0.3),
                     arrowprops={'width':1, 'color': '#00cccc'},
                     )
        x, y, s = dfb_p
        plt.annotate("closest to all good %.2g" % s, (x, y), (0.6, 0.6),
                     arrowprops={'width':1, 'color': '#cc0000'},
                     )


def _calc_stats(results, include_scores=False):
    from math import sqrt, log
    (results, sum_true, sum_false,
     tp_scale, fp_scale) = prepare_roc_data(results)
    auc = 0
    sum_dfd = 0 #distance from diagonal (but signed)
    max_dfd = 0
    best_dfd_score = 0
    sum_dfc2 = 0 #distance from centre, squared
    max_dfc2 = 0
    sum_dfb, min_dfb = 0, 1e99 #distance from best
    pos_95 = 0
    neg_95 = 0
    briar = 0
    cross_entropy = 0
    pos_95_score = 1
    neg_95_score = 0
    min_dfb_score = 0

    px, py = 0, 0 # previous position for area calculation
    true_positives, false_positives = sum_true, sum_false
    best_tp = true_positives
    best_fp = false_positives
    for score, target in results:
        false_positives -= not target
        true_positives -= target
        x = false_positives * fp_scale
        y = true_positives * tp_scale

        #area under ROC curve
        dx = x - px
        dy = y - py
        auc += px * dy       # bottom rectangle
        auc += dx * dy * 0.5 # top triangle
        #XXX AUC never actually has a top triangle -- every step is
        # either vertical or horizontal. There ought to be a better way.
        px = x
        py = y

        #distance from diagonal (needs scaling by .707)
        d = y - x
        sum_dfd += d
        if d > max_dfd:
            max_dfd = d
            best_tp = true_positives
            best_fp = false_positives
            best_dfd_score = score

        # distance from centre, squared
        # (x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5)
        d = x * x - x + y * y - y + 0.5
        sum_dfc2 += d

        #distance from best corner
        d = sqrt((1.0 - y) * (1.0 - y) + x * x)
        sum_dfb += d
        if d < min_dfb:
            min_dfb = d
            min_dfb_score = score

        # 95% positive and negative
        # intersections with 1:20 lines from the end corners
        if dx == 0 and y > 20.0 * x and not pos_95:
            pos_95 = y
            pos_95_score = score

        if 1.0 - x > 20.0 * (1.0 - y):
            neg_95 = 1.0 - x
            neg_95_score = score

        # briar score
        briar += (score - target) * (score - target)
        error = max(score if target else (1.0 - score), 1e-20)

        cross_entropy -= log(error, 2.0)

    #do the last little bit of area under curve
    dx = 1.0 - px
    dy = 1.0 - py
    auc += px * dy       # bottom rectangle
    auc += dx * dy * 0.5 # top triangle

    briar /= len(results)
    cross_entropy /= len(results)

    # Matthews correlation coefficient/ Phi coefficient at ROC tip
    best_tn = sum_false - best_fp
    best_fn = sum_true - best_tp
    mcc_bottom = ((best_tp + best_fp) *
                  (best_tp + best_fn) *
                  (best_tn + best_fp) *
                  (best_tn + best_fp))
    if mcc_bottom:
        mcc_top = best_tp * best_tn - best_fp * best_fn
        mcc = mcc_top / sqrt(mcc_bottom)
    else:
        mcc = 0

    #f1 = precision * sensitivity / (precision + sensitivity)
    if best_tp:
        best_p = best_tp / float(best_tp + best_fp)
        best_s = best_tp / float(sum_true)
        f1 = best_p * best_s / (best_p + best_s)
    else:
        f1 = 0

    #calculating mean and variance
    mean_data = [[0,0,0], [0,0,0]]
    for score, target in results:
        mean, n, nvar = mean_data[target]
        n += 1
        delta = score - mean
        mean += delta / n
        nvar += delta * (score - mean)
        mean_data[target] = [mean, n, nvar]

    mean_true, n, nvar = mean_data[1]
    if n == 0:
        n = 1.0
    var_true = nvar / n
    mean_false, n, nvar = mean_data[0]
    if n == 0:
        n = 1.0
    var_false = nvar / n
    if var_true + var_false:
        dprime = (mean_true - mean_false) / sqrt(0.5 * (var_true + var_false))
    else:
        #zero variance is in practice a sign of degeneracy
        dprime = 0.0
    sqrt_half =  0.7071067811865475244
    d = {
        'mean_dfd' : sum_dfd / len(results)  * sqrt_half,
        'max_dfd': max_dfd  * sqrt_half,
        'rms_dfc': sqrt(sum_dfc2 / len(results)),
        'mean_dfb': sum_dfb / len(results),
        'min_dfb': min_dfb,
        'auc': auc,
        'dprime': dprime,
        'mcc': mcc,
        'f1': f1,
        'pos_95': pos_95,
        'neg_95': neg_95,
        'briar': briar,
        'cross_entropy': cross_entropy,
    }
    if include_scores:
        d['best_dfd_score'] = best_dfd_score
        d['pos_95_score'] = pos_95_score
        d['neg_95_score'] = neg_95_score
        d['min_dfb_score'] = min_dfb_score

    return d

def calc_stats(results, presence_results=None, presence_gt=None, presence_i=0,
               include_scores=False):
    stats = _calc_stats([x[:2] for x in results], include_scores=include_scores)

    if presence_results is not None:
        p1 = zip([x[presence_i] for x in presence_results], presence_gt)
        presence_stats = _calc_stats(p1, include_scores=include_scores)
        for k, v in presence_stats.iteritems():
            stats['p.' + k] = v

    return stats

def actually_show_roc(title='ROC'):
    import matplotlib.pyplot as plt
    plt.axes().set_aspect('equal')
    plt.title(title, verticalalignment='bottom')
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.legend(loc='lower right')
    plt.show()

def draw_presence_roc(scores, label='presence', label_every=0.0):
    import matplotlib.pyplot as plt
    scores, true_positives, false_positives, \
        tp_scale, fp_scale = prepare_roc_data(scores)
    tp = []
    fp = []
    half = 0
    if label_every:
        step = len(scores) * label_every
    else:
        step =  1e555
    next_label = step
    labels = []

    for i, st in enumerate(scores):
        score, target = st
        false_positives -= not target
        true_positives -= target
        x = false_positives * fp_scale
        y = true_positives * tp_scale
        half += score < 0.5
        if i > next_label:
            labels.append((score, x, y))
            next_label += step
        fp.append(x)
        tp.append(y)

    if half < len(fp):
        hx = (fp[half - 1] + fp[half]) * 0.5
        hy = (tp[half - 1] + tp[half]) * 0.5
    else:
        hx = fp[half - 1]
        hy = tp[half - 1]

    fp.reverse()
    tp.reverse()
    colour = plt.plot(fp, tp, label=label)[0].get_color()
    if label_every:
        for score, x, y in labels:
            plt.annotate("%.2f" % score, xy=(x, y), xytext=(-3, 2), ha='right',
                         textcoords='offset points', color=colour)
