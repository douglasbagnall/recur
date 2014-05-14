# Copyright 2013 Douglas Bagnall <douglas@halo.gen.nz> LGPL/MPL2
import sys
from math import sqrt


def prepare_roc_data(results):
    results.sort()
    sum_true = sum(1 for x in results if x[1])
    sum_false = len(results) - sum_true

    tp_scale = 1.0 / (sum_true or 1)
    fp_scale = 1.0 / (sum_false or 1)
    return results, sum_true, sum_false, tp_scale, fp_scale


def draw_roc_curve(results, label='ROC', arrows=True):
    import matplotlib.pyplot as plt

    results, true_positives, false_positives, \
        tp_scale, fp_scale = prepare_roc_data(results)

    tp = []
    fp = []
    half = 0
    ax, ay, ad, ap = 0, 0, 0, 0
    bx, by, bd, bp = 0, 0, 99, 0
    cx, cy, cd, cp = 0, 0, 0, 0
    dx, dy, dp = 0, 0, 0
    ex, ey, ep = 1.0, 1.0, 0.0

    prev = (None, None)

    for result in results:
        score = result[0]
        target = result[1]
        if prev == (score, target):
            prev
        false_positives -= not target
        true_positives -= target
        x = false_positives * fp_scale
        y = true_positives * tp_scale
        half += score < 0.5
        d = (1 - x) * (1 - x) + y * y
        if d > ad:
            ad = d
            ax = x
            ay = y
            ap = score
        d = x * x + (1 - y) * (1 - y)
        if d < bd:
            bd = d
            bx = x
            by = y
            bp = score
        d = y - x
        if d > cd:
            cd = d
            cx = x
            cy = y
            cp = score
        if dx == 0 and y > 20.0 * x:
            #print x, y
            dx = x
            dy = y
            dp = score
        if 1.0 - x > 20.0 * (1.0 - y):
            #print x, y, (1.0 - y) / (1.0 - x)
            ex = x
            ey = y
            ep = score
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
    print "~best %0.3f  %.3f true, %.3f false" % (cp, cy, cx)
    print "halfway 0.5  %.3f true, %.3f false" % (hy, hx)
    plt.plot(fp, tp, label=label)
    if arrows:
        plt.annotate("95%% negative %.2g" % ep, (ex, ey), (0.7, 0.7),
                     arrowprops={'width':1, 'color': '#0088aa'},
                     )
        plt.annotate("95%% positive %.2g" % dp, (dx, dy), (0.2, 0.2),
                     arrowprops={'width':1, 'color': '#8800aa'},
                     )
        plt.annotate("0.5", (hx, hy), (0.4, 0.4),
                     arrowprops={'width':1, 'color': '#00cc00'})
        plt.annotate("furthest from all bad %.2g" % ap, (ax, ay), (0.3, 0.3),
                     arrowprops={'width':1, 'color': '#00cccc'},
                     )
        plt.annotate("closest to all good %.2g" % bp, (bx, by), (0.6, 0.6),
                     arrowprops={'width':1, 'color': '#cc0000'},
                     )
        plt.annotate("furthest from diagonal %.2g" % cp, (cx, cy), (0.5, 0.5),
                     arrowprops={'width':1, 'color': '#aa6600'},
                     )

def _calc_stats(results):
    from math import sqrt, log
    (results, sum_true, sum_false,
     tp_scale, fp_scale) = prepare_roc_data(results)
    auc = 0
    sum_dfd = 0 #distance from diagonal (but signed)
    max_dfd = 0
    sum_dfc2 = 0 #distance from centre, squared
    max_dfc2 = 0
    sum_dfb, min_dfb = 0, 1e99 #distance from best
    pos_95 = 0
    neg_95 = 0
    briar = 0
    cross_entropy = 0

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

        # distance from centre, squared
        # (x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5)
        d = x * x - x + y * y - y + 0.5
        sum_dfc2 += d

        #distance from best corner
        d = sqrt((1.0 - y) * (1.0 - y) + x * x)
        sum_dfb += d
        if d < min_dfb:
            min_dfb = d

        # 95% positive and negative
        # intersections with 1:20 lines from the end corners
        if dx == 0 and y > 20.0 * x and not pos_95:
            pos_95 = y

        if 1.0 - x > 20.0 * (1.0 - y):
            neg_95 = 1.0 - x

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
    var_true = nvar / n
    mean_false, n, nvar = mean_data[0]
    var_false = nvar / n
    if var_true + var_false:
        dprime = (mean_true - mean_false) / sqrt(0.5 * (var_true + var_false))
    else:
        #zero variance is in practice a sign of degeneracy
        dprime = 0.0
    sqrt_half =  0.7071067811865475244
    return {
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


def calc_stats(results, presence_results, presence_gt, presence_i=0):
    instantaneous_stats = _calc_stats([x[:2] for x in results])
    p1 = zip([x[presence_i] for x in presence_results], presence_gt)
    presence_stats = _calc_stats(p1)

    stats = instantaneous_stats
    for k, v in presence_stats.iteritems():
        stats['p.' + k] = v

    return stats

def actually_show_roc(title='ROC'):
    import matplotlib.pyplot as plt
    plt.axes().set_aspect('equal')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.show()

def draw_presence_roc(scores, label='presence', label_every=0.0):
    import matplotlib.pyplot as plt
    #print scores

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
    plt.plot(fp, tp, label=label)
    if label_every:
        for score, x, y in labels:
            plt.annotate("%.2f" % score, xy=(x, y), xytext=(-5, 5), ha='right',
                         textcoords='offset points')