

class AttentionAlgorithms(object):
    Ec = "Ec_from_Pope1995"
    Proportion_of_Beta = "proportion_of_beta"
    Beta_over_Alpha = "beta_over_alpha"

    XY_RATIO = "XY_RATIO"

    Delta = (0.5, 4)
    Theta = (4, 7)
    Alpha = (7, 13)
    Beta = (13, 30)

    # delta(0.5-4), theta(4-7), alpha(7-13), beta(13-30)
    # 0             1           2           3
    @staticmethod
    def compute_ec(res):
        # return res[3] / (res[2] + res[1])
        return res[2] / (res[1] + res[2] + res[3])

    @staticmethod
    def compute_proportion_of_beta(res):
        return res[3] / (res[1] + res[2] + res[3])

    @staticmethod
    def compute_beta_over_alpha(res):
        return res[3] / res[2]

    @staticmethod
    def compute_xy_ratio(res):
        x = (res[2] + res[3]) / sum(res)  # alpha(7-13), beta(13-30)  attention index
        y = (res[0] + res[1]) / sum(res)  # delta(0.5-4), theta(4-7)  relaxation index
        return (x/y)*(x-y)  # d index = (x/y)(x-y)

    # @staticmethod
    # def smoothed_d_value(d_features_li, new_d_value, max_length=2):
    #     if len(d_features_li) >= max_length:
    #         d_features_li.pop(0)
    #     if d_features_li:
    #         last_smoothed_d = d_features_li[-1]
    #         new_smoothed_d = max(last_smoothed_d, new_d_value) * new_d_value
    #     else:
    #         new_smoothed_d = new_d_value
    #
    #     d_features_li.append(new_smoothed_d)
    #
    #     return d_features_li