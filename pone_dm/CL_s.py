# For Likelihood analysis
# Required functions

from tqdm import tqdm
import numpy as np
from scipy.stats import poisson


def q_single_s_b(s, b_sim, b_obs):
    return poisson.pmf(b_obs, b_sim + s) / poisson.pmf(b_obs, b_sim)


def q_single_b(b_sim, b_obs):
    return poisson.pmf(b_obs, b_sim) / poisson.pmf(b_obs, b_obs)


def q_multi_s_b(signal, background, measured):
    return np.prod(q_single_s_b(signal, background, measured))


def q_multi_b(background, measured):
    return np.prod(q_single_b(background, measured))

# Sample measured (pseudo) NOT the expectation


def sampling_b_1d(signal, background, sample_count=1000):
    q_sampled = []
    expectation = background
    for i in range(sample_count):
        measured = np.random.poisson(expectation)
        q_sampled.append(q_multi_s_b(signal, background, measured))
    q_sampled = np.array(q_sampled)
    return q_sampled


def sampling_b_1d_v2(background, sample_count=1000):
    q_sampled = []
    expectation = background
    for i in range(sample_count):
        measured = np.random.poisson(expectation)
        q_sampled.append(q_multi_b(background, measured))
    q_sampled = np.array(q_sampled)
    return q_sampled


def sampling_s_b_1d(signal, background, sample_count=1000):
    q_sampled = []
    expectation = signal + background
    for i in range(sample_count):
        measured = np.random.poisson(expectation)
        q_sampled.append(q_multi_s_b(signal, background, measured))
    q_sampled = np.array(q_sampled)
    return q_sampled


def CL_scan(signal_grid, background, observed, sv_range, mass_range,
            sample_count=1000):
    CL_mat = []
    prob_mat = []
    tmp_prob_b = []
    
    for i_s, sv in tqdm(enumerate(sv_range)):
        tmp_p_b = []
        CL_s = []
        prob_array = []
        for i_m, mass in enumerate(mass_range):

            # CL_s_b
            signal = signal_grid[i_s][i_m]
            # print("background +sim ")
            # print(background + signal)
            # print("observed")
            # print(observed)
            # print("q_single_b")
            # print(q_single_b(background, observed))
            # print("q_signle_s_b")
            # print(q_single_s_b(signal, background, observed))

            q_obs_s_b = q_multi_s_b(
                signal,
                background,
                observed
            )

            # s + b
            q_pdf_s_b_lower = sampling_s_b_1d(signal, background,
                                              sample_count=sample_count)
            # print("q_pdf_s_b_lower")
            # print(q_pdf_s_b_lower)

            q_pdf_s_b_upper = q_pdf_s_b_lower[q_pdf_s_b_lower < q_obs_s_b]
            # print("q_pdf_s_b_upper")
            # print(q_pdf_s_b_upper)

            p_s_b = float(len(q_pdf_s_b_upper)) / float(len(q_pdf_s_b_lower))
            # print("p_s_b")
            # print(p_s_b)

            # b
            q_pdf_b_lower = sampling_b_1d(signal, background,
                                          sample_count=sample_count)
            # print("q_pdf_b_lower")
            # print(q_pdf_b_lower)

            q_pdf_b_upper = q_pdf_b_lower[q_pdf_b_lower < q_obs_s_b]
            # print("q_pdf_b_upper")
            # print(q_pdf_b_upper)

            p_b = float(len(q_pdf_b_upper)) / float(len(q_pdf_b_lower))

            # Fraction
            try:
                CL_s.append(p_s_b / (1. - p_b))
            except:
                CL_s.append(0.)
            prob_array.append(p_s_b)
            tmp_p_b.append(p_b)
        
        CL_mat.append(CL_s)
        prob_mat.append(prob_array)
        tmp_prob_b.append(tmp_p_b)
    print('p_b')
    print(tmp_prob_b)
    print('1 - p_b')
    print(1 - np.array(tmp_prob_b))
    return CL_mat, prob_mat

