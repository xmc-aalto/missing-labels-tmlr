using Combinatorics
using Statistics


#=
    Calculates vanilla recall given the predictions and ground_truth
    as integer arrays. If the ground_truth is empty, a value of 0 is 
    returned.
=#
function recall(predictions::Array, ground_truth::Array)
    Iy = ground_truth
    Iyhat = predictions
    S = intersect(Iy, Iyhat)
    # the max(1, ...) term prevents division by zero. If ground_truth is empty, then length(S) == 0 and the result is still zero,
    # and if ground_truth is not empty, then the max does not change the result.
    return float(length(S)) / max(1, length(ground_truth))
end

function biased_recall(predictions::Array, observation::Array, propensities::Array)
    Iy = observation
    Iyhat = predictions

    S = intersect(Iy, Iyhat)
    if length(S) == 0
        return 0
    end
    all = length(observation)
    return float(sum(1.0 / (all - 1 + propensities[i]) for i in S))
end 

#=
    Calculates the propensity-scored recall of a single multi-label instance with observed labels
    `observation` and predicted labels `predictions` given as integer arrays. The propensities
    are given as a float array, such that the propensity of the i'th observed labels is given
    by `p_i = propensities[observation[i]]`, i.e. the entries in `observation` are expected
    to be in the interval `[1, length(propensities)]`.
=#
function direct_ps_recall(predictions::Array, observation::Array, propensities::Array)
    # rename as in the paper text 
    Iy = observation
    Iyhat = predictions
    p = propensities

    # this is the set of correctly predicted labels
    S = intersect(Iy, Iyhat)

    # first, we calculate the prefactor common to all summands
    c = 1
    for i in Iy
        c *= (p[i] - 1.0) / p[i]
    end

    # this is the set of unpredicted but observed labels
    T = setdiff(Iy, S)

    # given a set `U`, this helper function calculates the product
    # `\Prod_{u ∈ U} (p[u] - 1)

    function d(U)
        product = 1.0
        for k in U
            product *= (p[k] - 1)
        end
        return product
    end

    summation = 0.0

    # the outer sum of equation 19: sum over all possible sizes of subsets of correctly predicted labels
    for s in 1:length(S)
        ps = 1.0 / s        
        # combinations(T) skips the empty set, so we add this here explicitly as the 1.0 / s term
        for U in combinations(T)
            ps += 1.0 / ((length(U) + s) * d(U))
        end

        inner_sum = 0
        # the sum over all subsets of S of size s
        for K in combinations(S, s)   
            inner_sum += s / d(K)
        end

        summation += inner_sum * ps
    end

    return c*summation
end


#=
This function also calculates PS Recall, but instead of doing the sum over all subsets, 
which can become computationally too expensive, so instead this function artificially
removes some more ground-truth labels, but compensates this into an unbiased estimate
by adapting the propensities. Since this can cause a large increase in variance, we
repeat the processs 100 times and return the average.
=#
function sampled_ps_recall(predictions::Array, ground_truth::Array, propensities::Array)
    summed = 0.0
    ss_p  = copy(propensities) ./ 2
    for i in 1:100
        # subsamples each label with a chance of 50%
        ss_gt = [g for g in ground_truth if rand() < 0.5]
        summed += ps_recall(predictions, ss_gt, ss_p)
    end

    return summed / 100
end


# This funtion calculates the propensity scored recall, using either the exact
# calculation if the number of ground_truth labels is low (`direct_ps_recall`), 
# or a stochastic approximation (`sampled_ps_recall`)
function ps_recall(predictions::Array, ground_truth::Array, propensities::Array)
    if length(ground_truth) < 26
        return direct_ps_recall(predictions, ground_truth, propensities)
    else
        return sampled_ps_recall(predictions, ground_truth, propensities)
    end
end