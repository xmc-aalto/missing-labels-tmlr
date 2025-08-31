@everywhere include("psrec.jl")

@everywhere using SharedArrays
using NPZ

# generate artificial data
# q: Marginal probability for each label
# l: Number of labels
# p: propensity
# n: Number of data points
@everywhere function fake_data(q::Float64, l::Int, p::Float64, n::Int)
    ground_truth = rand(Float64, (n, l)) .< q
    observation = ground_truth .* (rand(Float64, (n, l)) .< p)
    ground_truth = [get_nz(ground_truth[i, :]) for i in 1:n]
    prediction = [make_pred(ground_truth[i]) for i in 1:n]
    propensity = ones(l) .* p
    return  ground_truth, [get_nz(observation[i, :]) for i in 1:n], prediction, propensity
end

@everywhere function get_nz(gt)
    return [i for i in 1:length(gt) if gt[i]]
end

@everywhere function make_pred(gt)
    if length(gt) == 0
        return []
    else
        return [rand(gt)]
    end
end

@everywhere function run_test(q::Float64, l::Int, p::Float64, n::Int)
    gt, obs, pred, prop = fake_data(q, l, p, n)
    ps_vals = []
    gt_vals = []
    bs_vals = []
    vn_vals = []
    for i in 1:length(gt)
        r_ps = ps_recall(pred[i], obs[i], prop)
        r_gt = recall(pred[i], gt[i])
        r_bs = biased_recall(pred[i], obs[i], prop)
        r_vn = recall(pred[i], obs[i])
        append!(ps_vals, r_ps)
        append!(gt_vals, r_gt)
        append!(bs_vals, r_bs)
        append!(vn_vals, r_vn)
        #@show r_bs, r_vn
        #exit()
    end

    return gt_vals, ps_vals, bs_vals, vn_vals
end

@everywhere function gather_stats(q::Float64, l::Int, p::Float64, n::Int, m::Int)
    ps_vals = []
    vn_vals = []
    bs_vals = []
    gt_vals = []
    for k in 1:m
        gt, ps, bs, vn = run_test(q, l, p, n)
        append!(ps_vals, mean(ps))
        append!(vn_vals, mean(vn))
        append!(bs_vals, mean(bs))
        append!(gt_vals, mean(gt))
    end
    return gt_vals, ps_vals, bs_vals, vn_vals
end


function sweep(q::Float64, l::Int, p_values::Array{Float64}, n::Int, m::Int)
    ps_res = SharedArray{Float64}((length(p_values), m))
    vn_res = SharedArray{Float64}((length(p_values), m))
    bs_res = SharedArray{Float64}((length(p_values), m))
    gt_res = SharedArray{Float64}((length(p_values), m))
    @sync @distributed for i = 1:length(p_values)
        p = p_values[i]
        @show p
        gt_vals, ps_vals, bs_vals, vn_vals = gather_stats(q, l, p, n, m)
        ps_res[i, :] = ps_vals
        vn_res[i, :] = vn_vals 
        bs_res[i, :] = bs_vals 
        gt_res[i, :] = gt_vals 
    end
    return gt_res, ps_res, bs_res, vn_res
end


p_values = [0.9, 0.8, 0.7, 0.65, 0.6, 0.59, 0.58, 0.57, 0.56, 0.55, 0.54, 0.53, 0.52, 0.51, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2]
gt, ps, bs, vn = sweep(0.1, 100, p_values, 10000, 1000)
#ps, vn = sweep(0.1, 100, p_values, 1000, 10)
npzwrite("/home/erik/Documents/latex/pmlr/data/artificial-10k.npz"; ps=ps, vn=vn, bs=bs, gt=gt, p=p_values)
