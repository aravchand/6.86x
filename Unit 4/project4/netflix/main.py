import numpy as np
import kmeans
import common
import naive_em
import em

np.set_printoptions(suppress=True)


X = np.loadtxt("toy_data.txt")
# print(X.shape)
# print(X[:5])


# TODO: Your code here

# tab 2
"""
save_seed = {}
for K in [1, 2, 3, 4]: 
    best_seed, min_cost = None, None
    best_mixture, best_post = None, None
    for seed in [0, 1, 2, 3, 4]:
        mixture, post = common.init(X, K, seed)
        mixture, post, cost = kmeans.run(X, mixture, post)
        if min_cost is None or cost < min_cost:
            min_cost = cost
            best_seed = seed
            best_mixture = mixture
            best_post = post

    print(min_cost, K, best_seed)
    save_seed[K] = (best_seed, best_mixture, best_post, min_cost)

    # save plots to files
    common.plot(X, best_mixture, best_post, f"K={K}, seed={best_seed}",
                save=True, save_path=f"kmeans_plots/k{K}_seed{best_seed}.png")
"""

# tab 3
"""n=3
d=2
K=2

X = np.array([[1, 2], [3, 4], [5, 6]])


X = np.array([0.2, -0.9, -1, 1.2, 1.8]).reshape(5, 1)
mu1 = -3 
mu2 = 2 
v1 = 4 
v2 = 4 
p1 = 0.5 
p2 = 0.5 


mixture = common.GaussianMixture(np.array([[mu1], [mu2]]), np.array([v1, v2]), np.array([p1, p2]))
# print(f"The mixture is {mixture}")
"""

"""
save_seed = {}
for K in [1, 2, 3, 4]: 
    best_seed, max_ll = None, None
    best_mixture, best_post = None, None
    for seed in [0, 1, 2, 3, 4]:
        mixture, post = common.init(X, K, seed)
        mixture, post, ll = naive_em.run(X, mixture, post)
        if max_ll is None or ll < max_ll:
            max_ll = ll
            best_seed = seed
            best_mixture = mixture
            best_post = post

    print(f"K = {K}, with best log likelihood of {max_ll}, maximized w/ seed {best_seed}")
    # save_seed[K] = (best_seed, best_mixture, best_post, max_ll)

    # save plots to files
    # common.plot(X, best_mixture, best_post, f"K={K}, seed={best_seed}",
    #             save=True, save_path=f"naive_em_plots/k{K}_seed{best_seed}.png")
"""

# tab 5
"""
best_bic = None
for K in [1, 2, 3, 4]:
    mixture, post = common.init(X, K)
    mixture, post, ll = naive_em.run(X, mixture, post)
    bic = common.bic(X, mixture, ll)
    print(f"K={K}'s BIC value: {bic}")
    if best_bic is None or best_bic < bic:
        best_bic = bic
        best_mixture = mixture
        best_post = post
        best_K = K
print(f"\nBest BIC value: {best_bic}")
print(f"Best K: {best_K}")
# print(f"Best mixture: {best_mixture}")
# print(f"Best post: {best_post}")
"""


# tab 8
X = np.loadtxt("netflix_incomplete.txt")

K=12
seed=1
mixture, post = common.init(X, K, seed)
mixture, post, ll = em.run(X, mixture, post)
        
print(f"done with K={K}, seed={seed}, ll={ll}")
# save plots to files
# common.plot(X, mixture, post, f"K={K}, seed={seed} with ll={ll}",
#     save=True, save_path=f"em_plots/k{K}_seed{seed}.png")

# save results to files
np.savetxt(f"em_results/k{K}_seed{seed}.txt", mixture)
np.savetxt(f"em_results/k{K}_seed{seed}_post.txt", post)
# np.savetxt(f"em_results/k{K}_seed{best_seed}_ll.txt", [maxll])


X_gold = np.loadtxt('netflix_complete.txt')
X_pred = em.fill_matrix(X, mixture)
rmse = common.rmse(X_gold, X_pred)
print(f"RMSE: {rmse}")

# save pred to files
np.savetxt(f"em_results/k{K}_seed{seed}_pred.txt", X_pred)