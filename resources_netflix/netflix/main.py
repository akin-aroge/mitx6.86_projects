import numpy as np
import kmeans
import common
import naive_em
import em

#X = np.loadtxt("toy_data.txt")
X = np.loadtxt("netflix_incomplete.txt")
X_gold = np.loadtxt('netflix_complete.txt')

##====================================
# Kmeans
# TODO: Your code here
# n_clusters = np.array([1,2,3,4])
# seeds = np.array([0, 1, 2, 3, 4])
# costs = np.empty(seeds.shape[0])


# for n_cluster in n_clusters:
#     costs = np.empty(seeds.shape[0])
#     mixtures = []
#     posts = []

#     for i, seed in enumerate(seeds):

#         mixture, post = common.init(X, n_cluster, seed)
#         mixture, post, cost = kmeans.run(X, mixture, post)
#         costs[i] = cost
#         mixtures.append(mixture)
#         posts.append(post)

#     idx_min_seed = np.argmin(costs)
#     common.plot(X, mixtures[idx_min_seed], posts[idx_min_seed], str(n_cluster))    
#     print(costs[idx_min_seed])

##=============================================
# running naive_em

# n_clusters = np.array([1,2,3,4])
# seeds = np.array([0, 1, 2, 3, 4]) 

# for n_cluster in n_clusters:
#     log_lhs = np.empty(seeds.shape[0])
#     mixtures = []
#     posts = []

#     for i, seed in enumerate(seeds):

#         mixture, post = common.init(X, n_cluster, seed)
#         mixture, post, log_lh = naive_em.run(X, mixture, post)
#         log_lhs[i] = log_lh
#         mixtures.append(mixture)
#         posts.append(post)

#     idx_min_seed = np.argmax(log_lhs) # max becaus it is loglikelihoood
#     common.plot(X, mixtures[idx_min_seed], posts[idx_min_seed], str(n_cluster))    
#     print(log_lhs[idx_min_seed])

##===========================================
# Using BIC to select best K

# n_clusters = np.array([1,2,3,4])
# seeds = np.array([0, 1, 2, 3, 4]) 

# bics = np.empty(n_clusters.shape[0])

# for i, n_cluster in enumerate(n_clusters):
#     costs = np.empty(seeds.shape[0])
#     mixtures = []
#     posts = []

#     for j, seed in enumerate(seeds):

#         mixture, post = common.init(X, n_cluster, seed)
#         mixture, post, cost = naive_em.run(X, mixture, post)
#         costs[j] = cost
#         mixtures.append(mixture)
#         posts.append(post)

#     idx_min_seed = np.argmax(costs)
#     common.plot(X, mixtures[idx_min_seed], posts[idx_min_seed], str(n_cluster))    
    
#     # collect log_likeihoods
#     lh = costs[idx_min_seed]
    
#     bics[i] = common.bic(X, mixture, lh)

#     print(costs[idx_min_seed])
# print(np.max(bics))
# print(n_clusters[np.argmax(bics)])

##=============================================
# running em

n_clusters = np.array([12])
seeds = np.array([0, 1, 2, 3, 4]) 

for n_cluster in n_clusters:
    log_lhs = np.empty(seeds.shape[0])
    mixtures = []
    posts = []

    for i, seed in enumerate(seeds):

        mixture, post = common.init(X, n_cluster, seed)
        mixture, post, log_lh = em.run(X, mixture, post)
        log_lhs[i] = log_lh
        mixtures.append(mixture)
        posts.append(post)

    idx_max_seed = np.argmax(log_lhs) # max becaus it is loglikelihoood
    #common.plot(X, mixtures[idx_min_seed], posts[idx_min_seed], str(n_cluster))    
    print(log_lhs[idx_max_seed])

best_mixture = mixtures[idx_max_seed]

X_pred = em.fill_matrix(X, best_mixture)

print(common.rmse(X_gold, X_pred))

