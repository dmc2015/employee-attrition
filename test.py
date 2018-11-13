k = list(range(1, 101))
params = {'n_neighbors': k}

kf = cross_validation.KFold(len(attrition_df), n_folds=5)
gs = grid_search(
    estimator = neighbors.KNeighborsClassifier(),
    param_grid=params,
    cv=kf
)


# gs = GridSearchCV(
#     estimator = neighbors.KNeighborsClassifier(),
#     param_grid=params,
#     cv=kf,
#     return_train_score=True
# )


# gs = GridSearchCV(
#     estimator = neighbors.KNeighborsClassifier(),
#     param_grid=params,
#     cv=kf
# )

# print(
# gs.fit(X, y), 
# gs.grid_scores_, 
# gs.cv_results_attribute
# )


# lowest_std = []
# highest_mean = []
# for values in gs.grid_scores_:
# #     print(values[1])
# #     print(type(values[1]))
# #     print(lowest_std[0][1])
# #     print(lowest_std[0])
#     if len(highest_mean) == 0:
#         highest_mean.append(values)
#     else:# values[1] > lowest_std[0][1]:
#         highest_mean.pop()
#         highest_mean.append(values)
        
#     if len(lowest_std) == 0:
#         lowest_std.append(values)
#     else:# values[1] > lowest_std[0][1]:
#         lowest_std.pop()
#         lowest_std.append(values)