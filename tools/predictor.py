from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np


class Undersampler:
	def __init__(self, n_subsets):
		self.n_subsets = n_subsets

	def fit_sample(self, X, y):
		class_0_ids = np.where(y == 0)[0]
		class_1_ids = np.where(y == 1)[0]

		num_0 = len(class_0_ids)
		num_1 = len(class_1_ids)

		subsample_size = num_0 / self.n_subsets

		X_subs = []
		y_subs = []

		X_1 = X[class_1_ids]
		X_0 = X[class_0_ids]

		y_1 = np.ones(num_1)

		for _ in range(self.n_subsets):
			X_sub_0 = X_0[np.random.choice(X_0.shape[0], size = subsample_size, replace=False), :]
			y_sub_0 = np.zeros(subsample_size)

			X_sub = np.concatenate((X_sub_0, X_1), axis=0)
			y_sub = np.concatenate((y_sub_0, y_1), axis=0)

			X_subs.append(X_sub)
			y_subs.append(y_sub)

		return X_subs, y_subs



class EnsembleXGB:
	def __init__(self, num_estimators, **params):
		self.num_est = num_estimators
		self.estimators_ = list(XGBClassifier(**params) for _ in range(num_estimators))
		self.us_ = Undersampler(n_subsets = num_estimators)


	def fit(self, X, y):
		eex, eey = self.us_.fit_sample(X, y)
		for sub_x, sub_y, clf in zip(eex, eey, self.estimators_):
			clf.fit(sub_x, sub_y)
		return self


	def predict_proba_all(self, X):
		probas = []
		for clf in self.estimators_:
			y = clf.predict_proba(X)
			probas.append(y)
		return np.stack(probas, axis=0)


	def predict_proba(self, X):
		return np.mean(self.predict_proba_all(X), axis=0)



class EnsembleXGBLogReg:
	def __init__(self, num_estimators, **params):
		self.num_est = num_estimators
		self.clf_ = EnsembleXGB(num_estimators, **params)
		# self.logreg_ = LogisticRegression()
		self.logreg_ = XGBClassifier()


	def fit(self, X, y):
		cv = StratifiedKFold(n_splits=2)
		probas = np.zeros((X.shape[0], self.num_est))

		for train_i, test_i in cv.split(X, y):
			probas[test_i] = (self.clf_
				.fit(X[train_i], y[train_i])
				.predict_proba_all(X[test_i])[:,:,1].T)

		self.logreg_.fit(probas, y)
		return self


	def predict_proba(self, X):
		probas = self.clf_.predict_proba_all(X)[:,:,1].T
		return self.logreg_.predict_proba(probas)