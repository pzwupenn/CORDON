import argparse
import random
import time
import os

import sys
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from mscn.util import *
from mscn.data import get_train_datasets, make_dataset, ori_load_data
from mscn.model import SetConv
import mscn.get_bitmap_imdb as get_bitmap

rs = np.random.RandomState(42)

### import CORDON lib

sys.path.append('./CORDON')

from eligibility_checker.IMDb import check_eligible_add_pk
from eligibility_checker.IMDb import check_eligible_drop_pk
from eligibility_checker.IMDb import check_eligible_drop_pk_no_pred
from eligibility_checker.IMDb import check_eligible_add_filter

from query_augmentor.IMDb import add_pk
from query_augmentor.IMDb import drop_pk_no_pred

from loss_generator.IMDb import drop_pk
from loss_generator.IMDb import expand_query_add_filter
from loss_generator.IMDb import expand_k_query_add_filter

### complete importing CORDON lib

def unnormalize_torch(vals, min_val, max_val):
	vals = (vals * (max_val - min_val)) + min_val
	return torch.exp(vals)


def normalize_torch(vals, min_val, max_val):
	vals = torch.log(vals)
	labels_norm = (vals - min_val) / (max_val - min_val)
	return labels_norm


def qerror_loss(preds, targets, min_val, max_val):
	qerror = []
	preds = unnormalize_torch(preds, min_val, max_val)
	targets = unnormalize_torch(targets, min_val, max_val)

	for i in range(len(targets)):
		if (preds[i] > targets[i]).cpu().data.numpy()[0]:
			qerror.append(preds[i] / targets[i])
		else:
			qerror.append(targets[i] / preds[i])
	return torch.mean(torch.cat(qerror))


def predict(model, data_loader, cuda):
	preds = []
	t_total = 0.

	model.eval()
	for batch_idx, data_batch in enumerate(data_loader):

		samples, predicates, joins, targets, sample_masks, predicate_masks, join_masks, train_ids = data_batch
		if cuda:
			samples, predicates, joins, targets = samples.cuda(), predicates.cuda(), joins.cuda(), targets.cuda()
			sample_masks, predicate_masks, join_masks = sample_masks.cuda(), predicate_masks.cuda(), join_masks.cuda()
		samples, predicates, joins, targets = Variable(samples), Variable(predicates), Variable(joins), Variable(
			targets)
		sample_masks, predicate_masks, join_masks = Variable(sample_masks), Variable(predicate_masks), Variable(
			join_masks)

		t = time.time()
		outputs = model(samples, predicates, joins, sample_masks, predicate_masks, join_masks)
		t_total += time.time() - t

		for i in range(outputs.data.shape[0]):
			preds.append(outputs.data[i])

	return preds, t_total


def predict_and_get_labels(model, data_loader, cuda):
	preds = []
	labels = []

	model.eval()
	for batch_idx, data_batch in enumerate(data_loader):

		samples, predicates, joins, targets, sample_masks, predicate_masks, join_masks, train_ids = data_batch
		if cuda:
			samples, predicates, joins, targets = samples.cuda(), predicates.cuda(), joins.cuda(), targets.cuda()
			sample_masks, predicate_masks, join_masks = sample_masks.cuda(), predicate_masks.cuda(), join_masks.cuda()
		samples, predicates, joins, targets = Variable(samples), Variable(predicates), Variable(joins), Variable(
			targets)
		sample_masks, predicate_masks, join_masks = Variable(sample_masks), Variable(predicate_masks), Variable(
			join_masks)
		outputs = model(samples, predicates, joins, sample_masks, predicate_masks, join_masks)

		for i in range(outputs.data.shape[0]):
			preds.append(outputs.data[i])
			labels.append(targets.data[i])

	return preds, labels


def print_qerror(preds_unnorm, labels_unnorm):
	qerror_res = []

	preds_unnorm = np.squeeze(preds_unnorm)
	for i in range(len(preds_unnorm)):

		if preds_unnorm[i] > float(labels_unnorm[i]):
			qerror_res.append(preds_unnorm[i] / float(labels_unnorm[i]))
		else:
			qerror_res.append(float(labels_unnorm[i]) / float(preds_unnorm[i]))

	print('overall')
	print("num of queries {}".format(len(preds_unnorm)))
	print("Median: {}".format(np.median(qerror_res)))
	print("90th percentile: {}".format(np.percentile(qerror_res, 90)))
	print("95th percentile: {}".format(np.percentile(qerror_res, 95)))
	print("99th percentile: {}".format(np.percentile(qerror_res, 99)))
	print("Max: {}".format(np.max(qerror_res)))
	print("Mean: {}".format(np.mean(qerror_res)))

	return qerror_res

def check_duplicate(vector, v_list):
	vector = tuple(vector)
	res = vector in v_list
	return res

def train_and_predict(workload_name, num_queries, num_epochs, batch_size, hid_units, cuda, table_to_df, burn_in=10):
	# Load training and validation data
	num_materialized_samples = 1000
	weight = 1

	dicts, column_min_max_vals, min_val, max_val, labels_train, labels_validation, max_num_joins, max_num_predicates, train_data, test_data, candi_data, \
	ori_predicates_train, ori_samples_train, ori_tables_train, ori_predicates_test, ori_samples_test, ori_tables_test, num_joins_train, num_predicates_train, table_sets_train, num_joins_test, num_predicates_test, table_sets_test, \
	numerical_cols, candi_query_typeids, candi_joins, candi_predicates, candi_tables, candi_samples = get_train_datasets(
		num_queries, num_materialized_samples)

	table2vec, column2vec, op2vec, join2vec = dicts

	# Train model
	sample_feats = len(table2vec) + num_materialized_samples
	predicate_feats = len(column2vec) + len(op2vec) + 1
	join_feats = len(join2vec)

	model = SetConv(sample_feats, predicate_feats, join_feats, hid_units)

	optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

	if cuda:
		model.cuda()

	train_data_loader = DataLoader(train_data, batch_size=batch_size)
	candi_data_loader = DataLoader(candi_data, batch_size=batch_size)

	model.train()

	k = 5  # number of perturbations!

	### get all augmented queries ###
	existing_aug_queries = set([])
	queries2labels = {}
	aug_samples, aug_predicates, aug_joins, aug_targets, aug_sample_masks, aug_predicate_masks, aug_join_masks = [],[],[],[],[],[],[]

	for batch_idx, data_batch in enumerate(train_data_loader):
		samples, predicates, joins, targets, sample_masks, predicate_masks, join_masks, train_ids = data_batch
		batch_q_features = model.get_features(samples, predicates, joins, sample_masks, predicate_masks, join_masks)
		for q_id, q_f in enumerate(batch_q_features.numpy()):
			if not check_duplicate(q_f, existing_aug_queries):
				existing_aug_queries.add(tuple(q_f))
				queries2labels[tuple(q_f)] = targets[q_id]

	for batch_idx, data_batch in enumerate(train_data_loader):
		samples, predicates, joins, targets, sample_masks, predicate_masks, join_masks, train_ids = data_batch

		train_ids = train_ids.numpy()
		batch_predicates = [ori_predicates_train[x] for x in train_ids]
		batch_tables = [ori_tables_train[x] for x in train_ids]
		batch_samples = [ori_samples_train[x] for x in train_ids]


		has_pred_on_title_list = []
		for ori_predicate in batch_predicates:
			has_pred_on_title = False
			for a_predicate in ori_predicate:
				if a_predicate[0].split('.')[0] == 't':
					has_pred_on_title = True
					break
			has_pred_on_title_list.append(has_pred_on_title)

		add_pk_list = check_eligible_add_pk(predicates, batch_tables)
		drop_pk_total_list = check_eligible_drop_pk_no_pred(has_pred_on_title_list, batch_tables)

		subq1_list, sample1_list, join_list, subq1_mask_list, sample1_mask_list, join_mask_list, indicators = add_pk(
			add_pk_list, join2vec, max_num_joins, join_feats,
			predicates, samples, table2vec, batch_tables, batch_samples)

		subq1_list = torch.from_numpy(np.array(subq1_list, dtype='float32'))
		subq1_mask_list = torch.from_numpy(np.array(subq1_mask_list, dtype='float32'))
		sample1_list = torch.from_numpy(np.array(sample1_list, dtype='float32'))
		sample1_mask_list = torch.from_numpy(np.array(sample1_mask_list, dtype='float32'))
		join_list = torch.from_numpy(np.array(join_list, dtype='float32'))
		join_mask_list = torch.from_numpy(np.array(join_mask_list, dtype='float32'))
		indicators = torch.from_numpy(np.array(indicators, dtype='float32'))

		if cuda:
			subq1_list, subq1_mask_list, sample1_list, sample1_mask_list, join_list, join_mask_list, indicators = \
				subq1_list.cuda(), subq1_mask_list.cuda(), sample1_list.cuda(), sample1_mask_list.cuda(), join_list.cuda(), join_mask_list.cuda(), indicators.cuda()

		q_features = model.get_features(sample1_list, subq1_list, join_list, sample1_mask_list, subq1_mask_list,
		                               join_mask_list).numpy()

		for qid, q_feature in enumerate(q_features):
			if not check_duplicate(q_feature, existing_aug_queries):
				existing_aug_queries.add(tuple(q_feature))
				queries2labels[tuple(q_feature)] = targets[qid]


		### drop pk totally ###
		subq1_list, join_list, sample1_list, subq1_mask_list, join_mask_list, sample1_mask_list, indicators = drop_pk_no_pred(
			drop_pk_total_list, predicates, samples, table2vec, column2vec, join2vec,
			max_num_joins,
			join_feats,
			batch_tables, batch_samples)

		subq1_list = torch.from_numpy(np.array(subq1_list, dtype='float32'))
		subq1_mask_list = torch.from_numpy(np.array(subq1_mask_list, dtype='float32'))
		sample1_list = torch.from_numpy(np.array(sample1_list, dtype='float32'))
		sample1_mask_list = torch.from_numpy(np.array(sample1_mask_list, dtype='float32'))
		join_list = torch.from_numpy(np.array(join_list, dtype='float32'))
		join_mask_list = torch.from_numpy(np.array(join_mask_list, dtype='float32'))
		indicators = torch.from_numpy(np.array(indicators, dtype='float32'))

		if cuda:
			subq1_list, subq1_mask_list, sample1_list, sample1_mask_list, join_list, join_mask_list, indicators = \
				subq1_list.cuda(), subq1_mask_list.cuda(), sample1_list.cuda(), sample1_mask_list.cuda(), join_list.cuda(), join_mask_list.cuda(), indicators.cuda()

		q_features = model.get_features(sample1_list, subq1_list, join_list, sample1_mask_list, subq1_mask_list,
		                               join_mask_list).numpy()

		for qid, q_feature in enumerate(q_features):
			if not check_duplicate(q_feature, existing_aug_queries):

				aug_samples.append(sample1_list[qid, :, :])
				aug_predicates.append(subq1_list[qid, :, :])
				aug_joins.append(join_list[qid, :, :])
				aug_sample_masks.append(sample1_mask_list[qid, :, :])
				aug_predicate_masks.append(subq1_mask_list[qid, :, :])
				aug_join_masks.append(join_mask_list[qid, :, :])

				aug_targets.append(targets[qid])
				existing_aug_queries.add(tuple(q_feature))
				queries2labels[tuple(q_feature)] = targets[qid]

	print('aug_samples')
	print(len(existing_aug_queries))

	### end getting all augmented queries ###

	for epoch in range(num_epochs):
		loss_total = 0.

		for batch_idx, data_batch in enumerate(train_data_loader):

			samples, predicates, joins, targets, sample_masks, predicate_masks, join_masks, train_ids = data_batch

			train_ids = train_ids.numpy()
			batch_predicates = [ori_predicates_train[x] for x in train_ids]
			batch_tables = [ori_tables_train[x] for x in train_ids]
			batch_samples = [ori_samples_train[x] for x in train_ids]

			has_pred_on_title_list = []
			for ori_predicate in batch_predicates:
				has_pred_on_title = False
				for a_predicate in ori_predicate:
					if a_predicate[0].split('.')[0] == 't':
						has_pred_on_title = True
						break
				has_pred_on_title_list.append(has_pred_on_title)

			add_pk_list = check_eligible_add_pk(predicates, batch_tables)
			drop_pk_prob_list = check_eligible_drop_pk(has_pred_on_title_list, predicates, batch_tables)
			drop_pk_total_list = check_eligible_drop_pk_no_pred(has_pred_on_title_list, batch_tables)
			eligible_add_filter = check_eligible_add_filter(predicates, column2vec, numerical_cols, batch_predicates,
															batch_tables)

			if cuda:
				samples, predicates, joins, targets = samples.cuda(), predicates.cuda(), joins.cuda(), targets.cuda()
				sample_masks, predicate_masks, join_masks = sample_masks.cuda(), predicate_masks.cuda(), join_masks.cuda()
			samples, predicates, joins, targets = Variable(samples), Variable(predicates), Variable(joins), Variable(
				targets)
			sample_masks, predicate_masks, join_masks = Variable(sample_masks), Variable(predicate_masks), Variable(
				join_masks)

			optimizer.zero_grad()
			outputs = model(samples, predicates, joins, sample_masks, predicate_masks, join_masks)
			# loss = qerror_loss(outputs, targets.float(), min_val, max_val)
			loss = torch.mean(torch.square(torch.squeeze(outputs) - torch.squeeze(targets.float())))

			aug_loss = []
			constraint_loss = []

			aug_count = 0
			constraint_count = 0

			# check if can do drop pk
			subq3_list, sample3_list, subq3_mask_list, sample3_mask_list, indicators3 = drop_pk(
				drop_pk_prob_list, has_pred_on_title_list, predicates, samples, table2vec,
				column2vec, batch_tables, batch_samples)

			subq3_list = torch.from_numpy(np.array(subq3_list, dtype='float32'))
			subq3_mask_list = torch.from_numpy(np.array(subq3_mask_list, dtype='float32'))
			indicators3 = torch.from_numpy(np.array(indicators3, dtype='float32'))
			sample3_list = torch.from_numpy(np.array(sample3_list, dtype='float32'))
			sample3_mask_list = torch.from_numpy(np.array(sample3_mask_list, dtype='float32'))

			if cuda:
				subq3_list, subq3_mask_list, indicators3, sample3_list, sample3_mask_list = \
					subq3_list.cuda(), subq3_mask_list.cuda(), indicators3.cuda(), sample3_list.cuda(), sample3_mask_list.cuda()

			drop_q_feature = model.get_features(sample3_list, subq3_list, joins, sample3_mask_list, subq3_mask_list,
										   join_masks).numpy()

			subq3_out = model(sample3_list, subq3_list, joins, sample3_mask_list, subq3_mask_list, join_masks)


			is_violation = torch.where(torch.squeeze(subq3_out) < torch.squeeze(targets.float()), torch.tensor(1),
									   torch.tensor(0))
			is_violation = torch.where(indicators3 == 1, is_violation, torch.tensor(0))

			for qid, (can_add_pk, can_drop_pk_total, can_drop_pk, can_add_filter) in \
					enumerate(zip(add_pk_list, drop_pk_total_list, is_violation, eligible_add_filter)):

				ordering = []

				###
				# can_add_pk: 1
				# can_drop_pk_total: 2
				# can_drop_pk: 3
				# can_add_filter: 4
				###

				ordering_aug = []
				ordering_loss = []

				if can_add_pk == 1:
					ordering_aug.append(1)
				if can_drop_pk_total == 1:
					ordering_aug.append(2)

				if can_drop_pk == 1:
					if epoch > burn_in:
						ordering_loss.append(3)
				if can_add_filter == 1:
					ordering_loss.append(4)

				random.shuffle(ordering_aug)
				random.shuffle(ordering_loss)

				ordering.extend(ordering_aug)
				ordering.extend(ordering_loss)

				### we enable random choice in constraint generation, since we use break at the end of each if!
				for constraint_choice in ordering:
					if constraint_choice == 1:
						# for the case of add_pk
						subq1_list, sample1_list, join_list, subq1_mask_list, sample1_mask_list, join_mask_list, indicators = add_pk(
							[1], join2vec, max_num_joins, join_feats,
							[predicates[qid]], [samples[qid]], table2vec, [batch_tables[qid]], [batch_samples[qid]])

						subq1_list = torch.from_numpy(np.array(subq1_list, dtype='float32'))
						subq1_mask_list = torch.from_numpy(np.array(subq1_mask_list, dtype='float32'))
						sample1_list = torch.from_numpy(np.array(sample1_list, dtype='float32'))
						sample1_mask_list = torch.from_numpy(np.array(sample1_mask_list, dtype='float32'))
						join_list = torch.from_numpy(np.array(join_list, dtype='float32'))
						join_mask_list = torch.from_numpy(np.array(join_mask_list, dtype='float32'))
						indicators = torch.from_numpy(np.array(indicators, dtype='float32'))

						if cuda:
							subq1_list, subq1_mask_list, sample1_list, sample1_mask_list, join_list, join_mask_list, indicators = \
								subq1_list.cuda(), subq1_mask_list.cuda(), sample1_list.cuda(), sample1_mask_list.cuda(), join_list.cuda(), join_mask_list.cuda(), indicators.cuda()

						subq1_out = model(sample1_list, subq1_list, join_list, sample1_mask_list, subq1_mask_list,
										  join_mask_list)

						aug_loss.append(torch.square(
								torch.squeeze(subq1_out) - torch.squeeze(
									targets[qid].float())))

						aug_count += 1

						break

					elif constraint_choice == 2:
						# for the case of drop_pk with no predicate -> augment query
						subq1_list, join_list, sample1_list, subq1_mask_list, join_mask_list, sample1_mask_list, indicators = drop_pk_no_pred(
							[1], [predicates[qid]], [samples[qid]], table2vec, column2vec, join2vec,
							max_num_joins,
							join_feats,
							[batch_tables[qid]], [batch_samples[qid]])

						subq1_list = torch.from_numpy(np.array(subq1_list, dtype='float32'))
						subq1_mask_list = torch.from_numpy(np.array(subq1_mask_list, dtype='float32'))
						sample1_list = torch.from_numpy(np.array(sample1_list, dtype='float32'))
						sample1_mask_list = torch.from_numpy(np.array(sample1_mask_list, dtype='float32'))
						join_list = torch.from_numpy(np.array(join_list, dtype='float32'))
						join_mask_list = torch.from_numpy(np.array(join_mask_list, dtype='float32'))
						indicators = torch.from_numpy(np.array(indicators, dtype='float32'))

						if cuda:
							subq1_list, subq1_mask_list, sample1_list, sample1_mask_list, join_list, join_mask_list, indicators = \
								subq1_list.cuda(), subq1_mask_list.cuda(), sample1_list.cuda(), sample1_mask_list.cuda(), join_list.cuda(), join_mask_list.cuda(), indicators.cuda()

						subq1_out = model(sample1_list, subq1_list, join_list, sample1_mask_list, subq1_mask_list,
										  join_mask_list)

						aug_loss.append(torch.square(
							torch.squeeze(subq1_out) - torch.squeeze(
								targets[qid].float())))

						aug_count+=1

						break

					elif constraint_choice == 3:
						# for the case of pk-fk inequality constraint
						if not check_duplicate(drop_q_feature[qid], existing_aug_queries):
							### we apply pseudo labeling here
							subq1_pert, subq2_pert, subq1_mask_pert, subq2_mask_pert, indicators_pert, sample1_pert, sample2_pert = expand_k_query_add_filter(
								[1], k, [subq3_list[qid]], [sample3_list[qid]],
								column_min_max_vals,
								table2vec,
								column2vec,
								op2vec,
								numerical_cols, [batch_predicates[qid]], [batch_tables[qid]], [batch_samples[qid]], table_to_df,
								rs)

							subq1_pert = torch.from_numpy(np.array(subq1_pert, dtype='float32'))
							subq2_pert = torch.from_numpy(np.array(subq2_pert, dtype='float32'))
							subq1_mask_pert = torch.from_numpy(np.array(subq1_mask_pert, dtype='float32'))
							subq2_mask_pert = torch.from_numpy(np.array(subq2_mask_pert, dtype='float32'))
							sample1_pert = torch.from_numpy(np.array(sample1_pert, dtype='float32'))
							sample2_pert = torch.from_numpy(np.array(sample2_pert, dtype='float32'))
							indicators_pert = torch.from_numpy(np.array(indicators_pert, dtype='float32'))

							if cuda:
								subq1_pert, subq2_pert, subq1_mask_pert, subq2_mask_pert, sample1_pert, sample2_pert = \
									subq1_pert.cuda(), subq2_pert.cuda(), subq1_mask_pert.cuda(), subq2_mask_pert.cuda(), sample1_pert.cuda(), sample2_pert.cuda()

							subq1_out_pert = model(sample1_pert, subq1_pert, torch.unsqueeze(joins[qid,:,:], 0).repeat_interleave(k, dim=0),
												   torch.unsqueeze(sample3_mask_list[qid,:,:], 0).repeat_interleave(k, dim=0), subq1_mask_pert,
												   torch.unsqueeze(join_masks[qid,:,:], 0).repeat_interleave(k, dim=0))
							subq2_out_pert = model(sample2_pert, subq2_pert, torch.unsqueeze(joins[qid,:,:], 0).repeat_interleave(k, dim=0),
												   torch.unsqueeze(sample3_mask_list[qid,:,:], 0).repeat_interleave(k, dim=0), subq2_mask_pert,
												   torch.unsqueeze(join_masks[qid,:,:], 0).repeat_interleave(k, dim=0))

							subq1_out_pert = unnormalize_torch(torch.squeeze(subq1_out_pert), min_val, max_val)
							subq2_out_pert = unnormalize_torch(torch.squeeze(subq2_out_pert), min_val, max_val)
							norm_pred_pert = normalize_torch(subq1_out_pert + subq2_out_pert, min_val, max_val)

							norm_pred_pert = norm_pred_pert.view(-1, k)

							### get the avg value of k perturbations
							pse_labels = torch.mean(norm_pred_pert, dim=-1).detach()
							pse_labels = torch.where(pse_labels >= torch.squeeze(targets[qid].float()), pse_labels,
													 torch.squeeze(targets[qid].float()))
							pse_labels = torch.where(indicators_pert == 1, pse_labels, torch.squeeze(targets[qid].float()))

							constraint_loss.append(torch.square(torch.squeeze(subq3_out[qid]) - torch.squeeze(pse_labels)))
							constraint_count += 1
						else:
							# we know the true label!
							true_target  = queries2labels[tuple(drop_q_feature[qid])]
							constraint_loss.append(
								torch.square(torch.squeeze(subq3_out[qid]) - torch.squeeze(true_target)))
							constraint_count += 1
						break
					else:
						# for the case of consistency constraint
						subq1_list, subq2_list, subq1_mask_list, subq2_mask_list, indicators, sample1_list, sample2_list = expand_query_add_filter(
							[1], [predicates[qid]], [samples[qid]],
							column_min_max_vals,
							table2vec,
							column2vec,
							op2vec,
							numerical_cols, [batch_predicates[qid]], [batch_tables[qid]], [batch_samples[qid]], table_to_df,
							rs)

						subq1_list = torch.from_numpy(np.array(subq1_list, dtype='float32'))
						subq2_list = torch.from_numpy(np.array(subq2_list, dtype='float32'))
						subq1_mask_list = torch.from_numpy(np.array(subq1_mask_list, dtype='float32'))
						subq2_mask_list = torch.from_numpy(np.array(subq2_mask_list, dtype='float32'))
						indicators = torch.from_numpy(np.array(indicators, dtype='float32'))
						sample1_list = torch.from_numpy(np.array(sample1_list, dtype='float32'))
						sample2_list = torch.from_numpy(np.array(sample2_list, dtype='float32'))
						if cuda:
							subq1_list, subq2_list, subq1_mask_list, subq2_mask_list, indicators, sample1_list, sample2_list = \
								subq1_list.cuda(), subq2_list.cuda(), subq1_mask_list.cuda(), subq2_mask_list.cuda(), indicators.cuda(), sample1_list.cuda(), sample2_list.cuda()

						subq1_out = model(sample1_list, subq1_list, torch.unsqueeze(joins[qid,:,:], 0),
						                  torch.unsqueeze(sample_masks[qid,:,:], 0),  subq1_mask_list, torch.unsqueeze(join_masks[qid,:,:], 0))
						subq2_out = model(sample2_list, subq2_list, torch.unsqueeze(joins[qid,:,:], 0),
						                  torch.unsqueeze(sample_masks[qid,:,:], 0),  subq2_mask_list, torch.unsqueeze(join_masks[qid,:,:], 0))

						subq1_out = unnormalize_torch(torch.squeeze(subq1_out), min_val, max_val)
						subq2_out = unnormalize_torch(torch.squeeze(subq2_out), min_val, max_val)
						norm_pred = normalize_torch(subq1_out + subq2_out, min_val, max_val)

						constraint_loss.append(torch.square(
									torch.squeeze(norm_pred) - torch.squeeze(
										targets[qid].float())))
						constraint_count += 1
						break

			if aug_count > 0:
				avg_aug_loss = torch.mean(torch.stack(aug_loss))
			else:
				avg_aug_loss = 0

			if constraint_count > 0:
				avg_constraint_loss = torch.mean(torch.stack(constraint_loss))
			else:
				avg_constraint_loss = 0


			loss = loss +  weight * avg_aug_loss + weight * avg_constraint_loss

			loss_total += loss.item()
			loss.backward()
			optimizer.step()


		print("Epoch {}, loss: {}".format(epoch, loss_total / len(train_data_loader)))

		preds_candi, candi_label = predict_and_get_labels(model, candi_data_loader, cuda)
		candi_label = unnormalize_labels(candi_label, min_val, max_val)


		# Unnormalize
		preds_card_unnorm = unnormalize_labels(preds_candi, min_val, max_val)

		# Print metrics
		print("\nQ-Error " + 'test in' + ":")
		qerror_res = print_qerror(preds_card_unnorm, candi_label)

		#################################
		#### load out-of-distribution workload

		workload_name2 = 'job-light-subs-star'
		file_name = "workloads/" + workload_name2
		joins, predicates, tables, samples, label, test_num_joins, test_num_predicates, test_table_sets, numerical_cols = ori_load_data(
			file_name,
			num_materialized_samples)

		# Get feature encoding and proper normalization
		samples_test = encode_samples(tables, samples, table2vec)
		predicates_test, joins_test = encode_data(predicates, joins, column_min_max_vals, column2vec, op2vec, join2vec)
		labels_test, _, _ = normalize_labels(label, min_val, max_val)

		print("Number of test samples: {}".format(len(labels_test)))

		max_num_predicates = max([len(p) for p in predicates_test])
		max_num_joins = max([len(j) for j in joins_test])

		# Get test set predictions
		test_data = make_dataset(samples_test, predicates_test, joins_test, labels_test, max_num_joins,
								 max_num_predicates)
		test_data_loader = DataLoader(test_data, batch_size=batch_size)

		preds_test, t_total = predict(model, test_data_loader, cuda)
		print("Prediction time per test sample: {}".format(t_total / len(labels_test) * 1000))

		# Unnormalize
		preds_test_unnorm = unnormalize_labels(preds_test, min_val, max_val)

		# Print metrics
		print("\nQ-Error " + workload_name2 + ":")
		print_qerror(preds_test_unnorm, label)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--testset", help="workload of ood queries", default='job-light-subs-star')
	parser.add_argument("--queries", help="number of training queries (default: 30000)", type=int, default=30000)
	parser.add_argument("--epochs", help="number of epochs (default: 10)", type=int, default=50)
	parser.add_argument("--batch", help="batch size (default: 1024)", type=int, default=400)
	parser.add_argument("--hid", help="number of hidden units (default: 256)", type=int, default=256)
	parser.add_argument("--cuda", help="use CUDA", action="store_true")
	args = parser.parse_args()

	table_list = []
	q_f = open('./workloads/imdb-train.csv', 'r', encoding='utf-8')
	q_lines = q_f.readlines()
	for q_line in q_lines:
		parts = q_line.split('#')
		tables = parts[0].split(',')
		for table in tables:
			full_table = table.split(' ')[0]
			if full_table not in table_list:
				table_list.append(full_table)

	sample_dir = "./samples/IMDb"

	table_to_df = get_bitmap.load_tables(table_list, data_dir=sample_dir)

	train_and_predict(args.testset, args.queries, args.epochs, args.batch, args.hid, args.cuda, table_to_df, burn_in=20)

if __name__ == "__main__":
	main()
