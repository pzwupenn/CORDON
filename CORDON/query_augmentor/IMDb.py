import torch
import numpy as np

def add_pk(query_indicators, join2vec, num_joins, len_join, predicates, sample_encs, table2vec,
                ori_tables, ori_samples):
	if torch.is_tensor(predicates):
		predicates = predicates.numpy()

	num_filters = predicates[0].shape[0]
	len_vec = predicates[0].shape[1]

	num_samples = sample_encs[0].shape[0]
	len_samples = sample_encs[0].shape[1]

	sample_list = []
	sample_mask_list = []

	predicate_list = []
	predicate_mask_list = []

	join_list = []
	join_mask_list = []

	indicators = []

	for qid, query in enumerate(predicates):

		if torch.is_tensor(query):
			query = query.numpy()

		ori_table = ori_tables[qid]

		samples_enc = list()
		predicates_enc = list()
		join_enc = list()

		samples_enc_mask = list()
		predicates_enc_mask = list()
		join_enc_mask = list()

		if len(ori_table) != 1 or ori_table[0] == 'title t' or query_indicators[qid] == 0:
			for i in range(num_filters):
				predicates_enc.append(np.zeros((len_vec)))
			predicate_list.append(predicates_enc)
			predicate_mask_list.append(np.expand_dims(np.ones((num_filters)), 1))

			for i in range(num_samples):
				samples_enc.append(np.zeros((len_samples)))
			sample_list.append(samples_enc)
			sample_mask_list.append(np.expand_dims(np.ones((num_samples)), 1))

			for i in range(num_joins):
				join_enc.append(np.zeros((len_join)))
			join_list.append(join_enc)
			join_mask_list.append(np.expand_dims(np.ones((num_joins)), 1))
			indicators.append(0)
			continue


		ori_table = ori_tables[qid]
		ori_sample = ori_samples[qid]
		indicators.append(1)

		dimension_table = ori_table[0]

		chosen_table_abbrev = dimension_table.split(' ')[1]
		new_join = "t.id={}.movie_id".format(chosen_table_abbrev)
		new_join = join2vec[new_join]

		join_enc.append(new_join)
		join_enc_mask.append([1.])

		for i, table in enumerate(ori_table):
			sample_vec = []
			sample_vec.append(table2vec[table])
			sample_vec.append(ori_sample[i])
			sample_vec = np.hstack(sample_vec)
			samples_enc.append(sample_vec)
			samples_enc_mask.append([1.])

		fact_table = 'title t'
		# for the case of title
		sample_vec = []
		sample_vec.append(table2vec[fact_table])
		sample_vec.append(np.ones(1000, dtype=bool))
		sample_vec = np.hstack(sample_vec)
		samples_enc.append(sample_vec)
		samples_enc_mask.append([1.])

		for predicate_enc in query:
			# Proper predicate
			is_all_zero = np.all(predicate_enc == 0.)
			if not is_all_zero:
				predicates_enc.append(predicate_enc)
				predicates_enc_mask.append([1.])

		if len(predicates_enc) == 0:
			predicates_enc.append(np.zeros((len_vec)))
			predicates_enc_mask.append([1.])

		if len(predicates_enc) < num_filters:
			for _ in range(num_filters - len(predicates_enc)):
				predicates_enc.append(np.zeros((len_vec)))
				predicates_enc_mask.append([0.])

		if len(samples_enc) < num_samples:
			for _ in range(num_samples - len(samples_enc)):
				samples_enc.append(np.zeros((len_samples)))
				samples_enc_mask.append([0.])

		if len(join_enc) < num_joins:
			for _ in range(num_joins - len(join_enc)):
				join_enc.append(np.zeros((len_join)))
				join_enc_mask.append([0.])

		predicate_list.append(predicates_enc)
		join_list.append(join_enc)
		sample_list.append(samples_enc)

		predicate_mask_list.append(predicates_enc_mask)
		join_mask_list.append(join_enc_mask)
		sample_mask_list.append(samples_enc_mask)


	return predicate_list, sample_list, join_list,  predicate_mask_list, sample_mask_list, join_mask_list, indicators

def drop_pk_no_pred(drop_pk_total_indicator, predicates, sample_encs, table2vec, column2vec, join2vec, num_joins, len_join, ori_tables, ori_samples):
	len_colvec = len(column2vec)

	vec2column = {tuple(value): key for key, value in column2vec.items()}

	if torch.is_tensor(predicates):
		predicates = predicates.numpy()

	num_filters = predicates[0].shape[0]
	len_vec = predicates[0].shape[1]

	num_samples = sample_encs[0].shape[0]
	len_samples = sample_encs[0].shape[1]

	join_list = []
	join_mask_list = []

	sample_list = []
	sample_mask_list = []

	predicate_list = []
	predicate_mask_list = []

	indicators = []

	for qid, query in enumerate(predicates):

		if torch.is_tensor(query):
			query = query.numpy()

		samples_enc = list()
		joins_enc = list()
		predicates_enc = list()

		samples_enc_mask = list()
		joins_enc_mask = list()
		predicates_enc_mask = list()


		if drop_pk_total_indicator[qid] == 0:
			for i in range(num_filters):
				predicates_enc.append(np.zeros((len_vec)))
			predicate_list.append(predicates_enc)
			predicate_mask_list.append(np.expand_dims(np.ones((num_filters)), 1))

			for i in range(num_samples):
				samples_enc.append(np.zeros((len_samples)))
			sample_list.append(samples_enc)
			sample_mask_list.append(np.expand_dims(np.ones((num_samples)), 1))

			for i in range(num_joins):
				joins_enc.append(np.zeros((len_join)))
			join_list.append(joins_enc)
			join_mask_list.append(np.expand_dims(np.ones((num_joins)), 1))
			indicators.append(0)
			continue

		ori_table = ori_tables[qid]
		ori_sample = ori_samples[qid]

		# the case for two tables, we first separate two subqueries
		# then, C(dimension table) >= C(join) should hold
		indicators.append(1)

		new_join = ''
		new_join = join2vec[new_join]

		joins_enc.append(new_join)
		joins_enc_mask.append([1.])

		for i, table in enumerate(ori_table):
			if table != 'title t':
				sample_vec = []
				sample_vec.append(table2vec[table])
				sample_vec.append(ori_sample[i])
				sample_vec = np.hstack(sample_vec)
				samples_enc.append(sample_vec)
				samples_enc_mask.append([1.])
				break

		for predicate_enc in query:
			# Proper predicate
			is_all_zero = np.all(predicate_enc == 0.)
			if not is_all_zero:
				column = predicate_enc[:len_colvec]
				column = vec2column[tuple(column)]
				table_abbrev = column.split('.')[0]
				predicates_enc.append(predicate_enc)
				predicates_enc_mask.append([1.])

		if len(predicates_enc) == 0:
			predicates_enc.append(np.zeros((len_vec)))
			predicates_enc_mask.append([1.])

		if len(predicates_enc) < num_filters:
			for _ in range(num_filters - len(predicates_enc)):
				predicates_enc.append(np.zeros((len_vec)))
				predicates_enc_mask.append([0.])

		if len(joins_enc) < num_joins:
			for _ in range(num_joins - len(joins_enc)):
				joins_enc.append(np.zeros((len_join)))
				joins_enc_mask.append([0.])

		if len(samples_enc) < num_samples:
			for _ in range(num_samples - len(samples_enc)):
				samples_enc.append(np.zeros((len_samples)))
				samples_enc_mask.append([0.])

		predicate_list.append(predicates_enc)
		join_list.append(joins_enc)
		sample_list.append(samples_enc)

		predicate_mask_list.append(predicates_enc_mask)
		join_mask_list.append(joins_enc_mask)
		sample_mask_list.append(samples_enc_mask)

	return predicate_list, join_list, sample_list, predicate_mask_list, join_mask_list, sample_mask_list, indicators
