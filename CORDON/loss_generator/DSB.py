import numpy as np
import random
import torch
import mscn.get_bitmap_dsb as get_bitmap

random.seed(42)

def unnormalize_data(val_norm, column_name, column_min_max_vals):
	min_val = column_min_max_vals[column_name][0]
	max_val = column_min_max_vals[column_name][1]
	if val_norm == 0:
		return min_val
	if val_norm == 1:
		return max_val

	val_norm = float(val_norm)
	val = (val_norm * (max_val - min_val)) + min_val

	return round(val)

def encode_samples2(tables, samples, table2vec, chosen_tables, new_samples):
	samples_enc = []
	sub_list = []
	for i, query in enumerate(tables):
		samples_enc.append(list())
		chosen_table = chosen_tables[i]
		for j, table in enumerate(query):
			sample_vec = []
			# Append table one-hot vector
			sample_vec.append(table2vec[table])
			# Append bit vector
			table = table.split(' ')[1]
			if chosen_table != table:
				sample_vec.append(samples[i][j])
			else:
				sample_vec.append(new_samples[i][0])
				sub_list.append(new_samples[i][0])
				original_sample = samples[i][j]
			sample_vec = np.hstack(sample_vec)
			samples_enc[i].append(sample_vec)

	return samples_enc

def encode_samples3(tables, samples, table2vec, chosen_tables, new_samples):
	samples_enc = []
	for i, query in enumerate(tables):
		samples_enc.append(list())
		chosen_table = chosen_tables[i]
		for j, table in enumerate(query):
			sample_vec = []
			# Append table one-hot vector
			sample_vec.append(table2vec[table])
			# Append bit vector
			table = table.split(' ')[1]
			if chosen_table != table:
				sample_vec.append(samples[i][j])
			else:
				sample_vec.append(new_samples[i][0])
			sample_vec = np.hstack(sample_vec)
			samples_enc[i].append(sample_vec)

	return samples_enc

def expand_k_query_add_filter(break_query_indicators, k, predicates, sample_encs, column_min_max_vals, table2vec, column2vec,
                            op2vec, numerical_cols,
                            ori_predicates, ori_tables, ori_samples, dropped_table_list, table_to_df, rs):

	#### use consistent constraint to find out k sets of subqueries for queries that are extrated by PK-FK constraints

	len_colvec = len(column2vec)
	vec2column = {tuple(value): key for key, value in column2vec.items()}

	candi_cols = []
	for col_name in numerical_cols:
		col_vec = column2vec[col_name]
		col_id = np.where(col_vec == 1)[0][0]
		candi_cols.append(col_id)

	if torch.is_tensor(predicates):
		predicates = predicates.numpy()

	num_filters = predicates[0].shape[0]
	len_vec = predicates[0].shape[1]

	num_samples = sample_encs[0].shape[0]
	len_samples = sample_encs[0].shape[1]

	subq1_list = []
	subq1_mask_list = []
	subq2_list = []
	subq2_mask_list = []
	indicators = []

	sample1_list = []
	sample2_list = []

	for qid, query in enumerate(predicates):

		if torch.is_tensor(query):
			query = query.numpy()

		if break_query_indicators[qid] == 0:
			indicators.append(0)
			for _ in range(k):
				predicates_subq1_enc = []
				predicates_subq2_enc = []
				samples_subq1 = []
				samples_subq2 = []

				for i in range(num_filters):
					predicates_subq1_enc.append(np.zeros((len_vec)))
					predicates_subq2_enc.append(np.zeros((len_vec)))
				subq1_list.append(predicates_subq1_enc)
				subq1_mask_list.append(np.expand_dims(np.ones((num_filters)), 1))
				subq2_list.append(predicates_subq2_enc)
				subq2_mask_list.append(np.expand_dims(np.ones((num_filters)), 1))
				for i in range(num_samples):
					samples_subq1.append(np.zeros((len_samples)))
					samples_subq2.append(np.zeros((len_samples)))
				sample1_list.append(samples_subq1)
				sample2_list.append(samples_subq2)
			continue
		queried_cols = np.zeros((len_colvec))

		if ori_predicates[qid][0][0] != '':
			for predicate in ori_predicates[qid]:
				column = predicate[0]
				col_vec = column2vec[column]
				column_id = np.where(col_vec == 1)[0][0]
				queried_cols[column_id] = 1


		is_all_one = np.all(queried_cols == 1)
		if is_all_one:
			indicators.append(0)
			for _ in range(k):
				predicates_subq1_enc = []
				predicates_subq2_enc = []
				samples_subq1 = []
				samples_subq2 = []


				for i in range(num_filters):
					predicates_subq1_enc.append(np.zeros((len_vec)))
					predicates_subq2_enc.append(np.zeros((len_vec)))
				subq1_list.append(predicates_subq1_enc)
				subq1_mask_list.append(np.expand_dims(np.ones((num_filters)), 1))
				subq2_list.append(predicates_subq2_enc)
				subq2_mask_list.append(np.expand_dims(np.ones((num_filters)), 1))
				for i in range(num_samples):
					samples_subq1.append(np.zeros((len_samples)))
					samples_subq2.append(np.zeros((len_samples)))
				sample1_list.append(samples_subq1)
				sample2_list.append(samples_subq2)
			continue
		else:
			zero_indexes = np.where(queried_cols == 0)[0]
			intersection = [value for value in zero_indexes if value in candi_cols]

			dropped_table = dropped_table_list[qid]
			ori_predicate = ori_predicates[qid]
			ori_table = ori_tables[qid]
			ori_sample = ori_samples[qid]

			after_table = []
			after_sample = []

			for t, s in zip(ori_table, ori_sample):
				if t != dropped_table:
					after_table.append(t)
					after_sample.append(s)

			abbrev_ori_table = [t.split(' ')[1] for t in after_table]

			valid_table_indexes = []
			for col in numerical_cols:
				col_vec = column2vec[col]
				if col.split('.')[0] in abbrev_ori_table:
					col_id = np.where(col_vec == 1)[0][0]
					valid_table_indexes.append(col_id)

			# production_year_vec = column2vec['t.production_year']
			# production_year_col_id = np.where(production_year_vec == 1)[0][0]

			ultimate_intersection = []
			for value in intersection:
				if (value in valid_table_indexes):
					ultimate_intersection.append(value)

			intersection = ultimate_intersection

			if len(intersection) == 0:
				indicators.append(0)
				for _ in range(k):
					predicates_subq1_enc = []
					predicates_subq2_enc = []
					samples_subq1 = []
					samples_subq2 = []

					for i in range(num_filters):
						predicates_subq1_enc.append(np.zeros((len_vec)))
						predicates_subq2_enc.append(np.zeros((len_vec)))
					subq1_list.append(predicates_subq1_enc)
					subq1_mask_list.append(np.expand_dims(np.ones((num_filters)), 1))
					subq2_list.append(predicates_subq2_enc)
					subq2_mask_list.append(np.expand_dims(np.ones((num_filters)), 1))
					for i in range(num_samples):
						samples_subq1.append(np.zeros((len_samples)))
						samples_subq2.append(np.zeros((len_samples)))
					sample1_list.append(samples_subq1)
					sample2_list.append(samples_subq2)
				continue

			indicators.append(1)
			for _ in range(k):
				predicates_subq1_enc = []
				predicates_subq1_mask = []
				predicates_subq2_enc = []
				predicates_subq2_mask = []

				pred1_a_q = []
				pred2_a_q = []
				chosen_col = random.choice(intersection)

				mid_point = rs.uniform(0., 1.)
				col_vec = np.zeros(len_colvec)
				col_vec[chosen_col] = 1.
				unnorm_mid_point = unnormalize_data(mid_point, vec2column[tuple(col_vec)], column_min_max_vals)
				chosen_table = vec2column[tuple(col_vec)].split('.')[0]

				op_choice = rs.uniform(0., 1.)
				if op_choice < 0.5:
					smaller_op = '<='
					larger_op = '>'
				else:
					smaller_op = '<'
					larger_op = '>='

				pred_vec_subq1 = []
				pred_vec_subq1.append(col_vec)
				pred_vec_subq1.append(op2vec[smaller_op])
				pred_vec_subq1.append(mid_point)
				pred_vec_subq1 = np.hstack(pred_vec_subq1)

				predicates_subq1_enc.append(pred_vec_subq1)
				predicates_subq1_mask.append([1.])
				pred1_a_q.append([vec2column[tuple(col_vec)], smaller_op, unnorm_mid_point])

				pred_vec_subq2 = []
				pred_vec_subq2.append(col_vec)
				pred_vec_subq2.append(op2vec[larger_op])
				pred_vec_subq2.append(mid_point)
				pred_vec_subq2 = np.hstack(pred_vec_subq2)

				predicates_subq2_enc.append(pred_vec_subq2)
				predicates_subq2_mask.append([1.])
				pred2_a_q.append([vec2column[tuple(col_vec)], larger_op, unnorm_mid_point])

				for predicate_enc in query:
					# Proper predicate
					is_all_zero = np.all(predicate_enc == 0.)
					if not is_all_zero:
						predicates_subq1_enc.append(predicate_enc)
						predicates_subq1_mask.append([1.])
						predicates_subq2_enc.append(predicate_enc)
						predicates_subq2_mask.append([1.])

				for predicate in ori_predicate:
					if predicate[0].split('.')[0] == chosen_table:
						pred1_a_q.append(predicate)
						pred2_a_q.append(predicate)

				## compute samples for new queries
				bitmap_list = get_bitmap.compute_bitmap(table_to_df, [chosen_table, chosen_table], [pred1_a_q, pred2_a_q])
				sample_enc_list = encode_samples3([after_table, after_table], [after_sample, after_sample], table2vec,
				                                          [chosen_table, chosen_table], bitmap_list)


				samples_subq1 = sample_enc_list[0]
				samples_subq2 = sample_enc_list[1]

				if len(predicates_subq1_enc) < num_filters:
					for _ in range(num_filters - len(predicates_subq1_enc)):
						predicates_subq1_enc.append(np.zeros((len_vec)))
						predicates_subq1_mask.append([0.])

				if len(predicates_subq2_enc) < num_filters:
					for _ in range(num_filters - len(predicates_subq2_enc)):
						predicates_subq2_enc.append(np.zeros((len_vec)))
						predicates_subq2_mask.append([0.])

				if len(samples_subq1) < num_samples:
					for _ in range(num_samples - len(samples_subq1)):
						samples_subq1.append(np.zeros((len_samples)))

				if len(samples_subq2) < num_samples:
					for _ in range(num_samples - len(samples_subq2)):
						samples_subq2.append(np.zeros((len_samples)))

				subq1_list.append(np.array(predicates_subq1_enc))
				subq1_mask_list.append(np.array(predicates_subq1_mask))

				subq2_list.append(np.array(predicates_subq2_enc))
				subq2_mask_list.append(np.array(predicates_subq2_mask))

				sample1_list.append(samples_subq1)
				sample2_list.append(samples_subq2)

	return subq1_list, subq2_list, subq1_mask_list, subq2_mask_list, indicators, sample1_list, sample2_list

def drop_pk(dim_table_list, predicates, sample_encs, table2vec, column2vec, join2vec, num_joins, len_join, ori_tables, ori_samples):
	len_colvec = len(column2vec)

	vec2column = {tuple(value): key for key, value in column2vec.items()}

	predicates = predicates.numpy()

	num_filters = predicates[0].shape[0]
	len_vec = predicates[0].shape[1]

	num_samples = sample_encs[0].shape[0]
	len_samples = sample_encs[0].shape[1]

	table2joins = {'date_dim d': 'd.d_date_sk=ss.ss_sold_date_sk',
	               'customer_demographics cd': 'cd.cd_demo_sk=ss.ss_cdemo_sk',
	               'household_demographics hd': 'hd.hd_demo_sk=ss.ss_hdemo_sk',
	               'store s': 's.s_store_sk=ss.ss_store_sk'}

	join_list = []
	join_mask_list = []

	sample_list = []
	sample_mask_list = []

	predicate_list = []
	predicate_mask_list = []

	dropped_table_list = []

	indicators = []

	for qid, query in enumerate(predicates):

		ori_table = ori_tables[qid]
		dim_tables = dim_table_list[qid] # full name!
		samples_enc = list()
		joins_enc = list()
		predicates_enc = list()

		samples_enc_mask = list()
		joins_enc_mask = list()
		predicates_enc_mask = list()

		if len(dim_tables) == 0:
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
			dropped_table_list.append('')
			continue

		ori_sample = ori_samples[qid]
		chosen_table = random.choice(dim_tables)
		dropped_table_list.append(chosen_table)

		# the case for two tables, we first separate two subqueries
		# then, C(dimension table) >= C(join) should hold
		indicators.append(2)

		if len(ori_table) == 2: # just 1 after drop 1
			new_join = ''
			new_join = join2vec[new_join]

			joins_enc.append(new_join)
			joins_enc_mask.append([1.])
		else: # >= 3 tables
			for t_name in ori_table:
				if t_name != chosen_table and t_name != 'store_sales ss':
					joins_enc.append(join2vec[table2joins[t_name]])
					joins_enc_mask.append([1.])

		for i, table in enumerate(ori_table):
			if table != chosen_table:
				sample_vec = []
				sample_vec.append(table2vec[table])
				sample_vec.append(ori_sample[i])
				sample_vec = np.hstack(sample_vec)
				samples_enc.append(sample_vec)
				samples_enc_mask.append([1.])

		for predicate_enc in query:
			# Proper predicate
			is_all_zero = np.all(predicate_enc == 0.)
			if not is_all_zero:
				column = predicate_enc[:len_colvec]
				column = vec2column[tuple(column)]
				table_abbrev = column.split('.')[0]
				if table_abbrev != chosen_table.split(' ')[1]:
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

	return predicate_list, join_list, sample_list, predicate_mask_list, join_mask_list, sample_mask_list, indicators, dropped_table_list


def expand_query_add_filter(break_query_indicators, predicates, sample_encs, column_min_max_vals, table2vec, column2vec,
                            op2vec, numerical_cols, ori_predicates, ori_tables, ori_samples, table_to_df, rs):
	len_colvec = len(column2vec)
	len_table = len(table2vec)

	vec2column = {tuple(value): key for key, value in column2vec.items()}

	if torch.is_tensor(predicates):
		predicates = predicates.numpy()

	candi_cols = []
	for col_name in numerical_cols:
		col_vec = column2vec[col_name]
		col_id = np.where(col_vec == 1)[0][0]
		candi_cols.append(col_id)


	num_filters = predicates[0].shape[0]
	len_vec = predicates[0].shape[1]

	num_samples = sample_encs[0].shape[0]
	len_samples = sample_encs[0].shape[1]

	subq1_list = []
	subq1_mask_list = []
	subq2_list = []
	subq2_mask_list = []
	indicators = []

	sample1_list = []
	sample2_list = []

	for qid, query in enumerate(predicates):
		if torch.is_tensor(query):
			query = query.numpy()

		predicates_subq1_enc = []
		predicates_subq1_mask = []
		predicates_subq2_enc = []
		predicates_subq2_mask = []
		samples_subq1 = []
		samples_subq2 = []

		pred1_a_q = []
		pred2_a_q = []

		if break_query_indicators[qid] == 0:
			for i in range(num_filters):
				predicates_subq1_enc.append(np.zeros((len_vec)))
				predicates_subq2_enc.append(np.zeros((len_vec)))
			subq1_list.append(predicates_subq1_enc)
			subq1_mask_list.append(np.expand_dims(np.ones((num_filters)), 1))
			subq2_list.append(predicates_subq2_enc)
			subq2_mask_list.append(np.expand_dims(np.ones((num_filters)), 1))
			indicators.append(0)
			for i in range(num_samples):
				samples_subq1.append(np.zeros((len_samples)))
				samples_subq2.append(np.zeros((len_samples)))
			sample1_list.append(samples_subq1)
			sample2_list.append(samples_subq2)
			continue

		queried_cols = np.zeros((len_colvec))

		if ori_predicates[qid][0][0] != '':
			for predicate in ori_predicates[qid]:
				column = predicate[0]
				col_vec = column2vec[column]
				column_id = np.where(col_vec == 1)[0][0]
				queried_cols[column_id] = 1

		is_all_one = np.all(queried_cols == 1)
		if is_all_one:
			for i in range(num_filters):
				predicates_subq1_enc.append(np.zeros((len_vec)))
				predicates_subq2_enc.append(np.zeros((len_vec)))
			subq1_list.append(predicates_subq1_enc)
			subq1_mask_list.append(np.expand_dims(np.ones((num_filters)), 1))
			subq2_list.append(predicates_subq2_enc)
			subq2_mask_list.append(np.expand_dims(np.ones((num_filters)), 1))
			indicators.append(0)
			for i in range(num_samples):
				samples_subq1.append(np.zeros((len_samples)))
				samples_subq2.append(np.zeros((len_samples)))
			sample1_list.append(samples_subq1)
			sample2_list.append(samples_subq2)
		else:
			zero_indexes = np.where(queried_cols == 0)[0]
			intersection = [value for value in zero_indexes if value in candi_cols]

			ori_predicate = ori_predicates[qid]
			ori_table = ori_tables[qid]
			ori_sample = ori_samples[qid]
			abbrev_ori_table = [t.split(' ')[1] for t in ori_table]

			valid_table_indexes = []
			for col in numerical_cols:
				col_vec = column2vec[col]
				if col.split('.')[0] in abbrev_ori_table:
					col_id = np.where(col_vec == 1)[0][0]
					valid_table_indexes.append(col_id)

			ultimate_intersection = []
			for value in intersection:
				if (value in valid_table_indexes):
					ultimate_intersection.append(value)

			intersection = ultimate_intersection

			if len(intersection) == 0:
				for i in range(num_filters):
					predicates_subq1_enc.append(np.zeros((len_vec)))
					predicates_subq2_enc.append(np.zeros((len_vec)))
				subq1_list.append(predicates_subq1_enc)
				subq1_mask_list.append(np.expand_dims(np.ones((num_filters)), 1))
				subq2_list.append(predicates_subq2_enc)
				subq2_mask_list.append(np.expand_dims(np.ones((num_filters)), 1))
				indicators.append(0)
				for i in range(num_samples):
					samples_subq1.append(np.zeros((len_samples)))
					samples_subq2.append(np.zeros((len_samples)))
				sample1_list.append(samples_subq1)
				sample2_list.append(samples_subq2)
				continue

			chosen_col = random.choice(intersection)

			mid_point = rs.uniform(0., 1.)
			col_vec = np.zeros(len_colvec)
			col_vec[chosen_col] = 1.
			unnorm_mid_point = unnormalize_data(mid_point, vec2column[tuple(col_vec)], column_min_max_vals)
			chosen_table = vec2column[tuple(col_vec)].split('.')[0]

			op_choice = rs.uniform(0., 1.)
			if op_choice < 0.5:
				smaller_op = '<='
				larger_op = '>'
			else:
				smaller_op = '<'
				larger_op = '>='

			pred_vec_subq1 = []
			pred_vec_subq1.append(col_vec)
			pred_vec_subq1.append(op2vec[smaller_op])
			pred_vec_subq1.append(mid_point)
			pred_vec_subq1 = np.hstack(pred_vec_subq1)

			predicates_subq1_enc.append(pred_vec_subq1)
			predicates_subq1_mask.append([1.])
			pred1_a_q.append([vec2column[tuple(col_vec)], smaller_op, unnorm_mid_point])

			pred_vec_subq2 = []
			pred_vec_subq2.append(col_vec)
			pred_vec_subq2.append(op2vec[larger_op])
			pred_vec_subq2.append(mid_point)
			pred_vec_subq2 = np.hstack(pred_vec_subq2)

			predicates_subq2_enc.append(pred_vec_subq2)
			predicates_subq2_mask.append([1.])
			pred2_a_q.append([vec2column[tuple(col_vec)], larger_op, unnorm_mid_point])

			for predicate_enc in query:
				# Proper predicate
				is_all_zero = np.all(predicate_enc == 0.)
				if not is_all_zero:
					predicates_subq1_enc.append(predicate_enc)
					predicates_subq1_mask.append([1.])
					predicates_subq2_enc.append(predicate_enc)
					predicates_subq2_mask.append([1.])

			for predicate in ori_predicate:
				if predicate[0].split('.')[0] == chosen_table:
					pred1_a_q.append(predicate)
					pred2_a_q.append(predicate)

			## compute samples for new queries
			bitmap_list = get_bitmap.compute_bitmap(table_to_df, [chosen_table, chosen_table], [pred1_a_q, pred2_a_q])
			sample_enc_list = encode_samples2([ori_table, ori_table], [ori_sample, ori_sample], table2vec,
			                                          [chosen_table, chosen_table], bitmap_list)

			samples_subq1 = sample_enc_list[0]
			samples_subq2 = sample_enc_list[1]

			if len(predicates_subq1_enc) < num_filters:
				for _ in range(num_filters - len(predicates_subq1_enc)):
					predicates_subq1_enc.append(np.zeros((len_vec)))
					predicates_subq1_mask.append([0.])

			if len(predicates_subq2_enc) < num_filters:
				for _ in range(num_filters - len(predicates_subq2_enc)):
					predicates_subq2_enc.append(np.zeros((len_vec)))
					predicates_subq2_mask.append([0.])

			if len(samples_subq1) < num_samples:
				for _ in range(num_samples - len(samples_subq1)):
					samples_subq1.append(np.zeros((len_samples)))

			if len(samples_subq2) < num_samples:
				for _ in range(num_samples - len(samples_subq2)):
					samples_subq2.append(np.zeros((len_samples)))

			subq1_list.append(np.array(predicates_subq1_enc))
			subq1_mask_list.append(np.array(predicates_subq1_mask))

			subq2_list.append(np.array(predicates_subq2_enc))
			subq2_mask_list.append(np.array(predicates_subq2_mask))

			sample1_list.append(samples_subq1)
			sample2_list.append(samples_subq2)

			indicators.append(1)

	return subq1_list, subq2_list, subq1_mask_list, subq2_mask_list, indicators, sample1_list, sample2_list
