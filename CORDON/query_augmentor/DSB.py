import torch
import numpy as np
import random
random.seed(42)

def add_pk(is_add_list, table_can_added_list, join2vec, num_joins, len_join, predicates, sample_encs, table2vec,
                ori_tables, ori_samples, return_all=False):

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
	qids_returns = []

	table2joins = {'date_dim d': 'd.d_date_sk=ss.ss_sold_date_sk',
	               'customer_demographics cd': 'cd.cd_demo_sk=ss.ss_cdemo_sk',
	               'household_demographics hd': 'hd.hd_demo_sk=ss.ss_hdemo_sk',
	               'store s': 's.s_store_sk=ss.ss_store_sk'}

	for qid, query in enumerate(predicates):

		if torch.is_tensor(query):
			query = query.numpy()

		tables_can_added = table_can_added_list[qid]

		if len(tables_can_added) == 0 or is_add_list[qid] == 0:
			samples_enc = list()
			predicates_enc = list()
			join_enc = list()

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
			qids_returns.append(qid)
			continue


		if not return_all:
			samples_enc = list()
			predicates_enc = list()
			join_enc = list()

			samples_enc_mask = list()
			predicates_enc_mask = list()
			join_enc_mask = list()

			ori_table = ori_tables[qid]
			ori_sample = ori_samples[qid]
			indicators.append(1)
			qids_returns.append(qid)

			chosen_table = random.choice(tables_can_added)

			new_join = table2joins[chosen_table]
			new_join = join2vec[new_join]
			join_enc.append(new_join)
			join_enc_mask.append([1.])

			if len(ori_table) >= 2: # has join before
				for t_name in ori_table:
					if  t_name != 'store_sales ss':
						new_join = table2joins[t_name]
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

			# add the added table
			sample_vec = []
			sample_vec.append(table2vec[chosen_table])
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
		else:
			for chosen_table in tables_can_added:
				samples_enc = list()
				predicates_enc = list()
				join_enc = list()

				samples_enc_mask = list()
				predicates_enc_mask = list()
				join_enc_mask = list()

				ori_table = ori_tables[qid]
				ori_sample = ori_samples[qid]
				indicators.append(1)
				qids_returns.append(qid)

				new_join = table2joins[chosen_table]
				new_join = join2vec[new_join]
				join_enc.append(new_join)
				join_enc_mask.append([1.])

				if len(ori_table) >= 2:  # has join before
					for t_name in ori_table:
						if t_name != 'store_sales ss':
							new_join = table2joins[t_name]
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

				# add the added table
				sample_vec = []
				sample_vec.append(table2vec[chosen_table])
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

	return predicate_list, sample_list, join_list,  predicate_mask_list, sample_mask_list, join_mask_list, indicators, qids_returns

def drop_useless_pk(is_drop_list, useless_table_list, predicates, sample_encs, table2vec, column2vec,
                    join2vec, num_joins, len_join, ori_tables, ori_samples, return_all=False):
	len_colvec = len(column2vec)

	vec2column = {tuple(value): key for key, value in column2vec.items()}

	if torch.is_tensor(predicates):
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

	indicators = []
	qids_returns = []

	for qid, query in enumerate(predicates):

		if torch.is_tensor(query):
			query = query.numpy()

		ori_table = ori_tables[qid]
		useless_tables = useless_table_list[qid] # full name!

		if len(useless_tables) == 0 or is_drop_list[qid]==0:
			samples_enc = list()
			joins_enc = list()
			predicates_enc = list()

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
			qids_returns.append(qid)
			continue

		if not return_all:
			samples_enc = list()
			joins_enc = list()
			predicates_enc = list()

			samples_enc_mask = list()
			joins_enc_mask = list()
			predicates_enc_mask = list()

			ori_sample = ori_samples[qid]
			chosen_table = random.choice(useless_tables)

			# the case for two tables, we first separate two subqueries
			# then, C(dimension table) >= C(join) should hold
			indicators.append(1)
			qids_returns.append(qid)

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
		else:
			for chosen_table in useless_tables:
				qids_returns.append(qid)
				samples_enc = list()
				joins_enc = list()
				predicates_enc = list()

				samples_enc_mask = list()
				joins_enc_mask = list()
				predicates_enc_mask = list()

				ori_sample = ori_samples[qid]

				# the case for two tables, we first separate two subqueries
				# then, C(dimension table) >= C(join) should hold
				indicators.append(1)

				if len(ori_table) == 2:  # just 1 after drop 1
					new_join = ''
					new_join = join2vec[new_join]

					joins_enc.append(new_join)
					joins_enc_mask.append([1.])
				else:  # >= 3 tables
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


	return predicate_list, join_list, sample_list, predicate_mask_list, join_mask_list, sample_mask_list, indicators, qids_returns
