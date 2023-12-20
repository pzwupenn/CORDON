import torch
import numpy as np

def check_eligible_drop_pk(batch_predicates, batch_tables):
	queried_dim_table_list = []
	can_drop_queried_list = []
	for ori_tables, ori_predicate in zip(batch_tables, batch_predicates):
		has_pred_on_each_table = {}

		for table_name in ori_tables:
			t_alias = table_name.split(' ')[1]
			has_pred_on_each_table[t_alias] = False

		for a_predicate in ori_predicate:
			t_alias = a_predicate[0].split('.')[0]
			has_pred_on_each_table[t_alias] = True

		queried_dim_tables = []

		if len(ori_tables) == 1 and ori_tables[0] != 'store_sales ss':
			pass
		else:
			for table_name in ori_tables:
				t_alias = table_name.split(' ')[1]

				if has_pred_on_each_table[t_alias] is False:
					pass
				else:
					if t_alias != 'ss':
						queried_dim_tables.append(table_name)

		queried_dim_table_list.append(queried_dim_tables)
		if len(queried_dim_table_list):
			can_drop_queried_list.append(1)
		else:
			can_drop_queried_list.append(0)

	return queried_dim_table_list, can_drop_queried_list

def check_eligible_drop_useless_pk(batch_predicates, batch_tables):
	useless_table_list = []
	can_drop_list = []
	for ori_tables, ori_predicate in zip(batch_tables, batch_predicates):
		has_pred_on_each_table = {}

		for table_name in ori_tables:
			t_alias = table_name.split(' ')[1]
			has_pred_on_each_table[t_alias] = False

		for a_predicate in ori_predicate:
			t_alias = a_predicate[0].split('.')[0]
			has_pred_on_each_table[t_alias] = True

		useless_tables = []

		if len(ori_tables) == 1 and ori_tables[0] != 'store_sales ss':
			pass
		else:
			for table_name in ori_tables:
				t_alias = table_name.split(' ')[1]

				if has_pred_on_each_table[t_alias] is False:
					if t_alias != 'ss':
						useless_tables.append(table_name)

		useless_table_list.append(useless_tables)

		if len(useless_tables):
			can_drop_list.append(1)
		else:
			can_drop_list.append(0)

	return useless_table_list, can_drop_list

def check_eligible_add_pk(batch_predicates, batch_tables):
	table_can_added_list = []
	can_added_list = []
	for ori_tables, ori_predicate in zip(batch_tables, batch_predicates):
		possible_tables = ['date_dim d', 'customer_demographics cd',
		                   'household_demographics hd', 'store s']

		if len(ori_tables) == 1 and ori_tables[0] != 'store_sales ss':
			possible_tables = []
		else:
			for table_name in ori_tables:
				if table_name in possible_tables:
					possible_tables.remove(table_name)

		table_can_added_list.append(possible_tables)

		if len(table_can_added_list):
			can_added_list.append(1)
		else:
			can_added_list.append(0)
	return table_can_added_list, can_added_list

def check_eligible_add_filter(predicates, column2vec, numerical_cols,ori_predicates, ori_tables):
	len_colvec = len(column2vec)
	candi_cols = []
	for col_name in numerical_cols:
		col_vec = column2vec[col_name]
		col_id = np.where(col_vec == 1)[0][0]
		candi_cols.append(col_id)
	predicates = predicates.numpy()
	eligible = []
	for qid, query in enumerate(predicates):
		queried_cols = np.zeros((len_colvec))
		if ori_predicates[qid][0][0] != '':
			for predicate in ori_predicates[qid]:
				column = predicate[0]
				col_vec = column2vec[column]
				column_id = np.where(col_vec == 1)[0][0]
				queried_cols[column_id] = 1

		is_all_one = np.all(queried_cols == 1)

		if is_all_one:
			eligible.append(0)
		else:
			zero_indexes = np.where(queried_cols == 0)[0]
			intersection = [value for value in zero_indexes if value in candi_cols]
			ori_table = ori_tables[qid]
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
				eligible.append(0)
			else:
				eligible.append(1)
	return eligible