import torch
import numpy as np

def check_eligible_add_pk(predicates, ori_tables):
	if torch.is_tensor(predicates):
		predicates = predicates.numpy()
	check_eligible = []

	for qid, query in enumerate(predicates):
		ori_table = ori_tables[qid]
		if len(ori_table) != 1 or ori_table[0] == 'title t':
			check_eligible.append(0)
			continue
		else:
			check_eligible.append(1)
	return  check_eligible


def check_eligible_drop_pk(has_pred_on_title_list, predicates, ori_tables):
	if torch.is_tensor(predicates):
		predicates = predicates.numpy()
	eligible = []
	for qid, query in enumerate(predicates):
		ori_table = ori_tables[qid]
		has_pred_on_title = has_pred_on_title_list[qid]
		if len(ori_table) == 1 or (has_pred_on_title is False):
			eligible.append(0)
		else:
			eligible.append(1)
	return eligible


def check_eligible_drop_pk_no_pred(has_pred_on_title_list, ori_tables):
	eligible = []
	for qid, tables in enumerate(ori_tables):
		ori_table = ori_tables[qid]
		has_pred_on_title = has_pred_on_title_list[qid]
		if len(ori_table) == 2 and (has_pred_on_title is False):
			eligible.append(1)
		else:
			eligible.append(0)
	return eligible

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