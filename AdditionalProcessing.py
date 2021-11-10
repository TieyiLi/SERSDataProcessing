import numpy as np
from Utils import Utils


class AdditionalProcessing(Utils):

    def clustering_summary(self, cluster_pred):
        cluster_size_summary = []
        pred_set = set(cluster_pred)
        for s in pred_set:
            idx_s = [i for i in range(len(cluster_pred)) if cluster_pred[i] == s]
            cluster_size_summary.append(len(idx_s))
        return cluster_size_summary


    def relabel_by_purity(self, cluster_pred, y_train):
        new_label = np.zeros_like(y_train)
        total_cluster_num = 0
        total_interest_num = 0
        L = len(y_train)
        pred_set = set(cluster_pred)
        for s in pred_set:
            idx_s = [i for i in range(L) if cluster_pred[i] == s]
            if np.all(y_train[idx_s] == 1):
                new_label[idx_s] = 1
                total_cluster_num += 1
                total_interest_num += len(idx_s)

        print('%d/%d specific clusters' % (total_cluster_num, len(pred_set)))
        print('%d/%d interest instances' % (total_interest_num, L))

        return new_label


    def relabel_by_ratio(self, cluster_pred, ori_label, percentage=0.5):

        if set(ori_label) != {0, 1}:
            raise ValueError('Relabel can not be applied to multi-labels!')

        if not isinstance(cluster_pred, np.ndarray):
            cluster_pred = np.array(cluster_pred)
        if not isinstance(ori_label, np.ndarray):
            ori_label = np.array(ori_label)

        new_label = np.empty(shape=cluster_pred.shape)
        cluster_set = set(cluster_pred)
        for set_i in cluster_set:
            idx = [i for i in range(len(cluster_pred)) if cluster_pred[i] == set_i]

            group_label_i = ori_label[idx]
            types = list(set(group_label_i))
            if len(types) == 1:
                new_label[idx] = types[0]
            else:
                num_tot = len(group_label_i)
                num_p = len(group_label_i[group_label_i == 1])
                if num_p >= percentage * num_tot:
                    new_label[idx] = 1
                else:
                    new_label[idx] = 0

        return new_label

    def input_data(self, data_path, output='content'):
        data_dict = np.load(data_path, allow_pickle=True).item()
        if output == 'content':
            X = data_dict['data_matrix']
            label = data_dict['label']
            group = data_dict['group']
            sample_name = data_dict['sample_name']
            raman_shift = data_dict['raman_shift']
            map_index = data_dict['map_index']
            return X, label, group, sample_name, raman_shift, map_index
        elif output == 'dict':
            return data_dict
        else:
            raise ValueError('Option is not defined!')

    def test_summary_spec_base(self, test_individual_label, y_pred):
        individual = set(test_individual_label)
        pred_type = set(y_pred)
        for i in individual:
            print('Individual %d:' % i)
            idx = [j for j in range(len(test_individual_label)) if test_individual_label[j] == i]
            pred_i = y_pred[idx]
            for t in pred_type:
                num = len(pred_i[pred_i == t])
                print('    predicted as %d: %d' % (t, num), end='')
                print('(%.3f)' % (num / len(pred_i)))

    def test_summary_map_base(self, test_sample_label, map_index, y_pred):
        sample_type = set(test_sample_label)
        confusion_matrix = np.empty(shape=(len(sample_type), len(sample_type)), dtype=np.int32)
        s = 0
        for i in sample_type:
            content_summary = []
            print('Individual %d:' % i)
            sample_idx = [j for j in range(len(test_sample_label)) if test_sample_label[j] == i]
            pred_i = y_pred[sample_idx]

            map_for_sample = [map_index[s] for s in sample_idx]
            map_type = set(map_for_sample)
            for m in map_type:
                map_idx = [u for u in range(len(map_for_sample)) if map_for_sample[u] == m]
                pred_m = pred_i[map_idx]
                most_common_ele = np.argmax(np.bincount(pred_m))
                content_summary.append(most_common_ele)

            w = 0
            for n in set(content_summary):
                len_n = len([v for v in range(len(content_summary)) if content_summary[v] == n])
                confusion_matrix[s, w] = len_n
                w += 1
                print('    Prediction of %d: %d' % (int(n), len_n), end='')
                print('(%.3f)' % (len_n / len(content_summary)))
            s += 1
        return confusion_matrix
