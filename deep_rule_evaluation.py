import copy
import os
import json
from json import JSONDecodeError
import pytz
from os import path
from datetime import datetime

from networkx.readwrite import json_graph

from graph.deep_rule import DeepRule
from graph.groot_algo import deep_groot
from graph.rca_algo import deep_rca_rank
from util import log as logging
import conf.constants as const

logger = logging.getLogger(const.ROOT_API)


class DeepRuleEvaluation:
    def __init__(self, catalog, dirname, weekINC='INC0181417'):
        self.dirname = dirname
        self.catalog = catalog
        # all cases later than INC0181417 that have timestamps
        # Only Print out the cases after weekINC
        self.weekINC = weekINC
        self.dataset_path = path.join(self.dirname, self.catalog, 'dataset')
        self.report_folder = path.join(self.dirname, self.catalog, 'report')

        self.not_found_list = ['INC0419163.json',
                               'INC0529818.json',
                               'INC0579093.json',
                               'INC0542291.json',
                               'INC0539966.json',
                               'INC0498854.json',
                               'INC0379871.json',
                               'INC0448093.json',
                               'INC0545653.json',
                               'INC0498021.json']

        self.training_split = 0.5
        self.rca_parameters = {}
        self.dp = DeepRule(self.catalog, self.rca_parameters)

    def train(self):
        logger.info("Start to write report...")
        """
        calculate response & write accuracy report
        """
        # Training
        training_data = []
        for dir, _, files in os.walk(self.dataset_path):
            sorted_files = sorted(files)
            print(sorted_files)
            for file in sorted_files[:int(0.1 * len(sorted_files))]:
                if file == '.DS_Store':
                    continue
                if file in self.not_found_list:
                    continue
                filePath = path.join(dir, file)
                print(file)
                with open(filePath) as json_file:
                    current_case = json.load(json_file)

                for key, data in current_case.items():
                    print(key)
                    if not (data['events'] and data['events']['poolEvents']):
                        continue
                    data = self.purify(data)
                    training_data.append(data)

        model_stored_path = {
        }
        if self.catalog in model_stored_path:
            self.dp.train(training_data, load_model_path=model_stored_path[self.catalog])
        else:
            self.dp.train(training_data, load_model_path=None)

    def test(self, debug_files=None):
        scores, rca, report_obj = {}, {}, {}
        # Testing
        for dir, _, files in os.walk(self.dataset_path):
            if debug_files is not None:
                testing_files = debug_files
            else:
                testing_files = files[int(self.training_split * len(files)):]
            for file in testing_files:
                if file == '.DS_Store':
                    continue
                if file in self.not_found_list:
                    continue
                filePath = path.join(dir, file)
                # print(file)
                with open(filePath) as json_file:
                    current_case = json.load(json_file)

                for key, data in current_case.items():
                    # print(key)
                    report_obj[key], grades, length, preLength = {}, {}, 0, 0
                    if not (data['events'] and data['events']['poolEvents']):
                        continue
                    res, _ = self.get_res(data, file)

                    untied_res = {}
                    for n in res:
                        untied_res[n] = [res[n], 1]
                    res = untied_res

                    # sort res based on scores from high to low
                    res = {k: v for k, v in sorted(res.items(), key=lambda item: item[1], reverse=True)}
                    for n in res:
                        res[n] = tuple(res[n])
                    print(res)
                    report_obj[key]['fullResponse'] = res

                    # default value, which means rca result failed to hit any rank in groot result
                    report_obj[key]['rank'] = len(res)

                    rca[key] = [data['RCAEvent'] + ',' + data['RCAPool']]
                    print(rca[key])
                    # find the rank for rca result in groot result
                    for k, v in res.items():
                        if not v in grades:
                            length += list(res.values()).count(v)
                            # grade is a set, to explain tied rank
                            grades[v] = length

                        if k in rca[key]:
                            report_obj[key]['rank'], report_obj[key]['length'] = grades[v], length
                            print(file, report_obj[key]['rank'], report_obj[key]['length'])
                            break
        # raise RuntimeError()
        # report file name & wirte report
        date_format = "%y-%m-%d_%H-%M-%S"
        timeString = datetime.now().astimezone(pytz.timezone('US/Pacific')).strftime(date_format)
        report_file = 'report_' + timeString + '_.json'
        report_path = path.join(self.report_folder, report_file)

        with open(report_path, 'w') as out_file:
            json.dump(report_obj, out_file, indent=4)

        rank1_list = []
        rank23_list = []
        failure_list = []
        missing_list = []
        for key, data in report_obj.items():
            if 'length' in data:
                if data['rank'] == 1 and data['length'] <= 5:
                    rank1_list.append(key)
                elif data['rank'] <= 3 and data['length'] <= 5:
                    rank23_list.append(key)
                else:
                    failure_list.append(key)
            else:
                missing_list.append(key)
                print("RCA of {} is missing".format(key))

        week_rank1 = [el for el in rank1_list if el >= self.weekINC]
        week_rank23 = [el for el in rank23_list if el >= self.weekINC]
        week_fail = [el for el in failure_list if el >= self.weekINC]
        week_report = {k: v for k, v in report_obj.items() if k >= self.weekINC}
        print('Report:')
        print(json.dumps(week_report, indent=4))

        print(f'For This Week {self.catalog}:')
        print(f'Total: {len(week_rank1) + len(week_rank23) + len(week_fail) + len(missing_list)}')
        print(f'Top1({len(week_rank1)}): {week_rank1}')
        print(f'Rank2 - 3({len(week_rank23)}): {week_rank23}')
        print(f'Rank > 3({len(week_fail)}): {week_fail}')
        print(f'Missing cases({len(missing_list)}): {missing_list}')

    def get_ranks(self):
        report_path, reports_path = '', path.join(self.dirname, self.catalog, 'report')
        for dir, _, reports in os.walk(reports_path):
            report_path = path.join(dir, max(reports))
        report = {}
        with open(report_path) as json_obj:
            try:
                report_obj = json.load(json_obj)
                for incident_num, contents in report_obj.items():
                    if incident_num.startswith('__'):
                        continue
                    report[incident_num] = (contents['rank'], contents['length'] if 'length' in contents else None)

                return report, report_obj

            except JSONDecodeError:
                logger.info("Can't find accuracy report")

    def purify(self, data):
        for p in data['events']['poolEvents']:
            for e in list(data['events']['poolEvents'][p].keys()):
                if e == 'Badbox':
                    del_flag = True
                    for details in data['events']['poolEvents'][p][e]:
                        for d in data['events']['poolEvents'][p][e][details]:
                            if d['severity'] > 10:
                                del_flag = False
                    if del_flag:
                        del data['events']['poolEvents'][p][e]

        for rc_event in ['adyen', 'paypal', 'afterpay']:
            upper_rc_event = str.upper(rc_event[0]) + rc_event[1:]
            # Merge ThirdParty into Issue
            if data['RCAPool'] == rc_event:
                # data['RCAEvent'] = 'Issue'
                data['RCAEvent'] = f'Third Party {upper_rc_event} Error'

            # Merge ThirdParty into Issue
            if rc_event in data['events']['poolEvents']:
                tmp_dict = {}
                for k, v in data['events']['poolEvents'][rc_event].items():
                    tmp_dict.update(copy.deepcopy(v))
                data['events']['poolEvents'][rc_event] = {}
                data['events']['poolEvents'][rc_event][data['RCAEvent']] = tmp_dict

        return data

    # INC0568174
    def get_res(self, data, data_name):
        data = self.purify(data)
        pool_events = data['events']['poolEvents']
        domain_events = data['events']['domainEvents']
        pool_list, subG_obj = data['subGraphPools'], data['subGraph']
        for node in subG_obj['nodes']:
            node['events'] = set(node['events'])
        subG = json_graph.node_link_graph(subG_obj, True, True)
        domain = None
        if self.catalog == const.DOMAIN:
            domain = data['domain']
        else:  # cs_md Case
            pool_list = data[const.KEY_GENERAL].split(",")

        def rca_algo(EG, **kwargs):
            return deep_rca_rank(EG, self.dp, **self.rca_parameters,
                                 data_name=data_name, **kwargs)

        res, _, _ = deep_groot(pool_events, domain_events, subG, pool_list, domain, self.catalog, rca_algo)
        return res, 0


if __name__ == "__main__":
    # general / domain
    dirname = os.path.abspath(__file__)
    for var in [const.DOMAIN, const.GENERAL]:
        f = DeepRuleEvaluation(var, dirname)
        f.train()
        f.test()
