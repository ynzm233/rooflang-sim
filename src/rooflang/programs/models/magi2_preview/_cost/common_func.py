from typing import List

def get_level_op_size_stage(op_size: float, net_list:List[int]) -> List[float]:
    level_op_size = []
    for net_cnt in net_list:
        if net_cnt == 1:
            level_op_size.append(0)
            continue
        cur_op_size = op_size * (net_cnt-1) / net_cnt
        level_op_size.append(cur_op_size)
        op_size /= net_cnt
    return level_op_size

def get_level_op_size_mesh(each_size: float, net_list:List[int]) -> List[float]:
    level_op_size = []
    cur_cnt = 1
    for net_cnt in net_list:
        if net_cnt == 1:
            level_op_size.append(0)
            continue
        level_op_size.append(each_size * cur_cnt * (net_cnt - 1))
        cur_cnt = cur_cnt * net_cnt
    return level_op_size